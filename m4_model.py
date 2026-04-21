"""
Module 4.2: Model (HGT spatial + THP temporal + dual heads)
===========================================================
三块:
  1) HeteroEncoder: 2 层 HGT, 输入每个节点的 (static ⊕ dynamic) 特征,
     输出 route 节点的 contextualized embedding h_R ∈ ℝ^d.
  2) TemporalEncoder: causal Transformer, 吃过去 K 个事件 token,
     输出最后位置的 hidden state h_H ∈ ℝ^d.
  3) 双头:
     MarkHead:   用 h_H ⊕ mov 当 condition 给 h_R 做 FiLM, 输出 logits (B, N_R).
                  非法位置在 softmax 前 mask 成 -inf.
     TimeHead:   log-normal mixture (K=4 components), 输出 (μ, log_σ, log_w).
                  Loss = - log p(τ|ctx)

Batch 策略:
  静态 edge_index 在所有样本共享. 把 B 个样本的节点 stack 成 (B·N, d),
  edge_index 用 torch_geometric.data.Batch 风格偏移:
     edge_index_batch[b] = edge_index + b * N_per_type
  HGT 看到的还是一张"大图", 各样本的节点不互连.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


def stack_edge_index(edge_index, num_src, num_dst, B):
    """把一条边索引在 batch 维上复制并偏移."""
    # edge_index: (2, E)
    # 返回: (2, B*E), src 偏移 = b*num_src, dst 偏移 = b*num_dst
    E = edge_index.shape[1]
    device = edge_index.device
    offsets_src = torch.arange(B, device=device).repeat_interleave(E) * num_src
    offsets_dst = torch.arange(B, device=device).repeat_interleave(E) * num_dst
    rep = edge_index.repeat(1, B)
    rep[0] += offsets_src
    rep[1] += offsets_dst
    return rep


class HeteroEncoder(nn.Module):
    """2 层 HGT."""

    def __init__(self, node_feat_dims, metadata, d=128, n_layers=2, n_heads=4):
        """
        node_feat_dims: dict[str -> int]   每类节点的输入维度 (static + dynamic)
        metadata: (node_types, edge_types) — 由 HeteroData.metadata() 产生
        """
        super().__init__()
        self.node_types = metadata[0]
        self.proj = nn.ModuleDict({
            nt: nn.Linear(dim, d) for nt, dim in node_feat_dims.items()
        })
        self.convs = nn.ModuleList([
            HGTConv(d, d, metadata, heads=n_heads) for _ in range(n_layers)
        ])
        self.d = d

    def forward(self, x_dict, edge_index_dict):
        x = {nt: F.relu(self.proj[nt](x_dict[nt])) for nt in self.node_types}
        for conv in self.convs:
            x = conv(x, edge_index_dict)
            x = {nt: F.relu(h) for nt, h in x.items()}
        return x


class Time2Vec(nn.Module):
    def __init__(self, d):
        super().__init__()
        assert d >= 2
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(d - 1))
        self.b = nn.Parameter(torch.randn(d - 1))

    def forward(self, t):
        # t: (..., 1) or (...,)
        if t.dim() > 0 and t.shape[-1] != 1:
            t = t.unsqueeze(-1)
        v0 = self.w0 * t + self.b0
        vk = torch.sin(t * self.w + self.b)
        return torch.cat([v0, vk], dim=-1)


class TemporalEncoder(nn.Module):
    """Causal Transformer on event tokens."""

    def __init__(self, n_type, n_id, n_train, d=128, n_layers=3, n_heads=4, K=64):
        super().__init__()
        d_each = d // 4
        self.type_emb = nn.Embedding(n_type, d_each, padding_idx=0)
        self.id_emb = nn.Embedding(n_id, d_each, padding_idx=0)
        self.train_emb = nn.Embedding(n_train, d_each, padding_idx=0)
        self.time_emb = Time2Vec(d_each)
        # 三个 emb 加时间 embedding 后维度不一定整除, 用 Linear 投回 d
        self.proj = nn.Linear(d_each * 4, d)

        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=d * 4,
                                            batch_first=True, activation='gelu')
        self.tf = nn.TransformerEncoder(layer, n_layers)

        # 缓存的 causal mask
        self.register_buffer('causal_mask',
                             torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1))
        self.K = K

    def forward(self, events_cat, events_t):
        # events_cat: (B, K, 3) long   events_t: (B, K, 1) float
        te = self.type_emb(events_cat[..., 0])
        ie = self.id_emb(events_cat[..., 1])
        tre = self.train_emb(events_cat[..., 2])
        time_e = self.time_emb(events_t)   # (B, K, d_each)
        x = torch.cat([te, ie, tre, time_e], dim=-1)
        x = self.proj(x)
        h = self.tf(x, mask=self.causal_mask)
        return h[:, -1]  # 取最后一个位置作为 context (B, d)


class MarkHead(nn.Module):
    """FiLM-based scorer: score[r] = <h_route[r], gamma * ctx + beta>"""

    def __init__(self, d, mov_dim):
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Linear(d + mov_dim, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d * 2),  # -> gamma, beta
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(self, h_route_batched, ctx, legal_mask):
        """
        h_route_batched: (B, N_R, d)
        ctx: (B, d + mov_dim)
        legal_mask: (B, N_R) bool
        """
        gamma_beta = self.cond_proj(ctx)        # (B, 2d)
        gamma, beta = gamma_beta.chunk(2, dim=-1)   # (B, d) each
        modulated = h_route_batched * gamma.unsqueeze(1) + beta.unsqueeze(1)   # (B, N_R, d)
        logits = self.score_mlp(modulated).squeeze(-1)    # (B, N_R)
        logits = logits.masked_fill(~legal_mask, float('-inf'))
        # 特殊情况: 某些样本 mask 全 0 (n_legal=0), 此时 logits 全 -inf, log_softmax 会 NaN
        # 做保底: 至少把 label 位置学不到也要让模型不 NaN
        # 训练期 Dataset 已把 label 位置置 True; 但测试期可能真 n_legal=0 — 这种样本不参与 loss
        return logits


class TimeHead(nn.Module):
    """Log-normal mixture: log p(tau) = logsumexp_k [log w_k + log N(log(tau)|mu_k, sigma_k)] - log(tau)"""

    def __init__(self, d, n_mix=4):
        super().__init__()
        self.n_mix = n_mix
        self.proj = nn.Linear(d, n_mix * 3)  # mu, log_sigma, log_w

    def forward(self, ctx):
        params = self.proj(ctx)
        mu, log_sigma, log_w = params.chunk(3, dim=-1)  # (B, n_mix) each
        return mu, log_sigma, log_w

    def log_prob(self, tau, ctx):
        """tau: (B,) positive seconds (未取 log).  返回 (B,) log-prob."""
        mu, log_sigma, log_w = self.forward(ctx)
        # Guard against 0 or negative tau
        tau = torch.clamp(tau, min=1e-3)
        log_tau = torch.log(tau).unsqueeze(-1)        # (B, 1)
        z = (log_tau - mu) / torch.exp(log_sigma)     # (B, n_mix)
        log_normal = -0.5 * z ** 2 - log_sigma - 0.5 * math.log(2 * math.pi)
        log_w_norm = F.log_softmax(log_w, dim=-1)
        log_mix = torch.logsumexp(log_w_norm + log_normal, dim=-1)     # (B,)
        return log_mix - log_tau.squeeze(-1)   # log p(tau) = log p(log tau) - log tau


class DerbyModel(nn.Module):
    def __init__(self,
                 node_feat_dims,
                 metadata,
                 n_type, n_id, n_train,
                 mov_dim=23,
                 d=128, n_hgt_layers=2, n_tf_layers=3,
                 n_heads=4, K=64, n_mix=4):
        super().__init__()
        self.d = d
        self.spatial = HeteroEncoder(node_feat_dims, metadata,
                                      d=d, n_layers=n_hgt_layers, n_heads=n_heads)
        self.temporal = TemporalEncoder(n_type, n_id, n_train,
                                         d=d, n_layers=n_tf_layers,
                                         n_heads=n_heads, K=K)
        self.mark_head = MarkHead(d, mov_dim)
        self.time_head = TimeHead(d, n_mix=n_mix)

        # 保存节点数, 用来做 batch 偏移
        self.node_types = metadata[0]
        self.edge_types = metadata[1]

    def _batched_x_dict(self, batch, static_x):
        """
        batch: dict from DataLoader
        static_x: dict[nt -> (N_nt, d_static)]

        返回: x_dict with shape (B*N_nt, d_full) — 先 stack 再 concat static.
        """
        B = batch['dyn_berth'].shape[0]
        x_dict = {}
        dyn_map = {
            'berth': batch['dyn_berth'],
            'tc': batch['dyn_tc'],
            'route': batch['dyn_route'],
            'signal': batch['dyn_signal'],
        }
        for nt in self.node_types:
            dyn = dyn_map[nt]  # (B, N_nt, d_dyn)
            N_nt = dyn.shape[1]
            stat = static_x[nt].unsqueeze(0).expand(B, -1, -1)   # (B, N, d_static)
            full = torch.cat([stat, dyn], dim=-1)                # (B, N, d_full)
            x_dict[nt] = full.reshape(B * N_nt, -1)              # (B*N, d_full)
        return x_dict, B

    def _batched_edge_index(self, edge_index_dict, num_nodes_per_type, B):
        out = {}
        for et, ei in edge_index_dict.items():
            src_type, _, dst_type = et
            out[et] = stack_edge_index(ei, num_nodes_per_type[src_type],
                                        num_nodes_per_type[dst_type], B)
        return out

    def forward(self, batch, static_x, edge_index_dict, num_nodes_per_type):
        # 1) 批处理空间分支
        x_dict, B = self._batched_x_dict(batch, static_x)
        ei_dict = self._batched_edge_index(edge_index_dict, num_nodes_per_type, B)
        h_dict = self.spatial(x_dict, ei_dict)   # 每类 (B*N_nt, d)

        N_R = num_nodes_per_type['route']
        h_route = h_dict['route'].reshape(B, N_R, -1)

        # 2) 时间分支
        h_H = self.temporal(batch['events_cat'], batch['events_t'])   # (B, d)

        # 3) 双头
        ctx = torch.cat([h_H, batch['mov']], dim=-1)
        logits = self.mark_head(h_route, ctx, batch['legal_mask'])
        time_params = self.time_head(h_H)   # (mu, log_sigma, log_w)

        return dict(logits=logits, h_H=h_H, h_route=h_route,
                    time_params=time_params)
