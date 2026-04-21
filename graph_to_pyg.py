"""
把 Module 1 产出的 graph.pkl 转成 PyG HeteroData 对象
(仅在本地 PyG 环境下用)
"""
import pickle, argparse


def to_hetero_data(graph_pkl):
    import torch
    from torch_geometric.data import HeteroData
    g = pickle.load(open(graph_pkl, 'rb'))
    h = HeteroData()
    for nt, feat in g['node_features'].items():
        h[nt].x = torch.tensor(feat, dtype=torch.float32)
        h[nt].num_nodes = feat.shape[0]
    for et, ei in g['edge_index'].items():
        h[et].edge_index = torch.tensor(ei, dtype=torch.long)
    print(h)
    return h, g   # g 里还有 legal_routes_from_berth / berth_reach 等字典, 需要一并带走


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', required=True)
    ap.add_argument('--save', default=None)
    a = ap.parse_args()
    h, g = to_hetero_data(a.graph)
    if a.save:
        import torch
        torch.save({'hetero': h, 'meta': {k: g[k] for k in
                    ['node_ids','id_to_idx','legal_routes_from_berth',
                     'route_tcs','route_end_berth','berth_reach']}}, a.save)
        print(f'saved to {a.save}')
