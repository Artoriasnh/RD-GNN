import argparse
from pathlib import Path
import numpy as np


def convert_graph_states(samples_dir: Path):
    src = samples_dir / 'graph_states.npz'
    if not src.exists():
        print(f'[skip] missing {src.name}')
        return
    dsts = {
        'berth': samples_dir / 'graph_states_berth.npy',
        'tc': samples_dir / 'graph_states_tc.npy',
        'route': samples_dir / 'graph_states_route.npy',
        'signal': samples_dir / 'graph_states_signal.npy',
    }
    if all(p.exists() for p in dsts.values()):
        print('[skip] graph_states already converted')
        return
    print(f'[load] {src}')
    z = np.load(src)
    try:
        for key, dst in dsts.items():
            print(f'  [save] {dst.name}  shape={z[key].shape} dtype={z[key].dtype}')
            np.save(dst, z[key], allow_pickle=False)
    finally:
        z.close()


def convert_named_npz(samples_dir: Path, src_name: str, key: str, dst_name: str):
    src = samples_dir / src_name
    dst = samples_dir / dst_name
    if not src.exists():
        print(f'[skip] missing {src.name}')
        return
    if dst.exists():
        print(f'[skip] {dst.name} already exists')
        return
    print(f'[load] {src}')
    z = np.load(src)
    try:
        arr = z[key]
        print(f'  [save] {dst.name}  shape={arr.shape} dtype={arr.dtype}')
        np.save(dst, arr, allow_pickle=False)
    finally:
        z.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples-dir', required=True)
    args = ap.parse_args()

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        raise FileNotFoundError(samples_dir)

    convert_graph_states(samples_dir)
    convert_named_npz(samples_dir, 'event_windows.npz', 'X', 'event_windows_X.npy')
    convert_named_npz(samples_dir, 'legal_masks.npz', 'M', 'legal_masks_M.npy')
    convert_named_npz(samples_dir, 'mov_features.npz', 'X', 'mov_features_X.npy')

    print('\n[done] conversion finished')
    print('You can now use the updated m4_dataset.py which prefers .npy memmap files.')


if __name__ == '__main__':
    main()
