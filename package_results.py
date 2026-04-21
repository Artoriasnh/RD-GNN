"""
package_results.py — 把训练结果打成 tar.gz 方便回传
用法:
    python package_results.py m4_runs/20260421_143022_full
"""
import sys, os, tarfile


def main():
    if len(sys.argv) < 2:
        print("Usage: python package_results.py <run_dir>")
        sys.exit(1)

    run_dir = sys.argv[1].rstrip('/\\')
    if not os.path.isdir(run_dir):
        print(f"ERROR: {run_dir} is not a directory")
        sys.exit(1)

    base = os.path.basename(run_dir)
    parent = os.path.dirname(run_dir) or '.'
    out_file = f'{base}_results.tar.gz'

    keep = ['best.pt', 'history.json', 'train.log']
    print(f"Packaging {run_dir} -> {out_file}")
    print("Contents:")
    for f in os.listdir(run_dir):
        fp = os.path.join(run_dir, f)
        sz = os.path.getsize(fp) / 1e6
        flag = 'included' if f in keep else 'skipped'
        print(f"  [{flag}] {f}  ({sz:.2f} MB)")

    with tarfile.open(out_file, 'w:gz') as tar:
        for f in keep:
            fp = os.path.join(run_dir, f)
            if os.path.exists(fp):
                tar.add(fp, arcname=os.path.join(base, f))

    sz = os.path.getsize(out_file) / 1e6
    print(f"\nDone: {out_file}  ({sz:.2f} MB)")
    print("Upload this file to share results.")


if __name__ == '__main__':
    main()
