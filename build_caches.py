"""
CLIME 特征序列缓存构建。对应报告 Section 3.1 (时间划分)。

从 normalized_features.parquet 构建 L=40 滑动窗口序列，
按 train/val/holdout 三段时间切分存储。

用法:
  python build_caches.py --split val_v5     # 单个
  python build_caches.py --split all         # 全部
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.data.dataset import build_sequences_from_parquet

# V5 数据划分
SPLITS = {
    "train_v5":   ("20160104", "20250917", 10),   # training, step=10
    "val_v5":     ("20250918", "20251204", 1),    # validation, step=1
    "holdout_v5": ("20251205", "20260511", 1),    # holdout, step=1
}

L_VAL = 40


def build_split(name: str):
    start, end, step = SPLITS[name]
    print(f"\n{'='*60}")
    print(f"Building: {name}  [{start} ~ {end}]  L={L_VAL}  step={step}")
    print(f"{'='*60}")
    seqs, rets, codes = build_sequences_from_parquet(
        start, end, name, L_val=L_VAL, step=step, extra_feature_fn=None,
    )
    n_dates = len(seqs)
    n_samples = sum(len(v) for v in seqs.values())
    print(f"  Done: {n_dates} dates, {n_samples:,} samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "train_v5", "val_v5", "holdout_v5"])
    args = parser.parse_args()

    if args.split == "all":
        for name in ["train_v5", "val_v5", "holdout_v5"]:
            build_split(name)
    else:
        build_split(args.split)

    print("\n" + "=" * 60)
    print("All caches built.")
    import os
    for name in SPLITS:
        from src.data.dataset import _cache_path_for
        p = _cache_path_for(name, L_VAL, SPLITS[name][2])
        if p.exists():
            size_gb = p.stat().st_size / 1024**3
            print(f"  {p.name}: {size_gb:.1f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
