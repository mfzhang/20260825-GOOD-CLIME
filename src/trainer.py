"""
trainer.py — 训练循环 + validation 指标 + early stopping + checkpoint。

外部接口:
  trainer = Trainer(model, train_ds, val_data, cfg)
  trainer.fit()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PairDataset
from src.losses import pairwise_ranking_loss

# 默认输出目录
_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


class _LogTee:
    """同时输出到 stdout 和 log 文件。"""
    def __init__(self, filepath: Path):
        self.file = open(filepath, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_ds: PairDataset,
        val_x: np.ndarray,
        val_ret: np.ndarray,
        val_date_ids: np.ndarray,
        val_date_list: List[str],
        cfg: Optional[dict] = None,
        log_file: Optional[Path] = None,
    ):
        self.model = model
        self.train_ds = train_ds
        self.val_x = torch.from_numpy(val_x).float()
        self.val_ret = val_ret
        self.val_date_ids = val_date_ids
        self.val_date_list = val_date_list
        self.log_file = log_file
        self._tee: Optional[_LogTee] = None

        c = cfg or {}
        self.lr = c.get("lr", 1e-3)
        self.weight_decay = c.get("weight_decay", 1e-5)
        self.batch_size = c.get("batch_size", 128)
        self.max_epochs = c.get("max_epochs", 20)
        self.patience = c.get("patience", 2)
        self.grad_clip = c.get("grad_clip", 1.0)
        self.model_name = c.get("model_name", "model")
        self.output_dir = Path(c.get("output_dir", str(_OUTPUT_DIR / self.model_name)))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_epochs)

        self.history: List[Dict] = []
        self.best_metric = -float("inf")
        self.best_epoch = -1

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def fit(self) -> Dict:
        """完整训练流程。返回 best checkpoint 信息。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        if self.log_file is not None:
            self._tee = _LogTee(self.log_file)
            sys.stdout = self._tee

        print(f"\n{'='*60}")
        print(f"Training {self.model_name} | device={self.device} | "
              f"epochs={self.max_epochs} | patience={self.patience}")
        print(f"Log file: {self.log_file}")
        print(f"{'='*60}")

        wait = 0  # early stopping counter

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()

            # ---- Train ----
            self.train_ds.resample()
            loader = DataLoader(self.train_ds, batch_size=self.batch_size,
                                shuffle=True, num_workers=4, drop_last=True,
                                persistent_workers=True)
            train_loss = self._train_one_epoch(loader)

            # ---- Validate ----
            val_metrics = self._validate()

            # ---- Scheduler ----
            self.scheduler.step()

            elapsed = time.time() - t0
            self._log_epoch(epoch, train_loss, val_metrics, elapsed)

            # ---- Early stopping ----
            primary = val_metrics["top20_excess_ret"]
            if primary > self.best_metric:
                self.best_metric = primary
                self.best_epoch = epoch
                wait = 0
                self._save_checkpoint("best.pt")
                print(f"  ★ New best (Top-20 excess: {primary:.4f})")
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs)")
                    break

        # ---- 保存训练历史 ----
        history_path = self.output_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining done. Best epoch={self.best_epoch}, "
              f"Top-20 excess={self.best_metric:.4f}")
        print(f"History saved to {history_path}")

        if self._tee is not None:
            sys.stdout = self._tee.stdout
            self._tee.close()
        return {"best_epoch": self.best_epoch, "best_metric": self.best_metric}

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        lr = self.optimizer.param_groups[0]["lr"]
        pbar = tqdm(loader, desc=f"  Train", unit="batch", leave=False)
        for xi, xj, y in pbar:
            xi = xi.to(self.device)
            xj = xj.to(self.device)
            y = y.to(self.device)

            si = self.model(xi)
            sj = self.model(xj)
            loss = pairwise_ranking_loss(si, sj, y)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})

        return total_loss / max(n_batches, 1)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    @torch.no_grad()
    def _validate(self) -> Dict:
        self.model.eval()
        # 分批推理避免 OOM
        scores_list = []
        B = 4096
        for start in range(0, len(self.val_x), B):
            batch = self.val_x[start:start + B].to(self.device)
            scores_list.append(self.model(batch).cpu().numpy())
        scores = np.concatenate(scores_list)

        # 按天计算指标
        rank_ics = []
        top10_excess = []
        top20_excess = []
        top30_excess = []
        spreads = []

        for d_idx in range(len(self.val_date_list)):
            mask = self.val_date_ids == d_idx
            if mask.sum() < 10:
                continue

            day_scores = scores[mask]
            day_rets = self.val_ret[mask]

            # RankIC (Spearman)
            ic = self._spearman_corr(day_scores, day_rets)
            if not np.isnan(ic):
                rank_ics.append(ic)

            # Top-K excess return
            mean_ret = day_rets.mean()
            for k, lst in [(10, top10_excess), (20, top20_excess), (30, top30_excess)]:
                top_idx = np.argsort(day_scores)[-k:]
                top_ret = day_rets[top_idx].mean()
                lst.append(top_ret - mean_ret)

            # Top-bottom decile spread
            n = len(day_scores)
            decile_size = max(1, n // 10)
            sorted_idx = np.argsort(day_scores)
            bot_ret = day_rets[sorted_idx[:decile_size]].mean()
            top_ret = day_rets[sorted_idx[-decile_size:]].mean()
            spreads.append(top_ret - bot_ret)

        metrics = {
            "rank_ic_mean": np.mean(rank_ics) if rank_ics else 0.0,
            "rank_ic_std": np.std(rank_ics) if len(rank_ics) > 1 else 0.0,
            "rank_icir": (np.mean(rank_ics) / (np.std(rank_ics) + 1e-8)) if len(rank_ics) > 1 else 0.0,
            "top10_excess_ret": np.mean(top10_excess) if top10_excess else 0.0,
            "top20_excess_ret": np.mean(top20_excess) if top20_excess else 0.0,
            "top30_excess_ret": np.mean(top30_excess) if top30_excess else 0.0,
            "top_bottom_spread": np.mean(spreads) if spreads else 0.0,
        }
        if len(rank_ics) > 1:
            metrics["rank_icir"] = metrics["rank_ic_mean"] / (metrics["rank_ic_std"] + 1e-8)
        return metrics

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    @staticmethod
    def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
        """简单 Spearman 相关系数。"""
        from scipy.stats import spearmanr
        r, _ = spearmanr(x, y)
        return r

    def _log_epoch(self, epoch: int, train_loss: float,
                   val_metrics: Dict, elapsed: float):
        entry = {"epoch": epoch, "train_loss": train_loss,
                 "elapsed_sec": round(elapsed, 1), **val_metrics}
        self.history.append(entry)

        lr = self.optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | "
              f"RankIC={val_metrics['rank_ic_mean']:.4f} | "
              f"Top20_exc={val_metrics['top20_excess_ret']:.4f} | "
              f"Spread={val_metrics['top_bottom_spread']:.4f} | "
              f"lr={lr:.6f} | {elapsed:.1f}s")

    def _save_checkpoint(self, filename: str):
        path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
        }, path)
