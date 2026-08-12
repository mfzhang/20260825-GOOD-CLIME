"""
CLIME 模型训练入口。对应报告 Section 2.4 (Training Pipeline)。

用法:
  # Stage 1: Backbone 预训练（报告 Section 2.4.1）
  python train.py --stage1

  # Stage 2: CLIME 完整训练（报告 Section 2.4.2，需要先完成 Stage 1）
  python train.py --clime
  python train.py --clime --init-scale 0.3
  python train.py --clime --init-scale 0.3 --epochs 25
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.data.dataset import (
    PairDataset, build_validation_arrays,
    RegressionPeerDataset, build_regression_peer_validation_arrays,
    load_sequences, _cache_path_for, L as DEFAULT_L,
)
from src.models.transformer import TransformerPredictor
from src.models.v9ca_ab import CLIMEModel
from src.losses import pairwise_ranking_loss, directional_regression_loss
from src.trainer import Trainer

# ---- 输出目录 ----
OUTPUT_DIR = project_root / "output" / "transformer_v5"

# ---- 数据划分 ----
SPLITS = {
    "train_v5":   ("20160104", "20250917", 10),
    "val_v5":     ("20250918", "20251204", 1),
    "holdout_v5": ("20251205", "20260511", 1),
}

L_VAL = 40
_CACHE_DIR = Path(__file__).resolve().parent / "cache"

# ---- 默认超参 ----
CFG_STAGE1 = {
    "lr": 5e-4,
    "weight_decay": 1e-5,
    "batch_size": 2048,
    "max_epochs": 40,
    "patience": 10,
    "grad_clip": 1.0,
    "pairs_per_day": 5000,
    "margin_return": 0.002,
    "model_name": "transformer_v5_stage1",
}

# CLIME Stage 2 训练配置。对应报告 Section 2.4.2 (Table 1 + 附录 H)。
CFG_CLIME = {
    "lr_phase1": 1e-3,
    "lr_phase2_backbone": 1e-5,
    "lr_phase2_modulator": 5e-4,
    "lr_phase2_head": 5e-4,
    "lr_phase3_backbone": 1e-6,
    "lr_phase3_modulator": 5e-5,
    "lr_phase3_head": 1e-5,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "max_epochs": 25,
    "patience": 8,
    "grad_clip": 1.0,
    "samples_per_day": 5000,
    "model_name": "clime",
    "loss_alpha": 3.0,
    "loss_beta": 0.5,
    "loss_delta": 0.01,
    "phase1_epochs": 3,
    "phase2_epochs": 12,
    "K": 4,
    "d_factor": 64,
    "n_encoder_layers": 3,
    "init_scale": 0.1,
    "gate_hidden": 64,
}


# ===========================================================================
# Stage 1: Backbone 预训练
# ===========================================================================

def train_stage1(args):
    """Stage 1: 纯 Transformer backbone 预训练。"""
    cfg = dict(CFG_STAGE1)
    cfg["output_dir"] = str(OUTPUT_DIR)

    feat_dim_override = getattr(args, "feat_dim", None)
    if feat_dim_override:
        cfg["model_name"] = f"transformer_v12_s1_{feat_dim_override}d"
    else:
        cfg["model_name"] = "transformer_v5_stage1"

    if args.epochs:
        cfg["max_epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage 1: Pure Backbone Training")
    print("=" * 60)

    train_cache = _cache_path_for("train_v5", L_VAL, SPLITS["train_v5"][2])
    val_cache = _cache_path_for("val_v5", L_VAL, SPLITS["val_v5"][2])

    if not train_cache.exists():
        print(f"[error] Train cache not found: {train_cache}")
        print("  Run: python build_caches.py --split train_v5")
        sys.exit(1)
    if not val_cache.exists():
        print(f"[error] Val cache not found: {val_cache}")
        print("  Run: python build_caches.py --split val_v5")
        sys.exit(1)

    print("\nLoading data...")
    train_seqs, train_rets, train_codes = load_sequences("train_v5", L_val=L_VAL, step=SPLITS["train_v5"][2])
    val_seqs, val_rets, _ = load_sequences("val_v5", L_val=L_VAL, step=SPLITS["val_v5"][2])

    # Slice features if cache dim > requested feat_dim
    current_dim = train_seqs[sorted(train_seqs.keys())[0]].shape[-1]
    if feat_dim_override and current_dim > feat_dim_override:
        if feat_dim_override == 67:
            feat_idx = list(range(66)) + [86]  # 66 original base + lag_norm at dim 86
        elif feat_dim_override == 72:
            feat_idx = list(range(66)) + [86] + [75, 77, 78, 80, 81]
        else:
            feat_idx = list(range(feat_dim_override))
        print(f"  Slicing features: {current_dim} -> {feat_dim_override}")
        for d in train_seqs:
            train_seqs[d] = train_seqs[d][:, :, feat_idx]
        for d in val_seqs:
            val_seqs[d] = val_seqs[d][:, :, feat_idx]

    train_ds = PairDataset(
        train_seqs, train_rets, train_codes,
        pairs_per_day=cfg["pairs_per_day"],
        margin_return=cfg["margin_return"],
    )
    val_x, val_ret, val_ids, val_list = build_validation_arrays(val_seqs, val_rets)

    n_train = sum(len(v) for v in train_seqs.values())
    print(f"\nTrain: {len(train_seqs)} dates, {n_train:,} samples, {len(train_ds)} pairs")
    print(f"Val: {len(val_seqs)} dates, {val_x.shape[0]:,} samples")

    feat_dim = val_x.shape[-1]
    model = TransformerPredictor(
        seq_len=L_VAL, feat_dim=feat_dim,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, ffn_hidden=args.ffn_hidden,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Transformer | d_model={args.d_model} | n_layers={args.n_layers} | params={n_params:,}")

    if args.resume:
        print(f"[resume] Loading weights from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    log_file = OUTPUT_DIR / f"stage1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    trainer = Trainer(model, train_ds, val_x, val_ret, val_ids, val_list, cfg, log_file=log_file)
    result = trainer.fit()

    print(f"\nStage 1 Done. Best epoch={result['best_epoch']}, Top-20 excess={result['best_metric']:.6f}")

    if feat_dim_override:
        src = OUTPUT_DIR / "best.pt"
        dst = OUTPUT_DIR / f"stage1_v12_{feat_dim_override}d_best.pt"
        if src.exists():
            src.rename(dst)
            print(f"  Checkpoint saved as: {dst}")

    return result


# ===========================================================================
# CLIME Stage 2 三阶段课程训练器。对应报告 Section 2.4.2。
# ===========================================================================

class CLIMETrainer:
    """CLIME Stage 2 三阶段课程训练器。对应报告 Section 2.4.2 (Table 1)。

    Phase 1 (BCE Warmup,  epochs 1-3):  Backbone 冻结, BCE 方向预热
    Phase 2 (Core,        epochs 4-15): Backbone 解冻 (LR=1e-5), DirectionalReg
    Phase 3 (Fine-tune,   epochs 16-25):全部 LR 降低, 精细收敛

    支持 V9C 族模型（encoder, factor_heads, temporal_encoder, fusion 等）。
    """

    def __init__(self, model, train_ds, val_x, val_dyn, val_ret, val_ids,
                 val_dates, cfg, log_file=None):
        self.model = model
        self.train_ds = train_ds
        self.val_x = val_x
        self.val_dyn = val_dyn
        self.val_ret = val_ret
        self.val_ids = val_ids
        self.val_dates = val_dates
        self.cfg = cfg
        self.log_file = log_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.val_x = self.val_x.to(self.device)
        self.val_dyn = self.val_dyn.to(self.device)
        self.max_epochs = cfg["max_epochs"]
        self.batch_size = cfg["batch_size"]
        self.grad_clip = cfg["grad_clip"]
        self.patience = cfg["patience"]
        self.history: List[Dict] = []
        self.best_metric = -float("inf")
        self.best_epoch = -1

    def _get_phase(self, epoch):
        if epoch <= self.cfg["phase1_epochs"]:
            return 1
        elif epoch <= self.cfg["phase1_epochs"] + self.cfg["phase2_epochs"]:
            return 2
        return 3

    def _has_encoder(self):
        return hasattr(self.model, 'encoder') and \
               len(list(self.model.encoder.parameters())) > 0

    def _has_factor_heads(self):
        return hasattr(self.model, 'factor_heads') and \
               len(list(self.model.factor_heads.parameters())) > 0

    def _configure_optimizer(self, phase):
        param_groups = []

        # Backbone
        bb_params = list(self.model.backbone.parameters())
        if phase == 1:
            for p in bb_params:
                p.requires_grad = False
        else:
            for p in bb_params:
                p.requires_grad = True
            bb_lr = self.cfg.get(f"lr_phase{phase}_backbone",
                                self.cfg.get("lr_phase2_backbone", 1e-5))
            param_groups.append({"params": bb_params, "lr": bb_lr})

        # Encoder
        if self._has_encoder():
            enc_params = list(self.model.encoder.parameters())
            for p in enc_params:
                p.requires_grad = True
            enc_lr = self.cfg.get(f"lr_phase{phase}_modulator",
                                  self.cfg.get("lr_phase2_modulator", 5e-4))
            if phase != 1:
                param_groups.append({"params": enc_params, "lr": enc_lr})

        # temporal_encoder + fusion (V9CD)
        if hasattr(self.model, 'temporal_encoder'):
            for p in self.model.temporal_encoder.parameters():
                p.requires_grad = True
            if phase != 1:
                param_groups.append(
                    {"params": self.model.temporal_encoder.parameters(),
                     "lr": self.cfg.get("lr_phase2_modulator", 5e-4)})
        if hasattr(self.model, 'fusion'):
            for p in self.model.fusion.parameters():
                p.requires_grad = True
            if phase != 1:
                param_groups.append(
                    {"params": self.model.fusion.parameters(),
                     "lr": self.cfg.get("lr_phase2_modulator", 5e-4)})

        # Factor heads
        if self._has_factor_heads():
            fh_params = list(self.model.factor_heads.parameters())
            for p in fh_params:
                p.requires_grad = True
            fh_lr = self.cfg.get(f"lr_phase{phase}_modulator",
                                 self.cfg.get("lr_phase2_modulator", 5e-4))
            if phase != 1:
                param_groups.append({"params": fh_params, "lr": fh_lr})

        # Head
        head_params = list(self.model.head.parameters())
        for p in head_params:
            p.requires_grad = True
        head_lr = self.cfg.get(f"lr_phase{phase}_head",
                               self.cfg.get("lr_phase2_head", 5e-4))
        param_groups.append({"params": head_params, "lr": head_lr})

        # Phase 1 special: only trainable params
        if phase == 1:
            phase1_params = []
            for attr in ['encoder', 'temporal_encoder', 'fusion',
                         'factor_heads', 'head']:
                if hasattr(self.model, attr):
                    m = getattr(self.model, attr)
                    if isinstance(m, nn.Module):
                        phase1_params.extend(list(m.parameters()))
            if not phase1_params:
                phase1_params = list(self.model.head.parameters())
            self.optimizer = Adam(phase1_params,
                                  lr=self.cfg.get("lr_phase1", 1e-3),
                                  weight_decay=self.cfg["weight_decay"])
        else:
            self.optimizer = Adam(param_groups, weight_decay=self.cfg["weight_decay"])

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_epochs)

    def fit(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_name = self.cfg.get("model_name", "clime")
        n_bb = sum(p.numel() for p in self.model.backbone.parameters())
        n_head = sum(p.numel() for p in self.model.head.parameters())
        n_enc = sum(p.numel() for p in self.model.encoder.parameters()) \
            if self._has_encoder() else 0
        n_other = sum(p.numel() for p in self.model.parameters()) - n_bb - n_head - n_enc

        print(f"\n{'='*60}")
        print(f"CLIME ({model_name}): 3-Phase Curriculum Training")
        print(f"  device={self.device} | epochs={self.max_epochs}")
        print(f"  Params: backbone={n_bb:,} | encoder={n_enc:,} | "
              f"head={n_head:,} | other={n_other:,}")
        print(f"  Phase 1 (BCE Warmup, 报告 Section 2.4.2): epoch 1-{self.cfg['phase1_epochs']}")
        print(f"  Phase 2 (Core,      报告 Section 2.4.2): epoch {self.cfg['phase1_epochs']+1}"
              f"-{self.cfg['phase1_epochs']+self.cfg['phase2_epochs']}")
        print(f"  Phase 3 (Fine-tune, 报告 Section 2.4.2): epoch "
              f"{self.cfg['phase1_epochs']+self.cfg['phase2_epochs']+1}"
              f"-{self.max_epochs}")
        print(f"{'='*60}")

        wait = 0
        current_phase = 0
        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            phase = self._get_phase(epoch)
            if phase != current_phase:
                current_phase = phase
                self._configure_optimizer(phase)
                self.train_ds.resample()
                print(f"\n  --- Phase {phase} start ---")
            loader = DataLoader(
                self.train_ds, batch_size=self.batch_size,
                shuffle=True, num_workers=4, drop_last=True,
                persistent_workers=True,
            )
            train_loss = self._train_epoch(loader, phase)
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            val_metrics = self._validate()
            elapsed = time.time() - t0
            self._log_epoch(epoch, phase, train_loss, val_metrics, elapsed, lr)
            primary = val_metrics["top20_excess_ret"]
            if primary > self.best_metric:
                self.best_metric = primary
                self.best_epoch = epoch
                wait = 0
                self._save_checkpoint(f"{model_name}_best.pt")
                print(f"  [new best] Top-20 excess: {primary:.6f}")
            else:
                wait += 1
                if wait >= self.patience and epoch > self.cfg["phase1_epochs"]:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        history_path = OUTPUT_DIR / f"{model_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nCLIME Stage 2 Done. Best epoch={self.best_epoch}, "
              f"Top-20 excess={self.best_metric:.6f}")
        return {"best_epoch": self.best_epoch, "best_metric": self.best_metric}

    def _train_epoch(self, loader, phase):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        alpha = self.cfg["loss_alpha"]
        beta = self.cfg["loss_beta"]
        delta = self.cfg["loss_delta"]
        desc = {1: "Train(BCE)", 2: "Train(Reg)", 3: "Train(FT)"}[phase]
        for x, dyn, true_ret in tqdm(loader, desc=f"  {desc}", unit="batch",
                                      leave=False):
            x = x.to(self.device)
            dyn = dyn.to(self.device)
            true_ret = true_ret.to(self.device)
            pred = self.model(x, dyn)
            if phase == 1:
                target = (true_ret > 0).float()
                loss = F.binary_cross_entropy_with_logits(pred, target)
            else:
                loss = directional_regression_loss(pred, true_ret, alpha, beta, delta)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        B = 4096
        scores = []
        for start in range(0, len(self.val_x), B):
            end = min(start + B, len(self.val_x))
            scores.append(
                self.model(self.val_x[start:end],
                           self.val_dyn[start:end]).cpu().numpy()
            )
        scores = np.concatenate(scores)
        rank_ics, top10_exc, top20_exc, top30_exc, spreads = [], [], [], [], []
        sign_correct, rmse_vals = [], []
        for d_idx in range(len(self.val_dates)):
            mask = self.val_ids == d_idx
            if mask.sum() < 10:
                continue
            day_scores = scores[mask]
            day_rets = self.val_ret[mask]
            ic = np.corrcoef(day_scores, day_rets)[0, 1]
            if not np.isnan(ic):
                rank_ics.append(ic)
            mean_ret = day_rets.mean()
            for k, lst in [(10, top10_exc), (20, top20_exc), (30, top30_exc)]:
                top_idx = np.argsort(day_scores)[-k:]
                lst.append(day_rets[top_idx].mean() - mean_ret)
            n = len(day_scores)
            decile = max(1, n // 10)
            sorted_idx = np.argsort(day_scores)
            spreads.append(
                day_rets[sorted_idx[-decile:]].mean() -
                day_rets[sorted_idx[:decile]].mean()
            )
            sign_correct.append(
                np.mean(np.sign(day_scores) == np.sign(day_rets))
            )
            rmse_vals.append(np.sqrt(np.mean((day_scores - day_rets) ** 2)))
        return {
            "rank_ic_mean": np.mean(rank_ics) if rank_ics else 0.0,
            "rank_icir": (np.mean(rank_ics) / (np.std(rank_ics) + 1e-8)
                          if rank_ics else 0.0),
            "top10_excess_ret": np.mean(top10_exc) if top10_exc else 0.0,
            "top20_excess_ret": np.mean(top20_exc) if top20_exc else 0.0,
            "top30_excess_ret": np.mean(top30_exc) if top30_exc else 0.0,
            "top_bottom_spread": np.mean(spreads) if spreads else 0.0,
            "sign_accuracy": np.mean(sign_correct) if sign_correct else 0.0,
            "rmse": np.mean(rmse_vals) if rmse_vals else 0.0,
        }

    def _log_epoch(self, epoch, phase, train_loss, val_metrics, elapsed, lr):
        phase_tag = {1: "BCE", 2: "REG", 3: "FT"}[phase]
        entry = {
            "epoch": epoch, "phase": phase, "train_loss": train_loss,
            "elapsed_sec": round(elapsed, 1), **val_metrics,
        }
        self.history.append(entry)
        model_name = self.cfg.get("model_name", "clime")
        print(f"  E{epoch:3d}[CLIME/{phase_tag}] | loss={train_loss:.4f} | "
              f"SignAcc={val_metrics['sign_accuracy']:.3f} | "
              f"RMSE={val_metrics['rmse']:.4f} | "
              f"IC={val_metrics['rank_ic_mean']:.4f} | "
              f"Top20_exc={val_metrics['top20_excess_ret']:.6f} | "
              f"lr={lr:.2e} | {elapsed:.1f}s")

    def _save_checkpoint(self, filename):
        path = OUTPUT_DIR / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
        }, path)


# ===========================================================================
# 数据加载
# ===========================================================================

def _load_clime_data():
    """加载 CLIME Stage 2 训练数据。从缓存切片到 67 维特征。"""
    train_data = torch.load(
        _cache_path_for("train_v5", L_VAL, SPLITS["train_v5"][2]),
        weights_only=False,
    )
    val_data = torch.load(
        _cache_path_for("val_v5", L_VAL, SPLITS["val_v5"][2]),
        weights_only=False,
    )
    train_dyn_path = (_CACHE_DIR /
                      f"v7_peer_dynamics_train_v5_L{L_VAL}"
                      f"_step{SPLITS['train_v5'][2]}.pt")
    val_dyn_path = (_CACHE_DIR /
                    f"v7_peer_dynamics_val_v5_L{L_VAL}"
                    f"_step{SPLITS['val_v5'][2]}.pt")
    train_dyn = torch.load(train_dyn_path, weights_only=False)
    val_dyn = torch.load(val_dyn_path, weights_only=False)

    # Slice features to 67-dim
    v9_feat_idx = list(range(66)) + [86]
    for d in train_data["sequences"]:
        if train_data["sequences"][d].shape[-1] > 67:
            train_data["sequences"][d] = train_data["sequences"][d][:, :, v9_feat_idx]
    for d in val_data["sequences"]:
        if val_data["sequences"][d].shape[-1] > 67:
            val_data["sequences"][d] = val_data["sequences"][d][:, :, v9_feat_idx]
    return train_data, val_data, train_dyn, val_dyn


# ===========================================================================
# V9CA_AB 训练
# ===========================================================================

def train_clime(args):
    """CLIME Stage 2: Scaled + Gated Injection 完整训练。对应报告 Section 2.4.2。"""
    cfg = dict(CFG_CLIME)
    if args.epochs: cfg["max_epochs"] = args.epochs
    if args.batch_size: cfg["batch_size"] = args.batch_size
    if args.init_scale is not None:
        cfg["init_scale"] = args.init_scale
        is_tag = str(args.init_scale).replace(".", "p")
        cfg["model_name"] = f"clime_is{is_tag}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stage1_path = args.stage1_ckpt or str(OUTPUT_DIR / "stage1_best.pt")
    if not Path(stage1_path).exists():
        print(f"[error] Stage 1 checkpoint not found: {stage1_path}")
        sys.exit(1)

    print("=" * 60)
    print("CLIME Stage 2: Scaled + Gated Injection (combined)")
    print(f"  报告 Section 2.4.2: 3-Phase Curriculum Training")
    print(f"  init_scale: {cfg['init_scale']}  |  model_name: {cfg['model_name']}")
    print(f"  Stage 1 ckpt: {stage1_path}")
    print("=" * 60)

    train_data, val_data, train_dyn, val_dyn = _load_clime_data()
    train_ds = RegressionPeerDataset(
        train_data["sequences"], train_data["returns"], train_dyn,
        samples_per_day=cfg["samples_per_day"],
    )
    val_x, val_dyn_arr, val_ret, val_ids, val_dates = \
        build_regression_peer_validation_arrays(val_data, val_dyn)

    model = CLIMEModel(stage1_path, init_scale=cfg.get("init_scale", 0.1),
                       gate_hidden=cfg.get("gate_hidden", 64))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  CLIME Model: params={n_params:,} "
          f"(backbone ~2.3M + encoder ~0.5M + head)")

    trainer = CLIMETrainer(
        model, train_ds, val_x, val_dyn_arr, val_ret, val_ids, val_dates, cfg)
    return trainer.fit()


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="CLIME: Cross-Modal Injection via Learned Market Encoding — 训练入口")
    parser.add_argument("--stage1", action="store_true",
                        help="Stage 1: Backbone 预训练（报告 Section 2.4.1）")
    parser.add_argument("--clime", "--v9ca-ab", action="store_true", dest="clime",
                        help="Stage 2: CLIME 完整训练（报告 Section 2.4.2）")
    parser.add_argument("--stage1-ckpt", type=str, default=None,
                        help="Stage 1 checkpoint 路径（用于 Stage 2）")
    parser.add_argument("--init-scale", type=float, default=None,
                        help="logit_scale 初始值（报告推荐 0.3, 附录 H）")
    parser.add_argument("--resume", type=str, default=None,
                        help="Stage 1: resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--feat-dim", type=int, default=None,
                        help="Override feature dimension for Stage1")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--ffn-hidden", type=int, default=1024)
    args = parser.parse_args()

    if args.stage1:
        result = train_stage1(args)
    elif getattr(args, "clime", False):
        result = train_clime(args)
    else:
        print("Usage: python train.py --stage1 | --clime")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"DONE. Best epoch={result['best_epoch']}, "
          f"Top-20 excess={result['best_metric']:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
