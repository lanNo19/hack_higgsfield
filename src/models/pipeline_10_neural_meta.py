"""Pipeline 10: Neural Meta-Learner.

Replace the LogReg meta-learner in the stacking ensemble (P04) with a small MLP.
Architecture: 15 → 32 → 16 → 3 (softmax), Adam, label smoothing, early stopping.

Uses the Level-1 meta-features (OOF probas) saved by Pipeline 4.
Requires: Pipeline 4 must have run first (oof_p04_meta_tv.npy).
Expected time: ~30 minutes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.models.pipeline_utils import (
    ARTIFACTS, evaluate_proba, load_train_data,
    make_holdout, save_oof, save_result,
)
from src.utils.logger import get_logger

log = get_logger(__name__)
NAME = "P10_neural_meta"
DEVICE = "cpu"  # RTX 5090 (sm_120) not supported by current PyTorch build


# ── MLP Architecture ──────────────────────────────────────────────────────────

class MetaMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _train_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-3,
    label_smoothing: float = 0.05,
    patience: int = 20,
) -> MetaMLP:
    model = MetaMLP(X_tr.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr_t), y_tr_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def _predict_mlp(model: MetaMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    return proba


def run() -> dict:
    log.info("=" * 60)
    log.info("Pipeline 10: Neural Meta-Learner  [device: %s]", DEVICE)
    log.info("=" * 60)

    X, y = load_train_data()
    X_tv, X_hold, y_tv, y_hold = make_holdout(X, y)

    meta_tv_path = ARTIFACTS / "oof_p04_meta_tv.npy"
    if not meta_tv_path.exists():
        raise FileNotFoundError(
            "oof_p04_meta_tv.npy not found. Run Pipeline 4 first."
        )

    # Level-1 meta-features from P04 (trainval OOF, shape: N × 15)
    meta_tv = np.load(meta_tv_path)
    log.info("L1 meta-features shape: %s", meta_tv.shape)

    scaler = StandardScaler()
    meta_tv_scaled = scaler.fit_transform(meta_tv)

    # 5-fold CV to generate OOF predictions with neural meta-learner
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(X_tv), 3))

    for fold, (tr, val) in enumerate(cv.split(meta_tv_scaled, y_tv)):
        model = _train_mlp(
            meta_tv_scaled[tr], y_tv[tr],
            meta_tv_scaled[val], y_tv[val],
        )
        oof[val] = _predict_mlp(model, meta_tv_scaled[val])
        log.info("Fold %d done", fold + 1)

    cv_result = evaluate_proba(f"{NAME}_oof", y_tv, oof)
    save_result(cv_result)
    save_oof("p10", oof)

    # For holdout: need to rebuild L1 meta-features on holdout.
    # We reuse the P04 base learner predictions on holdout saved in test_meta.
    # Since P04 saves test_meta implicitly in oof_p04_holdout we can't trivially get
    # the raw L1 probas for holdout. Instead retrain meta on full meta_tv_scaled.
    log.info("Retraining neural meta-learner on full trainval for holdout eval...")

    # Split 15% of meta_tv for inner validation during final training
    n_val = max(1, int(0.15 * len(meta_tv_scaled)))
    X_m_tr, X_m_val = meta_tv_scaled[:-n_val], meta_tv_scaled[-n_val:]
    y_m_tr, y_m_val = y_tv[:-n_val], y_tv[-n_val:]

    final_model = _train_mlp(X_m_tr, y_m_tr, X_m_val, y_m_val, epochs=300)

    # Holdout meta-features: load P04's L1 holdout stack (saved by P04 as oof_p04_meta_hold.npy)
    meta_hold_path = ARTIFACTS / "oof_p04_meta_hold.npy"
    if not meta_hold_path.exists():
        log.warning("oof_p04_meta_hold.npy not found. Skipping holdout eval.")
        return cv_result

    hold_meta = np.load(meta_hold_path)

    hold_meta_scaled = scaler.transform(hold_meta)
    hold_proba = _predict_mlp(final_model, hold_meta_scaled)
    hold_result = evaluate_proba(f"{NAME}_holdout", y_hold, hold_proba)
    save_result(hold_result)
    np.save(ARTIFACTS / "oof_p10_holdout.npy", hold_proba)

    return hold_result


if __name__ == "__main__":
    run()
