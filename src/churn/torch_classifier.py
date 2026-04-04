"""PyTorch MLP with an sklearn-compatible interface for tabular binary classification."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            dim = hidden_dim
        layers.append(nn.Linear(dim, 2))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper around a BatchNorm MLP for binary classification.

    Handles NaN imputation and standard scaling internally so it can be used
    as a drop-in replacement for sklearn classifiers in the CV loop.
    Accepts eval_set=(X_val, y_val) in fit() for early stopping.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.3,
        batch_size: int = 2048,
        lr: float = 1e-3,
        max_epochs: int = 100,
        patience: int = 15,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state

    # ------------------------------------------------------------------
    def fit(self, X, y, eval_set: tuple | None = None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        self.classes_ = np.array([0, 1])

        Xn = self.scaler_.fit_transform(self.imputer_.fit_transform(
            X.values if hasattr(X, "values") else np.asarray(X)
        ))
        y = np.asarray(y)

        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        pos_w = n_neg / max(n_pos, 1)

        device = torch.device("cpu")
        Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
        yt = torch.tensor(y, dtype=torch.long, device=device)

        self.model_ = _MLP(Xt.shape[1], self.hidden_dim, self.n_layers, self.dropout).to(device)

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_w], dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        # Prepare validation tensors for early stopping
        Xvt = yvt = None
        if eval_set is not None:
            Xv, yv = eval_set
            Xvn = self.scaler_.transform(self.imputer_.transform(
                Xv.values if hasattr(Xv, "values") else np.asarray(Xv)
            ))
            Xvt = torch.tensor(Xvn, dtype=torch.float32, device=device)
            yvt = torch.tensor(np.asarray(yv), dtype=torch.long, device=device)

        best_loss = float("inf")
        patience_ctr = 0
        best_state: dict | None = None
        n = len(Xt)

        for _ in range(self.max_epochs):
            self.model_.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                optimizer.zero_grad()
                criterion(self.model_(Xt[idx]), yt[idx]).backward()
                optimizer.step()
            scheduler.step()

            if Xvt is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_loss = criterion(self.model_(Xvt), yvt).item()
                if val_loss < best_loss - 1e-4:
                    best_loss = val_loss
                    patience_ctr = 0
                    best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.patience:
                        break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        Xn = self.scaler_.transform(self.imputer_.transform(
            X.values if hasattr(X, "values") else np.asarray(X)
        ))
        Xt = torch.tensor(Xn, dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            proba = torch.softmax(self.model_(Xt), dim=1).numpy()
        return proba

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
