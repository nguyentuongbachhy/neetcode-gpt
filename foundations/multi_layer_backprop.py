import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Convert inputs once to 2D arrays (single sample, batch size = 1)
        X = np.array(x, ndmin=2, dtype=np.float64)          # (1, in_dim)
        Y = np.array(y_true, ndmin=2, dtype=np.float64)     # (1, out_dim)
        W1 = np.array(W1, ndmin=2, dtype=np.float64)        # (hidden_dim, in_dim)
        b1 = np.array(b1, ndmin=2, dtype=np.float64)        # (1, hidden_dim)
        W2 = np.array(W2, ndmin=2, dtype=np.float64)        # (out_dim, hidden_dim)
        b2 = np.array(b2, ndmin=2, dtype=np.float64)        # (1, out_dim)

        # ---- Forward pass ----
        z1 = X @ W1.T + b1               # (1, hidden_dim)
        a1 = np.maximum(z1, 0.0)         # ReLU
        y_pred = a1 @ W2.T + b2          # (1, out_dim)

        # ---- Loss (MSE) ----
        error = y_pred - Y
        loss = np.mean(error ** 2)

        # ---- Backward pass ----
        # dL/dy_pred  (1, out_dim)
        dL_dpred = 2.0 * error / Y.size   # Y.size == y_pred.size (batch_size * out_dim)

        # Gradients for W2, b2
        dW2 = dL_dpred.T @ a1            # (out_dim, 1) @ (1, hidden_dim) -> (out_dim, hidden_dim)
        db2 = dL_dpred.sum(axis=0)       # (out_dim,)

        # Backprop through ReLU and first linear layer
        dL_da1 = dL_dpred @ W2           # (1, hidden_dim)
        dL_dz1 = dL_da1 * (z1 > 0.0)     # ReLU derivative mask
        dW1 = dL_dz1.T @ X               # (hidden_dim, 1) @ (1, in_dim) -> (hidden_dim, in_dim)
        db1 = dL_dz1.sum(axis=0)         # (hidden_dim,)

        # ---- Round and return ----
        return {
            'loss': float(np.round(loss, 4)),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }