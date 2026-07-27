import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x, ndmin=2, dtype=np.float64)
        gamma = np.array(gamma, ndmin=2, dtype=np.float64)
        beta = np.array(beta, ndmin=2, dtype=np.float64)
        running_mean = np.array(running_mean, ndmin=2, dtype=np.float64)
        running_var = np.array(running_var, ndmin=2, dtype=np.float64)

        if training:
            mean_b = np.mean(x, axis=0, keepdims=True)
            x -= mean_b
            var_b = np.var(x, axis=0, keepdims=True)
            x_pred = x / np.sqrt(var_b + eps)

            y = gamma * x_pred + beta

            running_mean = (1 - momentum) * running_mean + momentum * mean_b
            running_var = (1 - momentum) * running_var + momentum * var_b
        else:
            x_pred = (x - running_mean) / np.sqrt(running_var + eps)
            y = gamma * x_pred + beta

        return (np.round(y, 4).tolist(), np.round(running_mean, 4).squeeze().tolist(), np.round(running_var, 4).squeeze().tolist())


        # pass
