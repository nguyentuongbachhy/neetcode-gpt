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
        x = np.array(x, dtype=np.float64, ndmin=2)
        gamma = np.array(gamma, dtype=np.float64, ndmin=2)
        beta = np.array(beta, dtype=np.float64, ndmin=2)
        running_mean = np.array(running_mean, dtype=np.float64, ndmin=2)
        running_var = np.array(running_var, dtype=np.float64, ndmin=2)

        if training:
            mean_b = np.mean(x, axis=0, keepdims=True)
            x -= mean_b
            var_b = np.var(x, axis=0, keepdims=True)

            x_hat = x / np.sqrt(var_b + eps)
            y = gamma * x_hat + beta

            running_mean = (1 - momentum) * running_mean + momentum * mean_b
            running_var = (1 - momentum) * running_var + momentum * var_b
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
            y = gamma * x_hat + beta
        
        y = np.round(y, 4)
        running_mean = np.round(running_mean, 4)
        running_var = np.round(running_var, 4)

        return y.tolist(), running_mean.reshape(-1).tolist(), running_var.reshape(-1).tolist()
        # pass
