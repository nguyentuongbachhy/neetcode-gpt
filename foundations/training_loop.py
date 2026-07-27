import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        
        n_samples, n_features = X.shape
        print("y:", y.shape, y)
        w = np.zeros(n_features)
        print("w:", w.shape, w)
        b = 0

        for _ in range(epochs):
            y_hat = X @ w + b - y
            print("y_hat:", y_hat.shape, y_hat)
            
            mse_loss = np.mean(y_hat ** 2)
            print("mse_loss:", mse_loss.shape, mse_loss)
            delta = 2.0 * y_hat / n_samples
            print("delta:", delta.shape, delta)
            w -= lr * X.T @ delta
            print("w:", w.shape, w)
            b -= lr * delta.sum()
            print("b:", b.shape, b)
            # break


        return (np.round(w, 5), round(b, 5))
        # pass
