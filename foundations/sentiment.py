import torch
import torch.nn as nn
from torchtyping import TensorType

class MeanPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim)

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        
        self.model = nn.Sequential(
            nn.Embedding(vocabulary_size, 16),   # (B, T) -> (B, T, 16)
            MeanPooling(dim=1),                  # (B, T, 16) -> (B, 16)
            nn.Linear(16, 1),                    # (B, 16) -> (B, 1)
            nn.Sigmoid()                         # (B, 1) -> (B, 1) trong (0,1)
        )
        # pass

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        
        return self.model(x).round(decimals=4)
        # pass
