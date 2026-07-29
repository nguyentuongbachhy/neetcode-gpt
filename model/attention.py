import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)  # for reproducible weights
        self.attention_dim = attention_dim

        # Instantiation order: key, query, value (bias=False)
        self.key   = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V
        K = self.key(embedded)   # (batch, seq_len, attention_dim)
        Q = self.query(embedded)
        V = self.value(embedded)

        # 2. Scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.attention_dim ** 0.5)

        # 3. Apply causal mask (lower triangular)
        seq_len = embedded.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=embedded.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4. Softmax over the last dimension
        attn_weights = torch.softmax(scores, dim=2)

        # 5. Compute output and round to 4 decimal places
        output = attn_weights @ V
        return torch.round(output * 10000) / 10000