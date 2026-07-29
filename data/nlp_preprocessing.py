import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        sentences = positive + negative
        token_set = set()

        for sent in sentences:
            token_set.update(sent.split())

        token_set = sorted(token_set)
        word2idx = {token: i + 1 for i, token in enumerate(token_set)}

        encoded = []
        for sent in sentences:
            ids = [word2idx[w] for w in sent.split()]
            encoded.append(torch.tensor(ids, dtype=torch.float))

        padded = nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=0.0)
        
        return padded
        # pass
