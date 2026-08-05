from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        vocab = set(text)

        vocab = sorted(vocab)

        stoi = {}
        itos = {}

        for i, tok in enumerate(vocab):
            stoi[tok] = i
            itos[i] = tok

        return stoi, itos  
        # pass

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        
        return [stoi[token] for token in list(text)]
        # pass

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        ans = ""
        for id in ids:
            ans += itos[id]
        return ans
        # pass
