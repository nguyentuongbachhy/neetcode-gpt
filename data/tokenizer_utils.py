from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        
        ans = []

        for number in numbers:
            s = str(number)

            tokens = []
            i = 0
            n = len(s)

            while i < n:
                longest = None
                for j in range(i + 1, n + 1):
                    candidate = s[i: j]
                    if candidate in vocab:
                        longest = candidate
                if not longest:
                    longest = s[i]
                    i += 1
                else:
                    tokens.append(longest)
                    i += len(longest)

            ans.append(tokens)

        return ans
        # pass

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        
        count = 0
        i = 0
        n = len(text)

        while i < n:
            longest = None
            for j in range(i + 1, n + 1):
                candidate = text[i: j]
                if candidate in vocab:
                    longest = candidate

            if not longest:
                i += 1
            else:
                i += len(longest)

            count += 1

        return count
        # pass

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words = text.split()
        if not words:
            return 0.0

        return round(self.count_tokens(text, vocab) / len(words), 4)
        
        # pass
