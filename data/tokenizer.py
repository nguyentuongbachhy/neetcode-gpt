from typing import List
from collections import defaultdict


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            if len(tokens) < 2:
                break

            freq = defaultdict(int)
            best_freq = 0
            for i in range(len(tokens) - 1):
                key = (tokens[i], tokens[i + 1])
                freq[key] += 1
                if best_freq < freq[key]:
                    best_freq = freq[key]

            if not freq:
                break

            candidates = [pair for pair, count in freq.items() if count == best_freq]
            candidates.sort()

            best_candidate = candidates[0]
            merges.append([best_candidate[0], best_candidate[1]])

            i = 0
            n = len(tokens)
            new_tokens = []
            merged_token = best_candidate[0] + best_candidate[1]
            while i < n:
                if i < n - 1 and tokens[i] == best_candidate[0] and tokens[i + 1] == best_candidate[1]:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return merges
        # pass
