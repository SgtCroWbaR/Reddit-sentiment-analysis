import numpy as np
from copy import deepcopy
from math import ceil, sqrt
from tqdm import tqdm


class MultiHeadAttention:

    def __init__(self, embeddings_array, num_of_heads, input_dim=100):
        self.embeddings_array = embeddings_array
        self.num_of_heads = num_of_heads
        self.input_dim = input_dim
        self._calc_attention()

    def _calc_attention(self):
        self.scores = []
        self.reweighted = []

        interval = ceil(self.input_dim / self.num_of_heads)
        for embeddnings in tqdm(self.embeddings_array, desc="Calculating attention"):
            i = 0

            score = np.dot(embeddnings[:, i * interval:(i + 1) * interval],
                           np.transpose(embeddnings[:, i * interval:(i + 1) * interval]))
            reweight = np.matmul(score, embeddnings[:, i * interval:(i + 1) * interval])
            i += 1
            while i < self.num_of_heads:
                next_score = np.dot(embeddnings[:, i * interval:(i + 1) * interval],
                                    np.transpose(embeddnings[:, i * interval:(i + 1) * interval]))
                next_reweight = np.matmul(next_score, embeddnings[:, i * interval:(i + 1) * interval])

                score = np.concatenate((deepcopy(score), deepcopy(next_score)), axis=1)
                reweight = np.concatenate((deepcopy(reweight), deepcopy(next_reweight)), axis=1)
                i += 1
            self.scores.append(deepcopy(score))
            self.reweighted.append(deepcopy(reweight) / sqrt(self.input_dim))  # return value
