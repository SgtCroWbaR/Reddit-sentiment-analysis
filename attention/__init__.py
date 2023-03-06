import numpy as np
from copy import deepcopy
from math import ceil


class MultiHeadAttention:

    def __init__(self, embeddings_array, num_of_heads):
        self.embeddings_array = embeddings_array
        self.num_of_heads = num_of_heads
        self._calc_attention()

    def _calc_attention(self):
        self.scores = []
        self.normalised_scores = []
        self.reweighted = []

        interval = ceil(100 / self.num_of_heads)
        for embeddnings in self.embeddings_array:
            i = 0
            score = np.dot(embeddnings[:, i * interval:(i + 1) * interval],
                           np.transpose(embeddnings[:, i * interval:(i + 1) * interval]))
            normalised_score = np.array([row / max(1, np.sum(row)) for row in score])
            reweight = np.matmul(score, embeddnings[:, i * interval:(i + 1) * interval])
            i += 1
            while i < self.num_of_heads:
                np.concatenate((score, np.dot(embeddnings[:, i * interval:(i + 1) * interval],
                                              np.transpose(embeddnings[:, i * interval:(i + 1) * interval]))), axis=1)
                np.concatenate((normalised_score, np.array([row / max(1, np.sum(row)) for row in score])), axis=1)
                np.concatenate((reweight, np.matmul(score, embeddnings[:, i * interval:(i + 1) * interval])), axis=1)
                i += 1
            self.scores.append(deepcopy(score))
            self.normalised_scores.append(deepcopy(normalised_score))
            self.reweighted.append(deepcopy(reweight))  # return value
