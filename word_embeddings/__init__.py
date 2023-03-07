import numpy as np
from typing import List
from copy import deepcopy
from tqdm import tqdm


class Embeddings:
    def __init__(self, texts: List[str], dictionary: dict):
        self.texts = texts
        self.dictionary = dictionary
        self.texts_embedings = []
        self._compute_embeddings()

    def _compute_embeddings(self):
        for comment in tqdm(self.texts, desc="Computing Embeddings"):
            comment_embeddings = []
            for word in comment.strip().split(' '):
                comment_embeddings.append(np.array(self.dictionary.get(word, np.zeros((100,))), dtype=float))
            self.texts_embedings.append(deepcopy(np.array(comment_embeddings)))
