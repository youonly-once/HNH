
# @Time    : 2023/11/7
# @Author  : SXS
# @Github  : https://github.com/SXS-PRIVATE/HNH
import torch
import numpy as np


class RandomErasure(object):
    """
    Randomly erasure input bow vector.
    change one to a assign value in a bow vector
    """
    def __init__(self, prob=0.2, value=0):
        if prob < 0 or prob > 1:
            raise ValueError("probability only can be a float in 0 to 1")
        self.prob = prob
        self.value = value

    def __call__(self, vector: torch.Tensor):
        vector_length = vector.shape[-1]
        index = np.random.permutation(vector_length)
        change_num = int(vector_length * self.prob)
        vector[index[: change_num]] = self.value
        return vector

    def __repr__(self):
        return self.__class__.__name__ + '(prob={0}, value={1})'.format(self.prob, self.value)

