import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import kneighbors_graph as knn_graph


class KNNTrainer:
    def __init__(self, X: torch.Tensor, k: int):
        self.X = X
        self.k = k

    def fit(self):
        A = knn_graph(self.X, self.k, mode="distance", include_self=True, p=2)
        A = sp.csr_matrix(A) if not sp.issparse(A) else A

        row_idx, col_idx = A.nonzero()
        edge_index = torch.tensor(np.vstack((row_idx, col_idx)), dtype=torch.long)
        edge_attr = torch.tensor(A[row_idx, col_idx], dtype=torch.float)
        return edge_index, edge_attr
