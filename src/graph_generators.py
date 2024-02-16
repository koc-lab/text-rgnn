from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import kneighbors_graph as knn_graph

from src import TF_IDF_GRAPHS_PATH
from src.text_dataset import TextDataset
from src.utils import load_vocab


def generate_tfidf_graph(dataset_name: str):
    textdataset = TextDataset(dataset_name)
    vocab = load_vocab(dataset_name)
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    tfidf_matrix = vectorizer.fit_transform(textdataset.documents)
    A = coo_matrix(tfidf_matrix)

    edge_index = torch.tensor(np.array([A.row, A.col]), dtype=torch.long)
    edge_attr = torch.tensor(A.data, dtype=torch.float32).view(-1)
    return edge_index, edge_attr


def tfidf_pipeline(dataset_name: str):
    file_path = TF_IDF_GRAPHS_PATH / f"{dataset_name}.pth"
    if file_path.exists():
        print(f"TF-IDF graph for {dataset_name} already exists. Skipping...")
        return
    edge_index, edge_attr = generate_tfidf_graph(dataset_name)
    torch.save((edge_index, edge_attr), file_path)


def generate_knn_graph(X: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a k-nearest neighbors graph based on the input data.

    Parameters:
        X (torch.Tensor): Input data tensor of shape (N, D) where N is the number of samples and D is the feature dimension.
        k (int): Number of nearest neighbors to consider for each sample.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the edge index tensor and edge attribute tensor.
            - edge_index (torch.Tensor): A tensor containing the indices of edges in the graph. Shape: (2, E) where E is the number of edges.
            - edge_attr (torch.Tensor): A tensor containing the attributes (distances) of edges in the graph. Shape: (E,)
    """
    A = knn_graph(X, k, mode="distance", include_self=True, p=2)
    A = sp.csr_matrix(A) if not sp.issparse(A) else A

    row_indices, col_indices = A.nonzero()
    edge_index = torch.tensor(np.vstack((row_indices, col_indices)), dtype=torch.long)
    edge_attr = torch.tensor(A[row_indices, col_indices], dtype=torch.float)
    return edge_index, edge_attr
