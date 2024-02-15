import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from gensim.models import Word2Vec
from gensim.models.word2vec import KeyedVectors
from sklearn.neighbors import kneighbors_graph as knn_graph

from src.preprocess import ProcessedDataset


def load_word_embeddings(dataset_name: str):
    path = Path.cwd() / "w2v-models" / f"{dataset_name}" / "model.bin"
    word2vec_model = Word2Vec.load(str(path))
    wv = word2vec_model.wv
    embeddings = np.array([wv[word] for word in wv.index_to_key])
    embeddings = torch.tensor(embeddings, dtype=torch.float)
    return embeddings


def load_doc_embeddings(dataset_name: str):
    DOCUMENTS_PATH = Path.cwd().parent
    EMBEDDINGS_PATH = DOCUMENTS_PATH / "finetune-text-graphs/generator-output"
    file_name = f"{dataset_name}_embeddings.pth"
    embeddings = torch.load(EMBEDDINGS_PATH / file_name)
    return embeddings


def load_doc_word_graph(dataset_name: str):
    file_path = Path.cwd() / "tf-idf-graphs" / f"{dataset_name}tfidf_graph.pth"
    edge_index, edge_attr = torch.load(file_path)
    return edge_index, edge_attr


def generate_knn_graph(X: torch.tensor, n_neighbours: int = 10):
    A = knn_graph(X, n_neighbours, mode="distance", include_self=True, p=2)
    if not sp.issparse(A):
        A = sp.csr_matrix(A)

    row_indices, col_indices = A.nonzero()
    edge_index = np.vstack((row_indices, col_indices))
    edge_attr = A[row_indices, col_indices]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_index, edge_attr, A


def load_processed_dataset(dataset_name):
    data_path = Path.cwd() / f"processed-data/{dataset_name}_generated_from_trainer.pkl"
    vocab_path = Path.cwd() / f"w2v-models/{dataset_name}/vocab.pkl"

    with open(data_path, "rb") as file:
        processed_dataset: ProcessedDataset = pickle.load(file)

    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)
        vocab = vocab[0]

    stoi = {word: i for i, word in enumerate(vocab)}
    itos = {i: word for i, word in enumerate(vocab)}

    return processed_dataset, vocab, stoi, itos


def sanity_check_for_embedding_matrix(
    wv: KeyedVectors,
    similarity_matrix: torch.tensor,
    stoi: dict,
    w1: str = "bad",
    w2: str = "good",
):
    w1_vec = torch.tensor(wv[w1], dtype=torch.float).reshape(1, -1)
    w2_vec = torch.tensor(wv[w2], dtype=torch.float).reshape(1, -1)

    print(f"Word2Vec Simlarity:{wv.similarity(w1, w2)}")
    print(f"Cosine Similarity: {F.cosine_similarity(w1_vec, w2_vec)}")
    print(f"Similartiy matrix entry: {similarity_matrix[stoi[w1], stoi[w2]]}")
