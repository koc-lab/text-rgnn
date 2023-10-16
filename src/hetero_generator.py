# %%
import torch
import pickle

import scipy.sparse as sp
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec

from torch_geometric.data import HeteroData
from src.preprocess import ProcessedDataset
from sklearn.neighbors import kneighbors_graph as knn_graph
import torch.nn.functional as F
from gensim.models.word2vec import KeyedVectors


def get_w2v_embeddings(dataset_name: str):
    path = Path.cwd() / "w2v-models" / f"{dataset_name}" / "model.bin"
    word2vec_model = Word2Vec.load(str(path))
    wv = word2vec_model.wv
    embeddings = np.array([wv[word] for word in wv.index_to_key])
    embeddings = torch.tensor(embeddings, dtype=torch.float)
    return embeddings


def get_knn_graph(X: torch.tensor, n_neighbours: int = 10):
    A = knn_graph(X, n_neighbours, mode="distance", include_self=True, p=2)
    if not sp.issparse(A):
        A = sp.csr_matrix(A)

    row_indices, col_indices = A.nonzero()
    edge_index = np.vstack((row_indices, col_indices))
    edge_attr = A[row_indices, col_indices]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_index, edge_attr


def get_doc_embeddings(dataset_name: str):
    DOCUMENTS_PATH = Path.cwd().parent
    EMBEDDINGS_PATH = DOCUMENTS_PATH / "finetune-text-graphs/generator-output"
    file_name = f"{dataset_name}_embeddings.pth"
    embeddings = torch.load(EMBEDDINGS_PATH / file_name)
    return embeddings


def get_word_embeddings(dataset_name: str):
    embeddings = get_w2v_embeddings(dataset_name)
    return embeddings


def get_doc_word_graph(dataset_name: str):
    file_path = Path.cwd() / "tf-idf-graphs" / f"{dataset_name}tfidf_graph.pth"
    edge_index, edge_attr = torch.load(file_path)
    return edge_index, edge_attr


def load_variables(dataset_name):
    data_path = f"/Users/ardaaras/Documents/finetune-text-graphs/processed-data/{dataset_name}_generated_from_trainer.pkl"
    vocab_path = Path.cwd() / f"w2v-models/{dataset_name}/vocab.pkl"

    with open(data_path, "rb") as file:
        processed_dataset: ProcessedDataset = pickle.load(file)

    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)
        vocab = vocab[0]

    stoi = {word: i for i, word in enumerate(vocab)}
    itos = {i: word for i, word in enumerate(vocab)}

    return processed_dataset, vocab, stoi, itos


def get_hetero_data(
    dataset_name: str,
    processed_dataset: ProcessedDataset,
    doc_doc_k: int,
    word_word_k: int,
    word_to_word_graph: bool,
):
    data = HeteroData()
    data["doc"].x = get_doc_embeddings(dataset_name).to("cpu")
    data["word"].x = get_word_embeddings(dataset_name)

    data["doc"].y = processed_dataset.label
    data["doc"].train_mask = processed_dataset.train_mask
    data["doc"].test_mask = processed_dataset.test_mask

    # Doc-Doc Graph
    e_i, e_a = get_knn_graph(data["doc"].x, n_neighbours=doc_doc_k)
    data["doc", "d2d", "doc"].edge_index = e_i
    data["doc", "d2d", "doc"].edge_attr = e_a

    # Word-Word Graph
    if word_to_word_graph:
        e_i, e_a = get_knn_graph(data["word"].x, n_neighbours=word_word_k)
        data["word", "w2w", "word"].edge_index = e_i
        data["word", "w2w", "word"].edge_attr = e_a

    # Doc-Word Graph
    e_i, e_a = get_doc_word_graph(dataset_name)
    data["doc", "d2w", "word"].edge_index = e_i
    data["doc", "d2w", "word"].edge_attr = e_a

    return data


# %%
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
