# %%
import torch
import pickle
from pathlib import Path

from torch_geometric.data import HeteroData
from src.utils import get_w2v_embeddings, get_knn_graph
from src.preprocess import ProcessedDataset


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
    dataset_name: str, processed_dataset: ProcessedDataset, doc_doc_k=50
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
    # e_i, e_a = get_knn_graph(data["word"].x, n_neighbours=30)
    # data["word", "w2w", "word"].edge_index = e_i
    # data["word", "w2w", "word"].edge_attr = e_a

    # Doc-Word Graph
    e_i, e_a = get_doc_word_graph(dataset_name)
    data["doc", "d2w", "word"].edge_index = e_i
    data["doc", "d2w", "word"].edge_attr = e_a

    return data
