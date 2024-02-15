import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from src.configurations import GraphDatasetConfig
from src.graph_dataset_utils import (
    generate_knn_graph,
    load_doc_embeddings,
    load_doc_word_graph,
    load_processed_dataset,
    load_word_embeddings,
)
from src.preprocess import ProcessedDataset


class GraphDataset:
    def __init__(self, c: GraphDatasetConfig):
        dataset, vocab, stoi, itos = load_processed_dataset(c.dataset_name)

        self.hetero_data = generate_hetero(
            c.dataset_name, dataset, c.doc_doc_k, c.word_word_k, c.use_w2w_graph
        )

        self.hetero_data = T.ToUndirected()(self.hetero_data)
        self.hetero_data = T.AddSelfLoops()(self.hetero_data)

        self.config = c
        self.processed_dataset = dataset
        self.n_class = dataset.n_class

        self.vocab = vocab
        self.stoi = stoi
        self.itos = itos


def generate_hetero(
    dataset_name: str,
    processed_dataset: ProcessedDataset,
    doc_doc_k: int,
    word_word_k: int,
    use_w2w_graph: bool,
):
    data = HeteroData()
    data["word"].x = load_word_embeddings(dataset_name)
    data["doc"].x = load_doc_embeddings(dataset_name).to("cpu")

    data["doc"].y = processed_dataset.label
    data["doc"].train_mask = processed_dataset.train_mask
    data["doc"].test_mask = processed_dataset.test_mask

    # Doc-Doc Graph
    e_i, e_a, A = generate_knn_graph(data["doc"].x, n_neighbours=doc_doc_k)
    data["doc", "d2d", "doc"].edge_index = e_i
    data["doc", "d2d", "doc"].edge_attr = e_a

    # Word-Word Graph
    if use_w2w_graph:
        e_i, e_a, A = generate_knn_graph(data["word"].x, n_neighbours=word_word_k)
        data["word", "w2w", "word"].edge_index = e_i
        data["word", "w2w", "word"].edge_attr = e_a

    # Doc-Word Graph
    e_i, e_a = load_doc_word_graph(dataset_name)
    data["doc", "d2w", "word"].edge_index = e_i
    data["doc", "d2w", "word"].edge_attr = e_a

    return data
