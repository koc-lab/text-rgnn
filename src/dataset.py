import torch_geometric.transforms as T


from src.hetero_generator import load_variables, get_hetero_data
from dataclasses import dataclass
from enum import Enum


class DatasetNames(Enum):
    MR = "mr"
    R8 = "R8"
    R52 = "R52"
    OHSUMED = "ohsumed"


@dataclass
class GraphDatasetConfig:
    dataset_name: str
    doc_doc_k: int
    word_word_k: int


class GraphDataset:
    def __init__(self, config: GraphDatasetConfig, word_to_word_graph: bool = False):
        processed_dataset, vocab, stoi, itos = load_variables(config.dataset_name)
        data = get_hetero_data(
            config.dataset_name,
            processed_dataset,
            doc_doc_k=config.doc_doc_k,
            word_word_k=config.word_word_k,
            word_to_word_graph=word_to_word_graph,
        )

        data = T.ToUndirected()(data)
        data = T.AddSelfLoops()(data)
        # ! feature normalize edince bok gibi oluyor neden bak
        # data = T.NormalizeFeatures()(data)
        self.config = config
        self.data = data
        self.processed_dataset = processed_dataset
        self.vocab = vocab
        self.stoi = stoi
        self.itos = itos

    def apply_transform(self, transform: T):
        self.data = transform(self.data)
