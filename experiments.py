# %%
import torch
from torch_geometric.nn import to_hetero
from src.models import GNN, GAT
from src.utils import set_seeds

from src.dataset import GraphDataset, DatasetNames, GraphDatasetConfig
from src.engine import Trainer

config = GraphDatasetConfig(
    dataset_name=DatasetNames.MR.value, doc_doc_k=20, word_word_k=20
)
graph_dataset = GraphDataset(config)
processed_dataset = graph_dataset.processed_dataset

train_mask = graph_dataset.processed_dataset.train_mask

model = GNN(hidden_channels=1433, out_channels=7)
