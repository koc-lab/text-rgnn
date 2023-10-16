# %%
import torch
from torch_geometric.nn import to_hetero
from src.models import GNN, GAT
from src.utils import set_seeds

from src.dataset import GraphDataset, DatasetNames, GraphDatasetConfig
from src.engine import Trainer

config = GraphDatasetConfig(dataset_name=DatasetNames.MR, doc_doc_k=20, word_word_k=20)
graph_dataset = GraphDataset(config)
processed_dataset = graph_dataset.processed_dataset

seed_no = 43
set_seeds(seed_no)

print("**************GAT RESULTS***************")
model = GAT(hidden_channels=64, out_channels=processed_dataset.n_class)
model = to_hetero(model, graph_dataset.data.metadata(), aggr="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
trainer = Trainer(model, optimizer, graph_dataset)
trainer.pipeline(max_epochs=50, patience=10, wandb_flag=False)


print("**************GNN RESULTS***************")
model = GNN(hidden_channels=64, out_channels=processed_dataset.n_class)
model = to_hetero(model, graph_dataset.data.metadata(), aggr="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
trainer = Trainer(model, optimizer, graph_dataset)
trainer.pipeline(max_epochs=50, patience=10, wandb_flag=False)

# %%
