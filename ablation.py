# %%
import torch
from src.engine import Trainer
from src.utils import (
    set_seeds,
    save_checkpoint,
    get_used_params,
    find_best_run,
    get_sweep_variables,
)
import torch

from src.dataset import GraphDataset, GraphDatasetConfig
from src.models import GNN

from torch_geometric.nn import to_hetero


set_seeds(seed_no=42)

config = GraphDatasetConfig(dataset_name="mr", doc_doc_k=20, word_word_k=20)

graph_dataset = GraphDataset(config, word_to_word_graph=True)
n_class = graph_dataset.processed_dataset.n_class

model = GNN(hidden_channels=64, out_channels=n_class)

model = to_hetero(model, graph_dataset.data.metadata(), aggr="sum")
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5,
)

trainer = Trainer(model, optimizer, graph_dataset)
# trainer.train_mask = train_mask_new

trainer.pipeline(
    max_epochs=100,
    patience=10,
    wandb_flag=False,
)
# %%
