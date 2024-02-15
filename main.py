# %%
from torch.optim import Adam
from torch_geometric.nn import to_hetero

from src.engine import Trainer
from src.graph_dataset import GraphDataset, GraphDatasetConfig
from src.models import GAT, GNN
from src.utils import set_seeds

set_seeds(42)
graph_dataset = GraphDataset(
    GraphDatasetConfig(dataset_name="mr", doc_doc_k=10, word_word_k=10)
)

model1 = GNN(hidden_channels=64, out_channels=graph_dataset.n_class)
model1 = to_hetero(model1, graph_dataset.hetero_data.metadata(), aggr="max")

# %%
model2 = GAT(
    hidden_channels=64, out_channels=graph_dataset.n_class, use_edge_attr=False
)
model2 = to_hetero(model2, graph_dataset.hetero_data.metadata(), aggr="max")

optimizer1 = Adam(model1.parameters(), lr=0.01, weight_decay=5e-4)
optimizer2 = Adam(model2.parameters(), lr=0.01, weight_decay=5e-4)

trainer1 = Trainer(model1, optimizer1, graph_dataset)
trainer2 = Trainer(model2, optimizer2, graph_dataset)

trainer1.pipeline(max_epochs=100, patience=10)
trainer2.pipeline(max_epochs=100, patience=10)
# %%
d = graph_dataset.hetero_data
