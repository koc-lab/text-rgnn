# %%
import time

from torch.optim import Adam
from torch_geometric.nn import to_hetero

from src.engine import Trainer
from src.graph_dataset import GraphDataset, GraphDatasetConfig
from src.models import GraphSAGE

configurations_list = []
for dataset_name in ["cola"]:
    for ratio in [0.20, 1.0]:
        config = GraphDatasetConfig(
            dataset_name=dataset_name,
            train_ratio=ratio,
            use_w2w_graph=False,
            doc_doc_k=30,
            word_word_k=30,
        )
        configurations_list.append(config)


for config in configurations_list:
    start = time.time()
    graph_dataset = GraphDataset(config)
    print(f"Time to generate graph for {config.dataset_name} :", time.time() - start)
    n_class = graph_dataset.text_dataset.n_class
    model = GraphSAGE(64, n_class)
    model = to_hetero(model, graph_dataset.data.metadata(), aggr="mean")
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    trainer = Trainer(model, optimizer, graph_dataset)
    trainer.pipeline(max_epochs=100, patience=10)


# %%
