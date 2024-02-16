import pytest
import torch
from torch.optim import Adam
from torch_geometric.nn import to_hetero

from src.engine import Trainer
from src.graph_dataset import GraphDataset, GraphDatasetConfig
from src.models import GAT, GraphSAGE


@pytest.fixture
def configurations():
    # Initialize any variables required for testing
    configurations_list = []
    for dataset_name in ["ohsumed", "R8", "R52", "mr"]:
        for ratio in [0.01, 0.05, 0.20, 1.0]:
            for use_w2w_graph in [True, False]:
                config = GraphDatasetConfig(
                    dataset_name=dataset_name,
                    train_ratio=ratio,
                    use_w2w_graph=use_w2w_graph,
                    doc_doc_k=10,
                    word_word_k=10,
                )
                configurations_list.append(config)
    return configurations_list


def test_train_mask(configurations):
    for config in configurations:
        graph_dataset = GraphDataset(config)
        original_mask = graph_dataset.text_dataset.train_mask
        new_mask = graph_dataset.train_mask

        check_shape_condition(original_mask, new_mask)
        check_subset_condition(original_mask, new_mask)
        check_tensor_condition(original_mask, new_mask)
        check_ratio_condition(original_mask, new_mask, config.train_ratio)


def test_model_run(configurations):
    for config in configurations:
        graph_dataset = GraphDataset(config)
        n_class = graph_dataset.text_dataset.n_class
        model_settings = [("GAT", True), ("GAT", False), ("GraphSAGE", False)]
        for conv_layer, use_edge_attr in model_settings:
            if conv_layer == "GAT":
                model = GAT(64, n_class, use_edge_attr=use_edge_attr)
            else:
                model = GraphSAGE(64, n_class)
            model = to_hetero(model, graph_dataset.data.metadata(), aggr="max")
            optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            trainer = Trainer(model, optimizer, graph_dataset)
            trainer.pipeline(max_epochs=10, patience=2)


def check_shape_condition(original_mask, new_mask):
    assert original_mask.shape == new_mask.shape


def check_subset_condition(original_mask, new_mask):
    nonzeros_original = torch.nonzero(original_mask).squeeze()
    nonzeros_new = torch.nonzero(new_mask).squeeze()
    assert torch.all(torch.isin(nonzeros_new, nonzeros_original))


def check_tensor_condition(original_mask, new_mask):
    assert torch.is_tensor(new_mask) and torch.is_tensor(original_mask)


def check_ratio_condition(original_mask, new_mask, train_ratio):
    original_train_size = torch.sum(original_mask)
    new_train_size = torch.sum(new_mask)
    assert new_train_size == int(train_ratio * original_train_size)
