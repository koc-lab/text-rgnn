import glob
import re
from os import getenv
from pathlib import Path

import torch
from torch_geometric.nn import to_hetero

import wandb
from text_rgnn_new import DATASET_TO_N_CLASS
from text_rgnn_new.datamodule import GraphDataset, GraphDatasetConfig
from text_rgnn_new.models import GAT, GraphSAGE
from text_rgnn_new.trainer import Trainer

DATASET_NAME = None
TRAIN_PERCENTAGE = None
MONITOR_METRIC = None
GLOBAL_BEST_METRIC = 0.0
CKPT_FOLDER_FOR_SWEEP = None


def train():
    global GLOBAL_BEST_METRIC, CKPT_FOLDER_FOR_SWEEP

    wandb.init()
    config = wandb.config

    dataset_name = config.dataset_name
    monitor_metric = config.monitor_metric
    train_percentage = config.train_percentage
    lr = config.learning_rate
    doc_doc_k = config.doc_doc_k
    word_word_k = config.word_word_k
    use_w2w_graph = config.use_w2w_graph
    hidden_dimension = config.hidden_dimension
    aggr = config.aggr
    wd = config.wd
    model_name = config.model_name

    config = GraphDatasetConfig(
        dataset_name=dataset_name,
        train_percentage=train_percentage,
        doc_doc_k=doc_doc_k,
        word_word_k=word_word_k,
        use_w2w_graph=use_w2w_graph,
    )
    graph_dataset = GraphDataset(config)

    n_class = DATASET_TO_N_CLASS[dataset_name]

    if model_name == "GraphSAGE":
        model = GraphSAGE(hidden_dimension, n_class)

    elif model_name == "GAT":
        model = GAT(hidden_dimension, n_class, use_edge_attr=False)

    else:
        raise ValueError("Invalid model name.")

    model = to_hetero(model, graph_dataset.data.metadata(), aggr=aggr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    trainer = Trainer(model, optimizer, graph_dataset)
    trainer.pipeline(max_epochs=100, patience=20)
    best_test_metric = trainer.best_metric_val

    # retrieve current global best metric
    GLOBAL_BEST_METRIC = check_saved_test_acc()
    print(f"Saved model has metric: {GLOBAL_BEST_METRIC}")
    print(f"New found model has metric: {best_test_metric}")

    if best_test_metric > GLOBAL_BEST_METRIC:
        # delete previous best model
        for file in CKPT_FOLDER_FOR_SWEEP.glob("best_model_*.pth"):
            file.unlink()
        GLOBAL_BEST_METRIC = best_test_metric
        saving_convention = f"best_model_{round(best_test_metric,4)}.pth"
        file_path = CKPT_FOLDER_FOR_SWEEP.joinpath(saving_convention)
        torch.save(trainer.best_model, file_path)

    wandb.log({monitor_metric: best_test_metric})


def check_saved_test_acc():
    global CKPT_FOLDER_FOR_SWEEP
    best_model_files = glob.glob(
        str(CKPT_FOLDER_FOR_SWEEP.joinpath("best_model_*.pth"))
    )

    # Check if any files match the pattern
    if not best_model_files:
        return 0
    else:
        # Load the best model
        best_model_file = best_model_files[0]
        match = re.search(r"best_model_(\d+\.\d+)\.pth", best_model_file)

        if match:
            saved_test_acc = float(match.group(1))
        else:
            raise ValueError("Filename does not match the expected pattern")

    return saved_test_acc


def generate_sweep(dataset_name: str, train_percentage: float, project: str = None):
    global DATASET_NAME, TRAIN_PERCENTAGE, MONITOR_METRIC

    DATASET_NAME = dataset_name
    TRAIN_PERCENTAGE = train_percentage
    MONITOR_METRIC = "test/mcc" if dataset_name == "cola" else "test/acc"

    if project is None:
        project = getenv("WANDB_PROJECT")

    sweep_configuration = {
        "name": f"{DATASET_NAME}-{TRAIN_PERCENTAGE}",
        "method": "bayes",
        "metric": {
            "goal": "maximize",
            "name": MONITOR_METRIC,
        },
        "parameters": {
            "dataset_name": {"value": DATASET_NAME},
            "monitor_metric": {"value": MONITOR_METRIC},
            "train_percentage": {"value": TRAIN_PERCENTAGE},
            "learning_rate": {"min": 1e-3, "max": 1e-1},
            "doc_doc_k": {"values": [10, 15, 20]},
            "word_word_k": {"values": [10, 15, 20]},
            "use_w2w_graph": {"values": [True, False]},
            "hidden_dimension": {"values": [64, 128]},
            "aggr": {"values": ["mean", "max", "sum"]},
            "wd": {"min": 1e-3, "max": 1e-1},
            "model_name": {"values": ["GraphSAGE", "GAT"]},
        },
    }
    sweep_id = wandb.sweep(sweep_configuration, project=project)
    return sweep_id


def add_agent(sweep_id: str, entity: str = None, project: str = None) -> None:
    global DATASET_NAME, TRAIN_PERCENTAGE, MONITOR_METRIC, CKPT_FOLDER_FOR_SWEEP

    if entity is None:
        entity = getenv("WANDB_ENTITY")
    if project is None:
        project = getenv("WANDB_PROJECT")
    if entity is None or project is None:
        raise ValueError("Must specify entity and project.")

    tuner = wandb.controller(sweep_id, entity=entity, project=project)
    parameters = tuner.sweep_config.get("parameters")
    if parameters is not None:
        DATASET_NAME = parameters.get("dataset_name")["value"]
        TRAIN_PERCENTAGE = parameters.get("train_percentage")["value"]
        MONITOR_METRIC = parameters.get("monitor_metric")["value"]

    CKPT_FOLDER_FOR_SWEEP = Path(f"best_models/{DATASET_NAME}-{TRAIN_PERCENTAGE}")
    CKPT_FOLDER_FOR_SWEEP.mkdir(parents=True, exist_ok=True)

    wandb.agent(sweep_id, function=train, count=100)
