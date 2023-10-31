import torch
import random

import numpy as np

from sklearn.metrics import accuracy_score, f1_score

import wandb
import random
import torch
import numpy as np

from pathlib import Path

from src.constants import PROJECT_PATH
from src.dataset import GraphDataset, GraphDatasetConfig

from src.models import GNN, GAT

from torch_geometric.nn import to_hetero


def get_used_params(c):
    used_params = {
        "dataset": {
            "dataset_name": c.dataset["dataset_name"],
            "doc_doc_k": c.dataset["doc_doc_k"],
            "word_word_k": c.dataset["word_word_k"],
        },
        "model": {
            "model_name": c.model["model_name"],
            "hidden_channels": c.model["hidden_channels"],
        },
        "optimizer": {
            "lr": c.optimizer["lr"],
            "weight_decay": c.optimizer["weight_decay"],
        },
        "trainer_pipeline": {
            "max_epochs": c.trainer_pipeline["max_epochs"],
            "patience": c.trainer_pipeline["patience"],
        },
        "seed_no": c.seed_no,
    }
    return used_params


def get_sweep_variables(c):
    config = GraphDatasetConfig(
        dataset_name=c.dataset["dataset_name"],
        doc_doc_k=c.dataset["doc_doc_k"],
        word_word_k=c.dataset["word_word_k"],
    )

    graph_dataset = GraphDataset(config, word_to_word_graph=c.word_to_word_graph)
    n_class = graph_dataset.processed_dataset.n_class
    model = None

    if c.model["model_name"] == "GNN":
        model = GNN(hidden_channels=c.model["hidden_channels"], out_channels=n_class)
    elif c.model["model_name"] == "GAT":
        model = GAT(hidden_channels=c.model["hidden_channels"], out_channels=n_class)

    model = to_hetero(model, graph_dataset.data.metadata(), aggr="sum")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=c.optimizer["lr"],
        weight_decay=c.optimizer["weight_decay"],
    )

    return graph_dataset, model, optimizer


def set_seeds(seed_no: int = 42):
    random.seed(seed_no)
    np.random.seed(seed_no)
    torch.manual_seed(seed_no)
    torch.cuda.manual_seed_all(seed_no)


def compute_metrics(output, labels):
    preds = output.max(1)[1].type_as(labels)
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()
    w_f1 = f1_score(y_true, y_pred, average="weighted")
    macro = f1_score(y_true, y_pred, average="macro")
    micro = f1_score(y_true, y_pred, average="micro")
    acc = accuracy_score(y_true, y_pred)
    return w_f1, macro, micro, acc


def save_checkpoint(ckpt: dict, dataset_name: str, acc, sweep_id):
    test_acc_str = str(round(acc, 5)).replace(".", "_")
    MODELS_DIR = Path.joinpath(
        PROJECT_PATH,
        "model_checkpoints",
        dataset_name,
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    SWEEP_ID_FOLDER = MODELS_DIR.joinpath(sweep_id)
    SWEEP_ID_FOLDER.mkdir(parents=True, exist_ok=True)
    FILE_NAME = f"acc:{test_acc_str}-{wandb.run.id}_ckpt.pth"

    ckpt_path = Path.joinpath(MODELS_DIR, SWEEP_ID_FOLDER, FILE_NAME)
    torch.save(ckpt, ckpt_path)


def find_best_run(target_dataset: str, verbose: bool = False):
    # Define the base directory where your model_checkpoints are located
    base_directory = Path.joinpath(PROJECT_PATH, "model_checkpoints")
    highest_accuracy = 0.0
    highest_accuracy_folder = None
    highest_accuracy_file = None

    # Iterate through the subfolders of the specified dataset
    dataset_directory = base_directory / target_dataset
    dataset_directory.mkdir(parents=True, exist_ok=True)

    if dataset_directory.is_dir():
        for subfolder in dataset_directory.iterdir():
            # Check if it's a directory
            if subfolder.is_dir():
                # Iterate through files in the subfolder
                for file_path in subfolder.iterdir():
                    file_name = file_path.name
                    if file_name.startswith("acc:"):
                        # Extract the accuracy from the file name
                        parts = file_name.split("-")
                        accuracy_str = parts[0].split("acc:")[1].replace("_", ".")
                        try:
                            accuracy = float(accuracy_str)
                            if accuracy > highest_accuracy:
                                highest_accuracy = accuracy
                                highest_accuracy_file = file_path
                        except ValueError:
                            pass

    # Print the highest accuracy and its corresponding folder
    if verbose:
        if highest_accuracy_file is not None:
            print("Highest Accuracy for", target_dataset, ":", highest_accuracy)
            print("File Path:", highest_accuracy_file)
        else:
            print(
                "No .pth files with accuracy found for",
                target_dataset,
                "in the directory structure.",
            )

    return highest_accuracy, highest_accuracy_file


def randomly_select_n_train_by_percentage(mask_tensor, n_percentage):
    # Find the indices that are already 1 in the mask_tensor
    nonzero_indices = torch.nonzero(mask_tensor).squeeze()

    # Calculate the number of training instances based on the percentage of 1s
    n_train = int(n_percentage * len(nonzero_indices))

    if n_train == 0:
        # If the percentage results in zero training instances, return the original mask_tensor
        return mask_tensor

    # Shuffle the indices of the nonzero entries
    shuffled_indices = nonzero_indices[torch.randperm(len(nonzero_indices))]

    # Initialize a new mask tensor with all zeros
    new_mask_tensor = torch.zeros_like(mask_tensor)

    # Set the first n_train shuffled indices to 1
    new_mask_tensor[shuffled_indices[:n_train]] = 1

    return new_mask_tensor
