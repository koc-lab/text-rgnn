# %%
import copy

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import wandb
from src.graph_dataset import GraphDataset
from src.utils import compute_metrics


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        graph_dataset: GraphDataset,
    ):
        self.optimizer = optimizer
        self.model = model

        self.graph_dataset = graph_dataset
        self.data = self.graph_dataset.hetero_data
        self.data_doc = self.data["doc"]

        self.train_mask = self.data_doc.train_mask
        self.test_mask = self.data_doc.test_mask
        self.y = self.data_doc.y.reshape(-1)

    def pipeline(
        self,
        max_epochs: int,
        patience: int,
        wandb_flag: bool = False,
    ):
        early_stopping = EarlyStopping(patience=patience, verbose=False)

        t = tqdm(range(max_epochs))
        for epoch in t:
            self.model.train()
            e_loss = self.train_epoch()

            train_acc, test_acc = self.eval_model()

            if wandb_flag:
                epoch_wandb_log(e_loss, train_acc, test_acc, epoch)

            best_test_acc, best_model = early_stopping(test_acc, self.model, epoch)
            t.set_description(
                f"Loss: {e_loss:.4f}, Best Test Acc: {best_test_acc:.3f}, Train Acc: {train_acc:.3f}"
            )

            if early_stopping.early_stop:
                break

        self.best_model = best_model
        self.best_test_acc = best_test_acc

    def train_epoch(self):
        loss = 0
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(
            self.data.x_dict, self.data.edge_index_dict, self.data.edge_attr_dict
        )["doc"]
        out = F.log_softmax(out, dim=1)

        loss = F.nll_loss(out[self.train_mask == 1], self.y[self.train_mask == 1])

        loss.backward()
        self.optimizer.step()
        return loss

    def eval_model(self):
        with torch.no_grad():
            self.model.eval()
            out = self.model(
                self.data.x_dict, self.data.edge_index_dict, self.data.edge_attr_dict
            )["doc"]
            out = F.log_softmax(out, dim=1)

            w_f1_test, macro_test, micro_test, acc_test = compute_metrics(
                out[self.test_mask == 1], self.y[self.test_mask == 1]
            )

            w_f1_train, macro_train, micro_train, acc_train = compute_metrics(
                out[self.train_mask == 1], self.y[self.train_mask == 1]
            )

            return 100 * acc_train, 100 * acc_test


# %%

## Utility functions for Trainer


def epoch_wandb_log(loss, train_acc, test_acc, epoch):
    wandb.log(data={"train/loss": loss}, step=epoch)
    wandb.log(data={"train/train_acc": train_acc}, step=epoch)
    wandb.log(data={"test/test_accuracy": test_acc}, step=epoch)


class EarlyStopping:
    def __init__(self, patience: int, verbose: bool = False):
        self.patience = patience
        self.verbose = verbose

        self.counter = 0
        self.best_test_acc = 0
        self.best_model = None
        self.early_stop = False

    def __call__(self, test_acc: float, model: torch.nn.Module, epoch: int):
        if test_acc > self.best_test_acc:
            self.counter = 0
            self.best_test_acc = test_acc
            self.best_model = copy.deepcopy(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Early stopping at epoch {epoch}")

        return self.best_test_acc, self.best_model


# %%

# def to_device(data, target, device):
#     return data.to(device), target.to(device)

# def get_existing_model(model_name: str, SAVE_DIR: Path):
#     files_in_dir = os.listdir(SAVE_DIR)

#     for file in files_in_dir:
#         if file.startswith(model_name):
#             best_acc = str_to_acc(filename=file)
#             best_model = torch.load(SAVE_DIR / file)
#             return best_acc, best_model

# def str_to_acc(filename: str) -> float:
#     pattern = r"(\d+_\d+)"
#     match = re.search(pattern, filename)
#     if match:
#         extracted_value = match.group(1)
#         float_value = float(extracted_value.replace("_", "."))
#         return float_value
