# %%
from pathlib import Path
import torch
import wandb
import torch.nn.functional as F
from tqdm.auto import tqdm
import copy
from src.dataset import GraphDataset


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data: GraphDataset,
        device: torch.device,
    ):
        self.optimizer = optimizer
        self.data = data
        self.device = device

        self.model = model.to(device)

    def pipeline(
        self,
        max_epochs: int,
        patience: int,
        wandb_flag: bool = True,
        early_stop_verbose: bool = False,
    ):
        early_stopping = EarlyStopping(patience=patience, verbose=early_stop_verbose)

        t = tqdm(range(max_epochs))
        for epoch in t:
            self.model.train()
            e_loss = self.train_epoch()

            train_acc, test_acc = self.eval_model(self.data)

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

        self.data.X = self.data.X.to(self.device)
        self.data.y = self.data.y.to(self.device)

        out = self.model(self.data.X)

        loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss

    def eval_model(self, data: GraphDataset):
        with torch.no_grad():
            self.model.eval()
            _, pred = self.model(self.data.X).max(dim=1)

            test_acc = (
                float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
                / data.test_mask.sum().item()
            )

            train_acc = (
                float(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
                / data.train_mask.sum().item()
            )

            return 100.0 * train_acc, 100.0 * test_acc


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
