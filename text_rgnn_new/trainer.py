# %%
import copy

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import wandb
from text_rgnn_new.datamodule import GraphDataset
from text_rgnn_new.utils import compute_metrics


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        graph_dataset: GraphDataset,
    ):
        self.optimizer = optimizer
        self.model = model
        self.data = graph_dataset.data

        self.train_mask = self.data["doc"].train_mask
        self.test_mask = self.data["doc"].test_mask
        self.y = self.data["doc"].y.reshape(-1)

        self.metric = "mcc" if graph_dataset.c.dataset_name == "cola" else "acc"
        self.graph_dataset = graph_dataset

    def pipeline(self, max_epochs: int, patience: int, wandb_flag: bool = False):
        self.counter = 0
        self.best_metric_val = 0
        self.best_model = None

        t = tqdm(range(max_epochs), desc="Training", leave=False)
        for epoch in t:
            self.e_loss = self.train_epoch()
            self.train_metrics, self.test_metrics = self.eval()
            self.logger(wandb_flag, epoch)

            metric_val = self.test_metrics[self.metric]
            self.give_description(t, patience)

            if metric_val > self.best_metric_val:
                self.best_metric_val = metric_val
                self.best_model = copy.deepcopy(self.model)
                self.counter = 0
            else:
                self.counter += 1

            if self.counter >= patience:
                break

    def train_epoch(self):
        loss = 0
        self.model.train()
        self.optimizer.zero_grad()

        out_full = self.model(
            self.data.x_dict,
            self.data.edge_index_dict,
            self.data.edge_attr_dict,
        )

        out = out_full["doc"]
        out = F.log_softmax(out, dim=1)
        loss = F.nll_loss(out[self.train_mask == 1], self.y[self.train_mask == 1])
        loss.backward()
        self.optimizer.step()
        return loss

    def eval(self):
        with torch.no_grad():
            self.model.eval()

            out_full = self.model(
                self.data.x_dict, self.data.edge_index_dict, self.data.edge_attr_dict
            )

            out = out_full["doc"]
            out = F.log_softmax(out, dim=1)

            y_train = self.y[self.train_mask == 1]
            y_test = self.y[self.test_mask == 1]
            y_train_pred = out[self.train_mask == 1]
            y_test_pred = out[self.test_mask == 1]

            train_metrics = compute_metrics(y_train_pred, y_train)
            test_metrics = compute_metrics(y_test_pred, y_test)

            return train_metrics, test_metrics

    def give_description(self, t: tqdm, patience):
        t.set_description(
            f"Loss: {self.e_loss:.4f}"
            + f" Best Test {self.metric}: {self.best_metric_val:.3f},"
            + f" Train {self.metric}: {self.train_metrics[self.metric]:.3f},"
            + f" Test {self.metric}: {self.test_metrics[self.metric]:.3f}"
            + f" Counter: {self.counter}/{patience}"
        )

    def logger(self, wandb_flag: bool, epoch: int):
        if wandb_flag:

            wandb.log(
                data={"train/loss": self.e_loss},
                step=epoch,
            )

            wandb.log(
                data={f"train/train_{self.metric}": self.train_metrics[self.metric]},
                step=epoch,
            )

            wandb.log(
                data={f"test/test_{self.metric}": self.test_metrics[self.metric]},
                step=epoch,
            )
