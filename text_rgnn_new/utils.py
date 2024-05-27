import random
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


## General Utils
def set_seeds(seed_no: int = 42):
    random.seed(seed_no)
    np.random.seed(seed_no)
    torch.manual_seed(seed_no)
    torch.cuda.manual_seed_all(seed_no)


def compute_metrics(y_pred: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    y_pred = y_pred.argmax(dim=1).numpy()
    y = y.numpy()

    return {
        "weighted_f1": 100 * f1_score(y, y_pred, average="weighted"),
        "macro_f1": 100 * f1_score(y, y_pred, average="macro"),
        "micro_f1": 100 * f1_score(y, y_pred, average="micro"),
        "acc": 100 * accuracy_score(y, y_pred),
        "mcc": 100 * matthews_corrcoef(y, y_pred),
    }
