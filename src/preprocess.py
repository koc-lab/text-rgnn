from dataclasses import dataclass
from typing import List


@dataclass
class ProcessedDataset:
    text: List[str]
    train_mask: List[int]
    test_mask: List[int]
    label: List[int]
    label_to_int: dict
    int_to_label: dict
    label_original: List[str]
    n_class: int
    train_size: int
    test_size: int
