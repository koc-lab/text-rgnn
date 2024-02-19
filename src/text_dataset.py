import pickle
from typing import List

import torch

from src import GLUE_DATA_PATH, LABEL_TO_INT_MAP, ORIGINAL_DATA_PATH


class TextDataset:
    def __init__(self, dataset_name):

        if dataset_name in ["mr", "R52", "R8", "ohsumed"]:
            self.pipeline1(dataset_name)
        elif dataset_name in ["sst2", "cola"]:
            self.pipeline2(dataset_name)

    def pipeline1(self, dataset_name):
        self.documents, raw_label, train_ids, test_ids = self.read_data(dataset_name)
        size = len(train_ids) + len(test_ids)

        self.train_mask = torch.tensor(ids_to_mask(train_ids, size)).reshape(-1)
        self.test_mask = torch.tensor(ids_to_mask(test_ids, size)).reshape(-1)
        self.labels = map_labels_to_int(raw_label, dataset_name)
        self.int_to_label = {v: k for k, v in LABEL_TO_INT_MAP[dataset_name].items()}
        self.raw_label = raw_label
        self.n_class = len(list(set(raw_label)))

    def pipeline2(self, dataset_name):
        file_path = GLUE_DATA_PATH.joinpath(f"{dataset_name}_raw_data.pkl")
        raw_data = pickle.load(open(file_path, "rb"))
        
        if dataset_name == "sst2":
            #! for sst2 select only 30k samples since it has 67k samples
            train = raw_data["train"].to_pandas()[:30000]
        else:
            train = raw_data["train"].to_pandas()

        test = raw_data["validation"].to_pandas()

        self.documents = train["sentence"].tolist() + test["sentence"].tolist()

        self.documents = [doc.lower() for doc in self.documents]

        train_labels = train["label"].tolist()
        test_labels = test["label"].tolist()
        train_size = len(train_labels)
        test_size = len(test_labels)

        self.raw_label = train_labels + test_labels
        self.labels = map_labels_to_int(self.raw_label, dataset_name)

        self.train_mask = torch.tensor([1] * train_size + [0] * test_size).reshape(-1)
        self.test_mask = torch.tensor([0] * train_size + [1] * test_size).reshape(-1)
        self.int_to_label = {v: k for k, v in LABEL_TO_INT_MAP[dataset_name].items()}
        self.n_class = len(list(set(self.raw_label)))

    #! For MR,R52,R8,OHSUMED
    def read_data(self, dataset_name):
        raw_label, train_ids, test_ids = [], [], []
        documents = []

        raw_data_path = ORIGINAL_DATA_PATH.joinpath(f"label-info/{dataset_name}.txt")

        file = open(raw_data_path)
        lines = file.readlines()

        for i, line in enumerate(lines):
            raw_label.append(line.strip().split("\t")[2])
            temp = line.split("\t")
            if temp[1].find("test") != -1:
                test_ids.append(i)
            elif temp[1].find("train") != -1:
                train_ids.append(i)
        file.close()

        raw_data_path = ORIGINAL_DATA_PATH.joinpath(f"corpus/clean/{dataset_name}.txt")
        file = open(raw_data_path)
        lines = file.readlines()
        for line in lines:
            documents.append(line.strip())
        file.close()
        return documents, raw_label, train_ids, test_ids

    # def save(self, dataset_name: str):
    #     file_path = Path.cwd() / "data" / "processed-data" / f"{dataset_name}.pkl"
    #     with open(file_path, "wb") as file:
    #         pickle.dump(self, file)


def ids_to_mask(ids, total_size):
    return [1 if i in ids else 0 for i in range(total_size)]


def map_labels_to_int(original_labels: List[str], dataset_name) -> List[int]:
    label_to_int = LABEL_TO_INT_MAP[dataset_name]
    y = [label_to_int[label] for label in original_labels]
    return torch.tensor(y).reshape(-1, 1)
