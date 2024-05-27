import pickle
from pathlib import Path

import numpy as np
import torch
from finetuning_encoders.utils import generate_required_dataset_format
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFTrainer:
    def __init__(self, dataset_name: str, saving_path: Path, vocab_path: Path):
        self.dataset_name = dataset_name
        self.saving_path = saving_path
        self.saving_path.mkdir(parents=True, exist_ok=True)
        self.vocab_path = vocab_path

        self.dataset = generate_required_dataset_format(dataset_name, train_mask=None)
        self.vocab = self.load_vocab()

    def load_vocab(self):

        with open(self.vocab_path, "rb") as file:
            vocab = pickle.load(file)
            vocab = vocab[0]
        return vocab

    def fit(self):
        vectorizer = TfidfVectorizer(vocabulary=self.vocab)
        tfidf_matrix = vectorizer.fit_transform(self.dataset.documents)
        A = coo_matrix(tfidf_matrix)

        edge_index = torch.tensor(np.array([A.row, A.col]), dtype=torch.long)
        edge_attr = torch.tensor(A.data, dtype=torch.float32).view(-1)
        return edge_index, edge_attr

    def pipeline(self):
        file_path = self.saving_path.joinpath(f"{self.dataset_name}.pth")
        if file_path.exists():
            # print(f"TF-IDF graph for {self.dataset_name} already exists. Skipping...")
            return
        edge_index, edge_attr = self.fit()
        torch.save((edge_index, edge_attr), file_path)
