import pickle
from pathlib import Path
from typing import List

from finetuning_encoders.utils import generate_required_dataset_format
from gensim.models import Word2Vec


class Word2VecTrainer:
    def __init__(self, dataset_name, path: Path):
        self.dataset_name = dataset_name
        self.folder_path = path.joinpath(dataset_name)
        self.folder_path.mkdir(parents=True, exist_ok=True)

    def pipeline(self):

        self.model_path = self.folder_path.joinpath("model.bin")
        self.vocab_path = self.folder_path.joinpath("vocab.pkl")

        if not self.check_exists():
            dataset = generate_required_dataset_format(
                self.dataset_name, train_mask=None
            )
            self.fit(dataset.documents)

    def fit(self, sentences: List[str]) -> None:
        model = Word2Vec(
            sentences,
            vector_size=300,
            window=10,
            min_count=10,
            epochs=100,
            workers=8,
        )
        vocab = list(model.wv.index_to_key)

        with open(self.vocab_path, "wb") as file:
            pickle.dump((vocab,), file)

        model.save(str(self.model_path))

    def check_exists(self):
        if self.model_path.is_file() and self.vocab_path.is_file():
            # print(f"W2V Model for '{self.dataset_name}' already exists.")
            # print(f"W2V Vocab for '{self.dataset_name}' already exists.")
            return True
        else:
            return False
