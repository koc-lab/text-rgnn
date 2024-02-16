import pickle
from pathlib import Path
from typing import List

from gensim.models import Word2Vec

from src import W2V_MODELS_PATH
from src.text_dataset import TextDataset


def w2v_pipeline(dataset_name: str):

    folder_path = W2V_MODELS_PATH / dataset_name
    folder_path.mkdir(parents=True, exist_ok=True)
    model_path = folder_path / "model.bin"
    vocab_path = folder_path / "vocab.pkl"

    if not check_model_and_vocab_path(dataset_name, model_path, vocab_path):
        textdataset = TextDataset(dataset_name)
        sentences = textdataset.documents
        train_w2v_model(sentences, model_path, vocab_path)


def train_w2v_model(sentences: List[str], model_path: Path, vocab_path: Path) -> None:
    model = Word2Vec(sentences, vector_size=300, window=10, min_count=10, epochs=100)
    vocab = list(model.wv.index_to_key)

    with open(vocab_path, "wb") as file:
        pickle.dump((vocab,), file)

    model.save(str(model_path))


def check_model_and_vocab_path(dataset_name: str, path1: Path, path2: Path):
    if path1.is_file() and path2.is_file():
        print(f"Model for '{dataset_name}' already exists.")
        print(f"Vocab for '{dataset_name}' already exists.")
        return True
    else:
        return False
