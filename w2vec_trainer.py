from gensim.test.utils import datapath
from gensim import utils
from pathlib import Path
from gensim.models import Word2Vec
import pickle


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def __iter__(self):
        corpus_path = datapath(
            Path.cwd() / f"original-data/corpus/clean/{self.dataset_name}.txt"
        )
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


def train_w2v_model(dataset_name: str):
    # Check if there existing model for this dataset

    # Define the directory path
    dir_path = Path.cwd() / "w2v-models" / f"{dataset_name}"
    file_name = "model.bin"  # Replace with your file name

    # Create a Path object for the file
    file_path = Path(dir_path) / file_name

    # Check if the file exists
    if file_path.is_file():
        print(f"The file '{file_name}' exists in the directory.")
        print(f"Model for '{dataset_name}' already exists.")

        # Save the vocab of model
        model = Word2Vec.load(str(dir_path / file_path))
        vocab = (list(model.wv.index_to_key),)

        # Your list of strings

        # Specify the file path where you want to save the pickle file
        vocab_file_path = dir_path / "vocab.pkl"

        # Open the file in binary write mode and save the list to the pickle file
        with open(vocab_file_path, "wb") as file:
            pickle.dump(vocab, file)

        print(f"The vocab has been saved to {file_path}")

        # Save the vocab to a file

    else:
        print(f"The file '{file_name}' does not exist in the directory.")
        print(f"Model for '{dataset_name}' does not exist.")

        sentences = MyCorpus(dataset_name=dataset_name)
        model = Word2Vec(
            sentences=sentences,
            vector_size=300,
            window=10,
            min_count=10,
            workers=1,
            epochs=100,
        )

        dir_path.mkdir(parents=True, exist_ok=True)
        model.save(str(dir_path / file_name))

        print(f"Model for '{dataset_name}' is created.")


if __name__ == "__main__":
    train_w2v_model("mr")
    train_w2v_model("R8")
    train_w2v_model("R52")
    train_w2v_model("ohsumed")
