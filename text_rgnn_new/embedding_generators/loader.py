import numpy as np
import torch
from gensim.models import Word2Vec

from text_rgnn_new import FINE_TUNED_ENCODERS_PATH, TF_IDF_GRAPHS_PATH, W2V_MODELS_PATH


class EmbeddingLoader:
    def __init__(self, dataset_name: str, train_percentage):
        self.dataset_name = dataset_name
        self.train_percentage = train_percentage
        saving_convention = f"roberta-base-{dataset_name}-{train_percentage}"

        self.doc_emb_dir = FINE_TUNED_ENCODERS_PATH.joinpath(
            saving_convention, "embeddings.pth"
        )

        self.train_mask_dir = FINE_TUNED_ENCODERS_PATH.joinpath(
            saving_convention, "train_mask.pth"
        )

    def load_word_embeddings(self):
        path = W2V_MODELS_PATH.joinpath(f"{self.dataset_name}", "model.bin")
        model = Word2Vec.load(str(path))
        wv = model.wv
        embeddings = np.array([wv[word] for word in wv.index_to_key])
        embeddings = torch.tensor(embeddings, dtype=torch.float)
        return embeddings

    def load_tfidf_graph(self):
        path = TF_IDF_GRAPHS_PATH.joinpath(f"{self.dataset_name}.pth")
        edge_index, edge_attr = torch.load(path)
        return edge_index, edge_attr

    def load_doc_embeddings(self):
        embeddings = torch.load(self.doc_emb_dir)
        return embeddings

    def load_train_mask(self):
        train_mask = torch.load(self.train_mask_dir)
        return train_mask
