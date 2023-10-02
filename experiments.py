# %%
import torch
import torch.nn.functional as F

from pathlib import Path
import pandas as pd

from main import (
    DocDocGraphType,
    DocWordGraphType,
    WordWordGraphType,
    DocEmbeddingType,
    WordEmbeddingType,
)
from torch_geometric.data import HeteroData
from utils import (
    get_tfidf,
    get_w2v_embeddings,
    get_w2v_knn_graph,
)

from dataclasses import dataclass


@dataclass
class TypeConfig:
    doc_emb_type: DocEmbeddingType
    word_emb_type: WordEmbeddingType
    dd_g_type: DocDocGraphType
    ww_g_type: WordWordGraphType
    dw_g_type: DocWordGraphType


import pickle


class TBD:
    def __init__(self, dataset_name, type: TypeConfig):
        df = pd.read_csv(Path.cwd() / f"data-df/{dataset_name}.csv")

        self.documents = df["doc_content"].values

        with open(Path.cwd() / f"w2v-models/{dataset_name}/vocab.pkl", "rb") as file:
            self.vocab = pickle.load(file)

        self.stoi = {word: i for i, word in enumerate(self.vocab)}
        self.itos = {i: word for i, word in enumerate(self.vocab)}

        self.dataset_name = dataset_name
        self.type = type

        # HeteroData
        self.data = HeteroData()
        self.data["doc"].x = self.get_doc_embeddings()
        self.data["doc"].y = df["label"].values
        self.data["doc"].train_mask = df["train_mask"].values
        self.data["doc"].test_mask = df["test_mask"].values

        self.data["word"].x = self.get_word_embeddings()

        e_i, e_a = self.get_word_word_graph()
        self.data["word", "w2w", "w"].edge_index = e_i
        self.data["word", "w2w", "w"].edge_attr = e_a

        e_i, e_a = self.get_doc_word_graph()
        self.data["doc", "d2w", "word"].edge_index = e_i
        self.data["doc", "d2w", "word"].edge_attr = e_a

        e_i, e_a = self.get_doc_doc_graph()
        self.data["doc", "d2d", "doc"].edge_index = e_i
        self.data["doc", "d2d", "doc"].edge_attr = e_a

    def get_word_embeddings(self):
        if self.type.word_emb_type is WordEmbeddingType.Word2Vec:
            embeddings = get_w2v_embeddings(self.dataset_name)

        elif self.type.word_emb_type is WordEmbeddingType.OneHot:
            pass

        return embeddings

    def get_doc_embeddings(self):
        pass

    def get_word_word_graph(self):
        if self.type.ww_g_type is WordWordGraphType.word2vecKNN:
            edge_index, edge_attr = get_w2v_knn_graph(self.data["word"].x, k=30)

        elif self.type.ww_g_type is WordWordGraphType.PPMI:
            pass

        return edge_index, edge_attr

    def get_doc_doc_graph(self):
        edge_index, edge_attr = 0, 0
        return edge_index, edge_attr

    def get_doc_word_graph(self):
        if self.type.dd_g_type is DocWordGraphType.TFIDF:
            edge_index, edge_attr = get_tfidf(self.documents, self.vocab, self.stoi)

        return edge_index, edge_attr


hetero_graph = TBD(
    dataset_name="mr",
    doc_emb_type=DocEmbeddingType.RoBERTa,
    word_emb_type=WordEmbeddingType.Word2Vec,
    dd_g_type=DocDocGraphType.RoBERTaKNN,
    ww_g_type=WordWordGraphType.word2vecKNN,
    dw_g_type=DocWordGraphType.TFIDF,
)

# %%
