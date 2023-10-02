# Constructing a heterogeneous graph
from torch_geometric.data import HeteroData

data = HeteroData()

data["documents"].x = ...  # [n_docs, n_doc_feats]
data["words"].x = ...  # [n_words, n_word_feats]

data["documents", "doc-to-doc", "documents"].edge_index = ...  # [2, n_doc_to_doc_edges]
data["words", "word-to-word", "words"].edge_index = ...  # [2, n_word_to_word_edges]
data["documents", "doc-to-word", "words"].edge_index = ...  # [2, n_doc_to_word_edges]

# Attributes are scalar values associated to edges.

data["documents", "doc-to-doc", "documents"].edge_attr = ...  # [n_edges_doc_to_doc, 1]
data["words", "word-to-word", "words"].edge_attr = ...  # [n_edges_word_to_word, 1]
data["documents", "doc-to-word", "words"].edge_attr = ...  # [n_edges_doc_to_word, 1]

from enum import Enum, auto


class DocEmbeddingType(Enum):
    RoBERTa = auto()


class WordEmbeddingType(Enum):
    Word2Vec = auto()
    OneHot = auto()

    # For future implementation
    GloVe = auto()
    FastText = auto()


class DocDocGraphType(Enum):
    RoBERTaKNN = auto()


class WordWordGraphType(Enum):
    word2vecKNN = auto()
    PPMI = auto()


class DocWordGraphType(Enum):
    TFIDF = auto()
    BM25 = auto()

    #! to use strategy below, we need to have same dimensions for doc and word embeddings
    RoBERTaCROSSword2vecKNN = auto()
