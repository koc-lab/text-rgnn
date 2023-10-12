from dataclasses import dataclass
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


class KNNType(Enum):
    # TODO: add other types in future
    ZeroMask = auto()
    SoftMax = auto()


@dataclass
class TypeConfig:
    doc_emb_type: DocEmbeddingType
    word_emb_type: WordEmbeddingType
    dd_g_type: DocDocGraphType
    ww_g_type: WordWordGraphType
    dw_g_type: DocWordGraphType
