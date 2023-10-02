from enum import Enum, auto
import numpy as np
import torch
import torch.nn.functional as F
from gensim.models.word2vec import KeyedVectors
from gensim.models import Word2Vec
from pathlib import Path


class KNNType(Enum):
    # TODO: add other types in future
    ZeroMask = auto()
    SoftMax = auto()


def get_tfidf(documents, vocab, stoi):
    """
    calculating term frequency
    """
    word_freq = {word: 0 for word in vocab}
    for doc in documents:
        words = [word for word in doc.split() if word in vocab]
        for word in words:
            word_freq[word] += 1

    doc_word_freq = {}
    for doc_id, doc in enumerate(documents):
        words = [word for word in doc.split() if word in vocab]
        for word in words:
            word_id = stoi[word]
            doc_word_str = str(doc_id) + "," + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    """
        calculating inverse document frequency
    """
    row_nf, col_nf, weight_nf = [], [], []

    for i, doc in enumerate(documents):
        words = [word for word in doc.split() if word in vocab]
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = stoi[word]
            key = str(i) + "," + str(j)
            freq = doc_word_freq[key]

            row_nf.append(i)
            col_nf.append(j)
            idf = np.log(1.0 * len(documents) / word_freq[vocab[j]])

            weight_nf.append(freq * idf + 1e-6)
            doc_word_set.add(word)

    tensor1 = torch.tensor(row_nf)
    tensor2 = torch.tensor(col_nf)

    # Stack the two tensors vertically (along the first dimension)
    edge_index = torch.stack([tensor1, tensor2], dim=0)
    edge_attr = torch.tensor(weight_nf)
    return edge_index, edge_attr


def get_knn_from_similarity_matrix(
    A: torch.Tensor,
    k: int = 10,
    strategy: KNNType = KNNType.ZeroMask,
):
    """
    Creating a KNN graph from similarity matrix

    Strategy:
        - ZeroMask: zero out the similarity matrix except for the top k similarities
        - SoftMax: fill the zero out entries with -inf and apply softmax
    """

    I = torch.eye(A.size(0), dtype=A.dtype)
    A_hat = A - I * A
    _, top_k_indices = torch.topk(A_hat, k, dim=1)
    mask = torch.zeros_like(A_hat)
    mask.scatter_(1, top_k_indices, 1)
    A_hat_masked = A_hat * mask

    if strategy is KNNType.ZeroMask:
        return A_hat_masked
    else:
        A_hat_masked[A_hat_masked == 0] = -float("inf")
        return F.softmax(A_hat_masked, dim=1)


def sanity_check_for_embedding_matrix(
    wv: KeyedVectors,
    similarity_matrix: torch.tensor,
    stoi: dict,
    w1: str = "bad",
    w2: str = "good",
):
    w1_vec = torch.tensor(wv[w1], dtype=torch.float).reshape(1, -1)
    w2_vec = torch.tensor(wv[w2], dtype=torch.float).reshape(1, -1)

    print(f"Word2Vec Simlarity:{wv.similarity(w1, w2)}")
    print(f"Cosine Similarity: {F.cosine_similarity(w1_vec, w2_vec)}")
    print(f"Similartiy matrix entry: {similarity_matrix[stoi[w1], stoi[w2]]}")


def get_w2v_embeddings(dataset_name: str):
    path = Path.cwd() / "w2v-models" / f"{dataset_name}" / "model.bin"
    word2vec_model = Word2Vec.load(str(path))
    wv = word2vec_model.wv
    embeddings = [wv[word] for word in wv.index_to_key]
    embeddings = torch.tensor(embeddings, dtype=torch.float)
    return embeddings


def get_w2v_knn_graph(X: torch.tensor, k: int = 10):
    A = F.cosine_similarity(X.unsqueeze(1), X.unsqueeze(0), dim=2)
    A = get_knn_from_similarity_matrix(A, k=k, strategy=KNNType.ZeroMask)
    edge_index = torch.nonzero(A, as_tuple=False).t()
    edge_attr = A[edge_index[0], edge_index[1]]
    return edge_index, edge_attr
