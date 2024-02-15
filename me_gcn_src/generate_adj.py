from math import log

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def ordered_word_pair(a, b):
    if a > b:
        return (b, a)
    else:
        return (a, b)


def get_adj_list(
    tokenized_docs,
    word_list,
    node_size,
    word_id_map,
    train_size,
    vocab_length,
    word_emb_dict,
    doc2vec_npy,
):
    tfidf_row = []
    tfidf_col = []
    tfidf_weight = []

    # get each word appears in which document
    word_doc_list = {}
    for word in word_list:
        word_doc_list[word] = []

    for i in range(len(tokenized_docs)):
        doc_words = tokenized_docs[i]
        unique_words = set(doc_words)
        for word in unique_words:
            exsit_list = word_doc_list[word]
            exsit_list.append(i)
            word_doc_list[word] = exsit_list

    # document frequency
    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    # term frequency
    doc_word_freq = {}

    for doc_id in range(len(tokenized_docs)):
        words = tokenized_docs[doc_id]
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + "," + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for i in range(len(tokenized_docs)):
        words = tokenized_docs[i]
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + "," + str(j)
            freq = doc_word_freq[key]
            if i < train_size:
                row_tmp = i
            else:
                row_tmp = i + vocab_length
            col_tmp = train_size + j

            idf = log(1.0 * len(tokenized_docs) / word_doc_freq[word_list[j]])
            weight_tmp = freq * idf
            doc_word_set.add(word)

            tfidf_row.append(row_tmp)
            tfidf_col.append(col_tmp)
            tfidf_weight.append(weight_tmp)

            tfidf_row.append(col_tmp)
            tfidf_col.append(row_tmp)
            tfidf_weight.append(weight_tmp)

    for i in range(node_size):
        tfidf_row.append(i)
        tfidf_col.append(i)
        tfidf_weight.append(1)

    co_dict = {}
    for sent in tokenized_docs:
        for i, word1 in enumerate(sent):
            for word2 in sent[i:]:
                co_dict[ordered_word_pair(word_id_map[word1], word_id_map[word2])] = 1

    co_occur_threshold = 15

    doc_vec_bow = []
    for sent in tokenized_docs:
        temp = np.zeros(vocab_length)
        for word in sent:
            temp[word_id_map[word]] = 1
        doc_vec_bow.append(temp)
    co_doc_dict = {}
    for i in range(len(doc_vec_bow) - 1):
        for j in range(i + 1, len(doc_vec_bow)):
            if np.dot(doc_vec_bow[i], doc_vec_bow[j]) >= co_occur_threshold:
                co_doc_dict[(i, j)] = 1

    adj_list = []

    for i in tqdm(range(25)):
        col = tfidf_col[:]
        row = tfidf_row[:]
        weight = tfidf_weight[:]
        for pair in co_dict:
            ind1, ind2 = pair

            word1 = word_list[ind1]
            word2 = word_list[ind2]
            tmp = np.tanh(1 / np.abs(word_emb_dict[word1][i] - word_emb_dict[word2][i]))

            row.append(ind2 + train_size)
            col.append(ind1 + train_size)
            weight.append(tmp)

            row.append(ind1 + train_size)
            col.append(ind2 + train_size)
            weight.append(tmp)

        for pair in co_doc_dict:
            ind1, ind2 = pair
            tmp = np.tanh(1 / np.abs(doc2vec_npy[ind1][i] - doc2vec_npy[ind2][i]))

            if ind1 > train_size:
                ind1 += vocab_length
            if ind2 > train_size:
                ind2 += vocab_length

            row.append(ind2)
            col.append(ind1)
            weight.append(tmp)

            row.append(ind1)
            col.append(ind2)
            weight.append(tmp)

        adj_tmp = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))
        adj_tmp = (
            adj_tmp
            + adj_tmp.T.multiply(adj_tmp.T > adj_tmp)
            - adj_tmp.multiply(adj_tmp.T > adj_tmp)
        )
        adj_tmp = normalize_adj(adj_tmp)
        adj_tmp = sparse_mx_to_torch_sparse_tensor(adj_tmp)
        adj_list.append(adj_tmp)
    return adj_list
