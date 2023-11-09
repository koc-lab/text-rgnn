# %%
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures, GCNNorm
import numpy as np
from src.hetero_generator import (
    load_variables,
    get_doc_embeddings,
    get_word_embeddings,
    get_knn_graph,
    get_doc_word_graph,
)
from torch_geometric.data import Data

import numpy as np


def get_homogenous_data(dataset_name: str):
    doc_doc_k = 50
    word_word_k = 50
    processed_dataset, vocab, stoi, itos = load_variables(dataset_name)

    doc_x = get_doc_embeddings(dataset_name).to("cpu")
    word_x = get_word_embeddings(dataset_name)

    doc_y = processed_dataset.label
    doc_train_mask = processed_dataset.train_mask
    doc_test_mask = processed_dataset.test_mask

    # Doc-Doc Graph
    e_i, e_a, N = get_knn_graph(doc_x, n_neighbours=doc_doc_k)
    e_i, e_a, V = get_knn_graph(word_x, n_neighbours=word_word_k)

    N = N.toarray()
    V = V.toarray()

    e_i, e_a = get_doc_word_graph(dataset_name)
    n = e_i[0].max() + 1  # Assuming document nodes are in the first row of e_i
    m = e_i[1].max() + 1  # Assuming word nodes are in the second row of e_i

    # Initialize an empty n x m matrix filled with zeros
    K = np.zeros((n, m))

    # Populate the adjacency matrix based on the edge information
    # for edge, weight in zip(e_i.T, e_a):
    #     document_index, word_index = edge
    #     K[document_index][word_index] = weight

    n = N.shape[0]
    m = V.shape[0]

    A = np.zeros((n + m, n + m))
    A[:n, :n] = N
    A[n : n + m, n : n + m] = V
    A[n : n + m, :n] = K.T
    A[:n, n : n + m] = K

    train_mask = torch.zeros(n + m, dtype=doc_train_mask.dtype)
    test_mask = torch.zeros(n + m, dtype=doc_test_mask.dtype)
    train_mask[:n] = doc_train_mask
    test_mask[:n] = doc_test_mask
    padding_size = doc_x.shape[1] - word_x.shape[1]

    if padding_size > 0:
        padding = torch.zeros(m, padding_size)
        B_padded = torch.cat((word_x, padding), dim=1)
    else:
        B_padded = word_x

    X = torch.cat((doc_x, B_padded), dim=0)
    y = torch.cat((doc_y, torch.zeros((m, 1), dtype=doc_y.dtype)), dim=0)
    row_indices, col_indices = A.nonzero()
    edge_index = np.vstack((row_indices, col_indices))
    edge_attr = A[row_indices, col_indices]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # edge_attr = gcn_norm(edge_index, edge_attr)
    data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.n_class = processed_dataset.n_class
    # data = T.ToUndirected()(data)
    # data = T.AddSelfLoops()(data)
    # data = T.NormalizeFeatures()(data)
    return data, A


class GCN(torch.nn.Module):
    def __init__(self, n_feats, n_class, n_hidden):
        super().__init__()
        self.conv1 = GCNConv(n_feats, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_class)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.conv2(x, edge_index, edge_attr)

        return F.log_softmax(x, dim=1)


def eval_model(model, data):
    with torch.no_grad():
        model.eval()
        out = model(data)
        out = F.log_softmax(out, dim=1)

        _, _, _, train_acc = compute_metrics(
            out[data.train_mask == 1], data.y[data.train_mask == 1].reshape(-1)
        )
        _, _, _, test_acc = compute_metrics(
            out[data.test_mask == 1], data.y[data.test_mask == 1].reshape(-1)
        )

        return 100 * train_acc, 100 * test_acc


def pipeline(model, optimizer, data, max_epochs):
    t = tqdm(range(max_epochs))
    best_test_acc = 0
    for epoch in t:
        model.train()
        optimizer.zero_grad()
        out = model(data)
        e_loss = F.nll_loss(
            out[data.train_mask == 1], data.y[data.train_mask == 1].reshape(-1)
        )
        e_loss.backward()
        optimizer.step()

        train_acc, test_acc = eval_model(model, data)
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        t.set_description(
            f"Loss: {e_loss:.4f}, Best Test Acc: {best_test_acc:.3f}, Train Acc: {train_acc:.3f}"
        )

    return best_test_acc


from tqdm.auto import tqdm
from src.utils import compute_metrics

for dataset_name in ["R52", "ohsumed"]:
    data, A = get_homogenous_data(dataset_name)
    data.x = (data.x - torch.mean(data.x, dim=1).unsqueeze(1)) / torch.std(
        data.x, dim=1
    ).unsqueeze(1)
    model = GCN(data.num_features, data.n_class, n_hidden=200)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-2)
    best_test_acc = pipeline(model, optimizer, data, max_epochs=200)
    print(f"Best test acc for {dataset_name}: {best_test_acc:.3f}")
# %%
