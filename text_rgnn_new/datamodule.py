from dataclasses import dataclass

import torch
import torch_geometric.transforms as T
from finetuning_encoders.datamodule import RequiredDatasetFormat
from finetuning_encoders.utils import generate_required_dataset_format, get_raw_data
from torch_geometric.data import HeteroData

from text_rgnn_new.embedding_generators.knn import KNNTrainer
from text_rgnn_new.embedding_generators.loader import EmbeddingLoader


@dataclass
class GraphDatasetConfig:
    dataset_name: str
    train_percentage: float
    doc_doc_k: int
    word_word_k: int
    use_w2w_graph: bool = False  #! Word 2 Word Graph sometimes not works


class GraphDataset:
    def __init__(self, c: GraphDatasetConfig):
        self.c = c
        text_data = generate_required_dataset_format(c.dataset_name, train_mask=None)
        self.emb_loader = EmbeddingLoader(c.dataset_name, c.train_percentage)

        d, documents = get_raw_data(c.dataset_name)
        # text_data.train_mask = self.emb_loader.load_train_mask()
        text_data.train_mask = d.train_mask
        self.generate_hetero(text_data)
        self.transform()

    def generate_hetero(self, text_data: RequiredDatasetFormat):
        self.data = HeteroData()
        self.data["word"].x = self.emb_loader.load_word_embeddings()
        self.data["doc"].x = self.emb_loader.load_doc_embeddings().to("cpu")

        self.data["doc"].y = text_data.labels
        self.data["doc"].train_mask = text_data.train_mask
        self.data["doc"].test_mask = text_data.test_mask

        # Doc-Doc Graph
        knn = KNNTrainer(self.data["doc"].x, self.c.doc_doc_k)
        e_i, e_a = knn.fit()
        self.data["doc", "d2d", "doc"].edge_index = e_i
        self.data["doc", "d2d", "doc"].edge_attr = e_a

        # Doc-Word Graph
        e_i, e_a = self.emb_loader.load_tfidf_graph()
        self.data["doc", "d2w", "word"].edge_index = e_i
        self.data["doc", "d2w", "word"].edge_attr = e_a

        # Word-Word Graph
        if self.c.use_w2w_graph:
            knn = KNNTrainer(self.data["word"].x, self.c.word_word_k)
            e_i, e_a = knn.fit()
            self.data["word", "w2w", "word"].edge_index = e_i
            self.data["word", "w2w", "word"].edge_attr = e_a

    def transform(self):
        self.data = T.ToUndirected()(self.data)
        self.data = T.AddSelfLoops()(self.data)


# class HomogeneousGraph:
#     def __init__(self, dataset_name: str, train_ratio):
#         self.text_data = generate_required_dataset_format(dataset_name, train_mask=None)

#         self.word_x = load_word_embeddings(dataset_name)
#         self.doc_y = self.text_data.labels
#         self.doc_x = load_doc_embeddings(dataset_name).to("cpu")

#         self.train_mask = ...  # load from fine-tuning encoder outputs

#         self.test_mask = self.text_data.test_mask
#         self.N, _, _ = generate_knn_graph(self.doc_x, k=10)
#         self.V, _, _ = generate_knn_graph(self.word_x, k=10)
#         self.K, _, _ = generate_tfidf_graph(dataset_name)

#         self.N = torch.tensor(self.N.toarray())
#         self.V = torch.tensor(self.V.toarray())
#         self.K = torch.tensor(self.K.toarray())

#         n = self.N.shape[0]
#         m = self.V.shape[0]
#         A = torch.zeros((n + m, n + m))
#         A[:n, :n] = self.N
#         A[n : n + m, n : n + m] = self.V
#         A[n : n + m, :n] = self.K.T
#         A[:n, n : n + m] = self.K

#         train_mask = torch.zeros(n + m, dtype=self.train_mask.dtype)
#         test_mask = torch.zeros(n + m, dtype=self.test_mask.dtype)

#         train_mask[:n] = self.train_mask
#         test_mask[:n] = self.test_mask
#         padding_size = self.doc_x.shape[1] - self.word_x.shape[1]

#         if padding_size > 0:
#             padding = torch.zeros(m, padding_size)
#             B_padded = torch.cat((self.word_x, padding), dim=1)
#         else:
#             B_padded = self.word_x

#         X = torch.cat((self.doc_x, B_padded), dim=0)
#         y = torch.cat((self.doc_y, torch.zeros((m, 1), dtype=self.doc_y.dtype)), dim=0)
#         row_idx, col_idx = A.nonzero(as_tuple=True)
#         edge_index = torch.stack((row_idx, col_idx), dim=0)
#         edge_attr = A[row_idx, col_idx]
#         # edge_index = torch.tensor(edge_index, dtype=torch.long)
#         # edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

#         self.data = Data(x=X, y=y, edge_index=edge_index, edge_attr=edge_attr)
#         self.data.train_mask = train_mask
#         self.data.test_mask = test_mask

#         self.data.x = (
#             self.data.x - torch.mean(self.data.x, dim=1).unsqueeze(1)
#         ) / torch.std(self.data.x, dim=1).unsqueeze(1)

#         self.data.num_features = self.data.x.shape[1]
#         self.data.n_class = self.text_data.n_class


def random_mask_selection(train_mask: torch.Tensor, train_ratio: float) -> torch.Tensor:
    """
    Randomly selects a subset of the training mask tensor.
    Parameters:
        None
    Returns:
        torch.Tensor: Mask tensor representing the randomly selected subset of the training data.
    """
    nonzero_idx = torch.nonzero(train_mask).squeeze()
    n_train = int(train_ratio * len(nonzero_idx))

    if train_ratio == 1.0:
        return train_mask
    else:
        shuffled_idxs = nonzero_idx[torch.randperm(len(nonzero_idx))]
        new_mask_tensor = torch.zeros_like(train_mask)
        new_mask_tensor[shuffled_idxs[:n_train]] = 1
        return new_mask_tensor
