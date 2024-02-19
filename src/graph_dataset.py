from dataclasses import dataclass
from pathlib import Path

import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from src.graph_generators import generate_knn_graph
from src.text_dataset import TextDataset
from src.utils import load_tfidf_graph, load_vocab, load_word_embeddings


@dataclass
class GraphDatasetConfig:
    dataset_name: str
    doc_doc_k: int
    word_word_k: int
    train_ratio: float  #! Train ratio for train_mask
    use_w2w_graph: bool = False  #! Word 2 Word Graph sometimes not works


class GraphDataset:
    def __init__(self, c: GraphDatasetConfig):
        self.c = c
        self.text_dataset = TextDataset(c.dataset_name)
        self.vocab = load_vocab(c.dataset_name)

        #! self.train_mask includes the certain percentage of the original train mask
        #! to access original train mask use self.text_dataset.train_mask
        self.train_mask = self.random_mask_selection()

        self.generate_hetero()
        self.transform()

    def generate_hetero(self):
        self.data = HeteroData()
        self.data["word"].x = load_word_embeddings(self.c.dataset_name)
        self.data["doc"].x = load_doc_embeddings(self.c.dataset_name).to("cpu")

        self.data["doc"].y = self.text_dataset.labels
        self.data["doc"].train_mask = self.train_mask
        self.data["doc"].test_mask = self.text_dataset.test_mask

        # Doc-Doc Graph
        e_i, e_a = generate_knn_graph(self.data["doc"].x, k=self.c.doc_doc_k)
        self.data["doc", "d2d", "doc"].edge_index = e_i
        self.data["doc", "d2d", "doc"].edge_attr = e_a

        # Doc-Word Graph
        e_i, e_a = load_tfidf_graph(self.c.dataset_name)
        self.data["doc", "d2w", "word"].edge_index = e_i
        self.data["doc", "d2w", "word"].edge_attr = e_a

        # Word-Word Graph
        if self.c.use_w2w_graph:
            e_i, e_a = generate_knn_graph(self.data["word"].x, k=self.c.word_word_k)
            self.data["word", "w2w", "word"].edge_index = e_i
            self.data["word", "w2w", "word"].edge_attr = e_a

    def transform(self):
        self.data = T.ToUndirected()(self.data)
        self.data = T.AddSelfLoops()(self.data)

    def random_mask_selection(self) -> torch.Tensor:
        """
        Randomly selects a subset of the training mask tensor.
        Parameters:
            None
        Returns:
            torch.Tensor: Mask tensor representing the randomly selected subset of the training data.
        """
        nonzero_idx = torch.nonzero(self.text_dataset.train_mask).squeeze()
        n_train = int(self.c.train_ratio * len(nonzero_idx))

        if self.c.train_ratio == 1.0:
            return self.text_dataset.train_mask
        else:
            shuffled_idxs = nonzero_idx[torch.randperm(len(nonzero_idx))]
            new_mask_tensor = torch.zeros_like(self.text_dataset.train_mask)
            new_mask_tensor[shuffled_idxs[:n_train]] = 1
            return new_mask_tensor


def load_doc_embeddings(dataset_name: str):
    if dataset_name == "sst2" or "cola":
        data = TextDataset(dataset_name)
        size = len(data.documents)
        #! need a fine-tuned embeddings for SST2 and COLA
        return torch.rand(size, 768)
    else:
        EMBEDDINGS_PATH = Path.cwd().parent.joinpath(
            "finetune-text-graphs/generator-output"
        )
        file_name = f"{dataset_name}_embeddings.pth"
        embeddings = torch.load(EMBEDDINGS_PATH.joinpath(file_name))
    return embeddings


# def sanity_check_for_embedding_matrix(
#     wv: KeyedVectors,
#     similarity_matrix: torch.tensor,
#     stoi: dict,
#     w1: str = "bad",
#     w2: str = "good",
# ):
#     w1_vec = torch.tensor(wv[w1], dtype=torch.float).reshape(1, -1)
#     w2_vec = torch.tensor(wv[w2], dtype=torch.float).reshape(1, -1)

#     print(f"Word2Vec Simlarity:{wv.similarity(w1, w2)}")
#     print(f"Cosine Similarity: {F.cosine_similarity(w1_vec, w2_vec)}")
#     print(f"Similartiy matrix entry: {similarity_matrix[stoi[w1], stoi[w2]]}")


#     stoi = {word: i for i, word in enumerate(vocab)}
#     itos = {i: word for i, word in enumerate(vocab)}
