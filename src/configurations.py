from dataclasses import dataclass


@dataclass
class GraphDatasetConfig:
    dataset_name: str
    doc_doc_k: int
    word_word_k: int
    use_w2w_graph: bool = False  # Word 2 Word Graph sometimes not works
