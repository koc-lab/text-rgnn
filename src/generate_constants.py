from pathlib import Path

import numpy as np
import torch
from graph_dataset_utils import load_processed_dataset


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


if __name__ == "__main__":
    for dataset_name in ["mr", "ohsumed", "R8", "R52"]:
        processed_dataset, vocab, stoi, itos = load_processed_dataset(dataset_name)
        edge_index, edge_attr = get_tfidf(processed_dataset.text, vocab, stoi)
        TF_IDF_DIR = Path.cwd() / "tf-idf-graphs"
        TF_IDF_DIR.mkdir(parents=True, exist_ok=True)
        file_name = f"{dataset_name}tfidf_graph.pth"
        torch.save((edge_index, edge_attr), TF_IDF_DIR / file_name)
