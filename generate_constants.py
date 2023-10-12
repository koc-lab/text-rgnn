import torch
from pathlib import Path
from hetero_generator import load_variables, get_tfidf

if __name__ == "__main__":
    for dataset_name in ["mr", "ohsumed", "R8", "R52"]:
        processed_dataset, vocab, stoi, itos = load_variables(dataset_name)
        edge_index, edge_attr = get_tfidf(processed_dataset.text, vocab, stoi)
        TF_IDF_DIR = Path.cwd() / "tf-idf-graphs"
        TF_IDF_DIR.mkdir(parents=True, exist_ok=True)
        file_name = f"{dataset_name}tfidf_graph.pth"
        torch.save((edge_index, edge_attr), TF_IDF_DIR / file_name)
