# %%
import numpy as np
import pandas as pd
import torch

from me_gcn_src.clean import tokenize_and_clean
from me_gcn_src.embeddings import generate_doc_embeddings, generate_word_embeddings
from me_gcn_src.engine import TrainerMEGCN
from me_gcn_src.generate_adj import get_adj_list
from me_gcn_src.utils import encode_labels, get_sent_label, print_info
from src.graph_dataset_utils import load_processed_dataset

results_dict = {"Dataset": [], "Train Portion": [], "Test Accuracy": []}
for train_portion in [0.01, 0.05, 0.1, 0.2, 1]:
    for dataset_name in ["R52", "R8", "ohsumed", "mr"]:
        processed_dataset, vocab, stoi, itos = load_processed_dataset(dataset_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_docs, test_docs, train_labels, test_labels, docs, labels = get_sent_label(
            processed_dataset, train_portion
        )
        train_size, test_size = len(train_docs), len(test_docs)
        labels = encode_labels(train_labels, test_labels, device)
        print_info(train_docs, test_docs, labels)

        word_list, tokenized_docs, word_id_map = tokenize_and_clean(docs)
        vocab_length = len(word_list)
        word_emb_dict = generate_word_embeddings(word_list, tokenized_docs)
        doc2vec_npy = generate_doc_embeddings(tokenized_docs)

        node_size = train_size + vocab_length + test_size
        adj_list = get_adj_list(
            tokenized_docs,
            word_list,
            node_size,
            word_id_map,
            train_size,
            vocab_length,
            word_emb_dict,
            doc2vec_npy,
        )

        features = []
        for i in range(train_size):
            features.append(doc2vec_npy[i])
        for word in word_list:
            features.append(word_emb_dict[word])
        for i in range(test_size):
            features.append(doc2vec_npy[train_size + i])
        features = torch.FloatTensor(np.array(features))

        real_train_size = int((1 - 0.1) * train_size)
        val_size = train_size - real_train_size

        idx_train = range(real_train_size)
        idx_val = range(real_train_size, train_size)
        idx_test = range(train_size + vocab_length, node_size)

        trainer = TrainerMEGCN(
            features,
            adj_list,
            idx_train,
            idx_val,
            idx_test,
            labels,
            test_size,
            len(np.unique(labels)),
            device,
        )

        acc_test, y_hat_test = trainer.pipeline(20)
        results_dict["Dataset"].append(dataset_name)
        results_dict["Train Portion"].append(train_portion)
        results_dict["Test Accuracy"].append(acc_test)

# %%


df = pd.DataFrame(results_dict)
df.to_csv("me_gcn_results.csv", index=False)
df.to_excel("me_gcn_results.xlsx", index=False)
