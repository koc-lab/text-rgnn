import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_sent_label(processed_dataset, train_size):
    docs = processed_dataset.text
    labels = processed_dataset.label_original

    train_docs, test_docs, train_labels, test_labels = train_test_split(
        docs, labels, stratify=labels, train_size=train_size, random_state=0
    )

    docs = train_docs + test_docs
    labels = train_labels + test_labels
    return train_docs, test_docs, train_labels, test_labels, docs, labels


def print_info(train_docs, test_docs, labels):
    print("Train size: ", len(train_docs))
    print("Test size: ", len(test_docs))
    print("Original size: ", len(train_docs) + len(test_docs))
    print("Number of classes: ", len(np.unique(labels)))


def encode_labels(train_labels, test_labels, device):
    unique_labels = np.unique(train_labels + test_labels)
    lEnc = LabelEncoder()
    lEnc.fit(unique_labels)

    train_labels = lEnc.transform(train_labels)
    test_labels = lEnc.transform(test_labels)
    labels = train_labels.tolist() + test_labels.tolist()
    labels = torch.LongTensor(labels).to(device)
    return labels
