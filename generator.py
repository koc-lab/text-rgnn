# %%
import pickle

import datasets

from src import TF_IDF_GRAPHS_PATH, W2V_MODELS_PATH
from src.graph_generators import tfidf_pipeline
from src.w2vec import w2v_pipeline

# Generate the required folders
#! There is an assumption that you downloaded the original-data and placed it in data/original-data

# Create the required folders
W2V_MODELS_PATH.mkdir(parents=True, exist_ok=True)
TF_IDF_GRAPHS_PATH.mkdir(parents=True, exist_ok=True)


# Load the raw data and save it as pickle
raw_data = datasets.load_dataset("glue", "cola")
pickle.dump(raw_data, open("cola_raw_data.pkl", "wb"))
raw_data = datasets.load_dataset("glue", "sst2")
pickle.dump(raw_data, open("sst2_raw_data.pkl", "wb"))

for dataset_name in ["mr", "R8", "R52", "ohsumed", "sst2", "cola"]:
    # Train word2vec models for all datasets both for embedding and vocab
    print(f"Training word2vec model for {dataset_name}...")
    w2v_pipeline(dataset_name)
    # Generate TF-IDF graphs for all datasets, vocab from word2vec models used here
    print(f"Generating tf-idf graph for {dataset_name}...")
    tfidf_pipeline(dataset_name)
