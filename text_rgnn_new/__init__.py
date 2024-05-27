from pathlib import Path

from text_rgnn_new.embedding_generators.tf_idf import TFIDFTrainer
from text_rgnn_new.embedding_generators.w2vec import Word2VecTrainer

#! if you have problems with project path, manually set it
# PROJECT_PATH = Path("/Users/ardaaras/Documents/text-rgcn")

PROJECT_PATH = Path.cwd()
ORIGINAL_DATA_PATH = PROJECT_PATH.joinpath("data/original-data")
GLUE_DATA_PATH = PROJECT_PATH.joinpath("data/glue-data")
W2V_MODELS_PATH = PROJECT_PATH.joinpath("data/w2v-models")
TF_IDF_GRAPHS_PATH = PROJECT_PATH.joinpath("data/tf-idf-graphs")

FINE_TUNED_ENCODERS_PATH = Path(
    "/Users/ardaaras/Documents/finetuning-encoders/best_models"
)

## Generate w2v and tfidf embeddings

for dataset_name in ["mr", "R8", "R52", "ohsumed", "sst2", "cola"]:
    w2v_trainer = Word2VecTrainer(dataset_name, W2V_MODELS_PATH)
    w2v_trainer.pipeline()

    vocab_path = W2V_MODELS_PATH.joinpath(f"{dataset_name}", "vocab.pkl")
    tfidf_trainer = TFIDFTrainer(dataset_name, TF_IDF_GRAPHS_PATH, vocab_path)
    tfidf_trainer.pipeline()


DATASET_TO_N_CLASS = {"mr": 2, "R8": 8, "R52": 52, "ohsumed": 23, "sst2": 2, "cola": 2}
