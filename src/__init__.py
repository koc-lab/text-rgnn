import pickle
from pathlib import Path

import datasets

#! if you have problems with project path, manually set it
# PROJECT_PATH = Path("/Users/ardaaras/Documents/text-rgcn")

PROJECT_PATH = Path.cwd()
ORIGINAL_DATA_PATH = PROJECT_PATH.joinpath("data/original-data")
W2V_MODELS_PATH = PROJECT_PATH.joinpath("data/w2v-models")
TF_IDF_GRAPHS_PATH = PROJECT_PATH.joinpath("data/tf-idf-graphs")
GLUE_DATA_PATH = PROJECT_PATH.joinpath("data/glue-data")

W2V_MODELS_PATH.mkdir(parents=True, exist_ok=True)
TF_IDF_GRAPHS_PATH.mkdir(parents=True, exist_ok=True)
GLUE_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Load the GLUE data and save it as pickle
for dataset in ["cola", "sst2"]:
    data = datasets.load_dataset("glue", dataset)
    pickle.dump(data, open(GLUE_DATA_PATH.joinpath(f"{dataset}_raw_data.pkl"), "wb"))


LABEL_TO_INT_MAP = {
    "ohsumed": {
        "C21": 0,
        "C01": 1,
        "C09": 2,
        "C15": 3,
        "C18": 4,
        "C19": 5,
        "C02": 6,
        "C05": 7,
        "C13": 8,
        "C08": 9,
        "C23": 10,
        "C12": 11,
        "C06": 12,
        "C10": 13,
        "C11": 14,
        "C03": 15,
        "C22": 16,
        "C16": 17,
        "C20": 18,
        "C07": 19,
        "C04": 20,
        "C17": 21,
        "C14": 22,
    },
    "R52": {
        "copper": 0,
        "lei": 1,
        "rubber": 2,
        "strategic-metal": 3,
        "lumber": 4,
        "jobs": 5,
        "wpi": 6,
        "jet": 7,
        "housing": 8,
        "livestock": 9,
        "dlr": 10,
        "heat": 11,
        "sugar": 12,
        "bop": 13,
        "cpu": 14,
        "ship": 15,
        "money-fx": 16,
        "instal-debt": 17,
        "nickel": 18,
        "retail": 19,
        "interest": 20,
        "potato": 21,
        "crude": 22,
        "gnp": 23,
        "trade": 24,
        "money-supply": 25,
        "earn": 26,
        "orange": 27,
        "fuel": 28,
        "tea": 29,
        "lead": 30,
        "gold": 31,
        "veg-oil": 32,
        "tin": 33,
        "cpi": 34,
        "gas": 35,
        "nat-gas": 36,
        "ipi": 37,
        "iron-steel": 38,
        "carcass": 39,
        "pet-chem": 40,
        "cocoa": 41,
        "income": 42,
        "platinum": 43,
        "cotton": 44,
        "grain": 45,
        "coffee": 46,
        "reserves": 47,
        "meal-feed": 48,
        "alum": 49,
        "zinc": 50,
        "acq": 51,
    },
    "R8": {
        "earn": 0,
        "interest": 1,
        "crude": 2,
        "ship": 3,
        "money-fx": 4,
        "acq": 5,
        "grain": 6,
        "trade": 7,
    },
    "mr": {"0": 0, "1": 1},
    "cola": {0: 0, 1: 1},
    "sst2": {0: 0, 1: 1},
}
