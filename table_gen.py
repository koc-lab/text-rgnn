# %%
import pandas as pd
import torch
import torch.nn.functional as F
from finetuning_encoders.utils import set_seeds

from text_rgnn_new import PROJECT_PATH
from text_rgnn_new.datamodule import GraphDataset, GraphDatasetConfig
from text_rgnn_new.utils import compute_metrics

set_seeds(42)


def evaluate(model, y, data, mask):
    """Evaluates the model and computes the metrics."""
    with torch.no_grad():
        model.eval()
        out_full = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        out = out_full["doc"]
        out = F.log_softmax(out, dim=1)
        y_pred = out[mask == 1]
        return compute_metrics(y_pred, y)


def get_mask(data, data_key):
    """Gets the appropriate mask for the data split."""
    if data_key == "train":
        mask = data["doc"].train_mask
    elif data_key == "val":
        # get random 10% of the training data
        mask = data["doc"].train_mask
        return torch.randperm(mask.size(0)) < int(0.1 * mask.size(0))
    return mask


def load_model(path):
    """Loads the model from the specified path."""
    file_name = list(path.glob("*.pth"))[0]
    return torch.load(str(file_name)), file_name


def evaluate_dataset(dataset_name, train_percentage):
    """Evaluates the model for a given dataset and training percentage."""
    results = []
    config = GraphDatasetConfig(
        dataset_name=dataset_name,
        train_percentage=train_percentage,
        doc_doc_k=10,
        word_word_k=10,
        use_w2w_graph=True,
    )
    g_data = GraphDataset(config)
    path = PROJECT_PATH.joinpath("best_models", f"{dataset_name}-{train_percentage}")
    model, file_name = load_model(path)

    for data_key in ["train", "val", "test"]:
        metric_name = "mcc" if dataset_name == "cola" else "acc"

        if data_key != "test":
            mask = get_mask(g_data.data, data_key)
            y = g_data.data["doc"].y.reshape(-1)[mask == 1]
            data = g_data.data
            metrics = evaluate(model, y, data, mask)
        else:
            saved_result = file_name.stem.split("_")[-1]
            metrics = {metric_name: float(saved_result)}

        value = f"{metrics[metric_name]:.4f}"
        print(f"{dataset_name} {train_percentage} {data_key}_{metric_name}: {value}")
        results.append(
            {
                "dataset_name": dataset_name,
                "train_percentage": train_percentage,
                "data_key": data_key,
                "metric_name": metric_name,
                "value": float(value),
            }
        )

    return results


def main():
    dataset_names = [
        "cola",
        "mr",
        "ohsumed",
        "R8",
        "R52",
        "sst2",
    ]  # Add more dataset names as needed
    train_percentages = [1, 5, 10, 20]  # Add more training percentages as needed

    all_results = []
    for dataset_name in dataset_names:
        for train_percentage in train_percentages:
            results = evaluate_dataset(dataset_name, train_percentage)
            all_results.extend(results)

    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv("evaluation_results.csv", index=False)
    print("Results saved to evaluation_results.csv")


if __name__ == "__main__":
    main()
    table = pd.read_csv("evaluation_results.csv")
    table.to_excel("evaluation_results.xlsx", index=False)
# %%
