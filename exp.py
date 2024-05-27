# %%
import torch
from finetuning_encoders.utils import set_seeds
from torch_geometric.nn import to_hetero

from text_rgnn_new import DATASET_TO_N_CLASS
from text_rgnn_new.datamodule import GraphDataset, GraphDatasetConfig
from text_rgnn_new.models import GraphSAGE
from text_rgnn_new.trainer import Trainer

set_seeds(42)
dataset_name = "mr"
n_class = DATASET_TO_N_CLASS[dataset_name]
config = GraphDatasetConfig(
    dataset_name=dataset_name,
    train_percentage=20,
    doc_doc_k=30,
    word_word_k=10,
    use_w2w_graph=False,
)
data = GraphDataset(config)

model = GraphSAGE(64, n_class)
model = to_hetero(model, data.data.metadata(), aggr="sum")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
trainer = Trainer(model, optimizer, data)
trainer.pipeline(max_epochs=100, patience=30)
# %%

# # Trial
# train_percentage = 20
# cls_embds = data.data["doc"].x
# best_model = torch.load(
#     "/Users/ardaaras/Documents/finetuning-encoders/best_models/roberta-base-mr-20/best_model_87.704_64.pth"
# )

# clf = best_model.classifier
# out = cls_embds @ clf.weight.T.to("cpu")

# from text_rgnn_new.utils import compute_metrics

# test_mask = data.data["doc"].test_mask
# y = data.data["doc"].y.reshape(-1)
# compute_metrics(y_pred=out[test_mask == 1], y=y[test_mask == 1])
