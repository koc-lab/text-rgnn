# %%
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import to_hetero
import torch.nn.functional as F

from tqdm.auto import tqdm
from src.hetero_generator import get_hetero_data, load_variables
from src.models import GNN, GAT
from src.utils import compute_metrics, set_seeds

dataset_name = "mr"
processed_dataset, vocab, stoi, itos = load_variables(dataset_name)
data = get_hetero_data(dataset_name, processed_dataset, doc_doc_k=20)


data = T.ToUndirected()(data)
data = T.AddSelfLoops()(data)
#! feature normalize edince bok gibi oluyor neden bak
# data = T.NormalizeFeatures()(data)
data_doc = data["doc"]

seed_no = 43


def train(model, optimizer, data_doc, max_epochs=100):
    set_seeds(seed_no)
    t = tqdm(range(max_epochs))
    best_test_acc = 0
    for epoch in t:
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)["doc"]
        out = F.log_softmax(out, dim=1)

        train_mask = data_doc.train_mask
        test_mask = data_doc.test_mask
        y = data_doc.y.reshape(-1)

        loss = F.nll_loss(out[train_mask == 1], y[train_mask == 1])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _, _, _, train_acc = compute_metrics(
                out[train_mask == 1], y[train_mask == 1]
            )
            _, _, _, test_acc = compute_metrics(out[test_mask == 1], y[test_mask == 1])

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        t.set_description(
            f"Loss: {loss:.4f}, Best Test Acc: {100* best_test_acc:.3f} , Train Acc: {100*train_acc:.3f}"
        )


set_seeds(seed_no)
print("**************GAT RESULTS***************")
model = GAT(hidden_channels=64, out_channels=processed_dataset.n_class)
model = to_hetero(model, data.metadata(), aggr="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

train(model, optimizer, data_doc, max_epochs=50)
print("**************GNN RESULTS***************")
model = GNN(hidden_channels=64, out_channels=processed_dataset.n_class)
model = to_hetero(model, data.metadata(), aggr="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

train(model, optimizer, data_doc, max_epochs=100)


# %%
