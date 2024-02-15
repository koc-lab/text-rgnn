import torch
import torch.nn as nn
import torch.optim as optim

# from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from me_gcn_src.model import MULTIGCN


def cal_accuracy(predictions, labels):
    pred = torch.argmax(predictions, -1).cpu().tolist()
    lab = labels.cpu().tolist()
    cor = 0
    for i in range(len(pred)):
        if pred[i] == lab[i]:
            cor += 1
    return cor / len(pred)


class TrainerMEGCN:
    def __init__(
        self,
        features,
        adj_list,
        idx_train,
        idx_val,
        idx_test,
        labels,
        test_size,
        n_class,
        device,
    ):
        self.model = MULTIGCN(
            nfeat=features.shape[1], nhid=25, nclass=n_class, dropout=0.5
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.002, weight_decay=0)
        self.criterion = nn.CrossEntropyLoss()

        self.features = features
        self.adj_list = adj_list

        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

        self.labels = labels
        self.test_size = test_size
        self.device = device

    def pipeline(self, max_epochs):
        t = tqdm(range(max_epochs))
        for _ in t:

            val_loss = []
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.features, self.adj_list)
            loss_train = self.criterion(
                output[self.idx_train], self.labels[self.idx_train]
            )
            acc_train = cal_accuracy(
                output[self.idx_train], self.labels[self.idx_train]
            )

            loss_train.backward()
            self.optimizer.step()

            self.model.eval()
            output = self.model(self.features, self.adj_list)
            loss_val = self.criterion(output[self.idx_val], self.labels[self.idx_val])
            val_loss.append(loss_val.item())
            acc_val = cal_accuracy(output[self.idx_val], self.labels[self.idx_val])

            t.set_description(
                f"Train Loss: {loss_train.item():.4f}, Train Acc: {100*acc_train:.3f}, Val Acc: {100*acc_val:.3f}"
            )

        print("Training Done! Evaluating on Test Set...")
        self.model.eval()
        output = self.model(self.features, self.adj_list)
        loss_test = self.criterion(
            output[self.idx_test], self.labels[-self.test_size :]
        )
        acc_test = cal_accuracy(output[self.idx_test], self.labels[-self.test_size :])
        print(
            "Test set results:",
            f"loss= {loss_test.item():.4f}",
            f"accuracy= {acc_test:.4f}",
        )
        return acc_test, output
