import torch
import torch.nn as nn
import numpy as np

# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x), axis=0)
#
#
# x = np.array([2.0, 1.0, 0.1])
# out = softmax(x)
# print("Softmax numpy:", out)
#
# x = torch.tensor([2.0, 1.0, 0.1])
# out = torch.softmax(x, dim=0)
# print(out)
#
#
# def cross_entropy(real, pred):
#     return -np.sum(real * np.log(pred))
#
#
# y = np.array([1, 0, 0])
# y_pred_good = np.array([.7, .2, .1])
# y_pred_bad = np.array([.1, .3, .6])
# l1 = cross_entropy(y, y_pred_good)
# l2 = cross_entropy(y, y_pred_bad)
#
# print(f"Loss1_numpy: {l1:.4f}")
# print(f"Loss2_numpy: {l2:.4f}")


# ---------- pytorch -------------------

# 3 samples
y = torch.tensor([2, 0, 1])

# n_samples x n_cls = 1 x 3
y_pred_good = torch.tensor([[.1, 1.0, 2.1], [2.1, 1.0, .1], [.1, 2.0, .8]])  # torch.Size([1, 3])
y_pred_bad = torch.tensor([[.5, 2.0, .3], [.1, 1.0, 2.1], [.1, 1.0, 2.1]])

print(y_pred_good)

criterion = nn.CrossEntropyLoss()  # (applies Softmax)

l1 = criterion(y_pred_good, y)
l2 = criterion(y_pred_bad, y)

print(l1)
# print(l1.item())
# print(l2.item())

# get actual prediction
values, prediction1 = torch.max(y_pred_good, dim=1)
_, prediction2 = torch.max(y_pred_bad, dim=1)

print(values, prediction1)
print(prediction2)


# ----binary classification
class BNN(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(BNN, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out, True)
        out = self.linear2(out)
        y_pred = torch.sigmoid(out)
        return y_pred


model = BNN(in_dim=28 * 28, hid_dim=5)
criterion = nn.BCELoss()


# -- Multiclass classification
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end
        return out


model2 = NeuralNet2(input_size=28 * 28, hidden_size=5, num_classes=3)
criterion2 = nn.CrossEntropyLoss()  # (applies Softmax)



