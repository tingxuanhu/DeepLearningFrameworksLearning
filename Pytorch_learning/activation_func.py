import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-1.0, 1.0, 2.0, 3.0])

# softmax  --  sigmoid    --tanh   --- relu
out = torch.softmax(x, dim=0)
print(out)
sm = nn.Softmax(dim=0)
out = sm(x)
print(out)


# nn.ReLU() creates an nn.Module which you can add e.g. to an nn.Sequential model.
# torch.relu on the other side is just the functional API call to the relu function,
# so that you can add it e.g. in your forward method yourself.

# option 1 (create nn modules)
class NN1(nn.Module):
    def __init__(self, in_, out_):
        super(NN1, self).__init__()
        self.linear1 = nn.Linear(in_, out_)
        self.relu = nn.ReLU(True)
        self.linear2 = nn.Linear(out_, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out__ = self.linear2(self.relu(self.linear1(x)))
        out__ = self.sigmoid(out__)
        return out__


# option 2 (use activation functions directly in forward pass)
class NN2(nn.Module):
    def __init__(self, in_, out_):
        super(NN2, self).__init__()
        self.linear1 = nn.Linear(in_, out_)
        self.linear2 = nn.Linear(out_, 1)

    def forward(self, x):
        out_ = torch.relu(self.linear1(x))
        # out_ = F.relu(self.linear1(x))
        # out_ = F.leaky_relu(self.linear1(x))
        out_ = F.leaky_relu_(self.linear1(x))  # some special API  , e.g.   F.leaky_relu()   is not available at torch
        out_ = torch.sigmoid(self.linear2(out_))
        return out_







