import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


model = Model(n_input_features=6)

# train the model ......

# ------------ save and load ------------
PATH = "model.pth"
torch.save(model.state_dict(), PATH)

loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(PATH))  # it takes the loaded dir, not the path file itself.
loaded_model.eval()

# ------------ checkpoint (CPU for example) ------------
lr = .01
optimizer = torch.optim.SGD(loaded_model.parameters(), lr=lr)

checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict(),
}

FILE = "checkpoint.pth"
torch.save(checkpoint, FILE)

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=.01)

checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optim_state"])
epoch = checkpoint["epoch"]

# Remember that you must call model.eval()
# to set dropout and batch normalization layers to evaluation mode before running inference.
# Failing to do this will yield inconsistent inference ending_results.
# If you wish to resuming training, call model.train() to ensure these layers are in training mode.
model.eval()
# -- or --
model.train()  # train from the checkpoint


# --------------- Save on GPU, load on CPU ------------------
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

device = torch.device("cpu")
model = Model(n_input_features=6)
# model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))


# --------------- Save on GPU, load on GPU ------------------
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

model = Model(n_input_features=6)
# model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)  # move the model to GPU


# --------------- Save on CPU, load on GPU ------------------
torch.save(model.state_dict(), PATH)

device = torch.device("cuda")
model = Model(n_input_features=6)
# model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # choose which GPU device to move the model to

model.to(device)

# This loads the model to a given GPU device.
# Next, be sure to call model.to(torch.device('cuda')) to convert the model’s parameter tensors to CUDA tensors













