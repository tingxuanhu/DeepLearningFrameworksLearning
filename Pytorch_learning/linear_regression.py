import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# ----------- prepare data---------------
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X, y = torch.from_numpy(X_numpy.astype(np.float32)), torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

#  model
input_size = n_features
output_size = 1

model = nn.Linear(in_features=input_size, out_features=output_size)

# loss and optimizer
lr = .01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# training loop
epochs = 100
for epoch in range(epochs):
    # forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"epoch : {epoch + 1}, loss = {loss.item():.4f}")


predicted = model(X).detach()
plt.plot(X_numpy, y_numpy, 'ro')   # --> red dots
plt.plot(X_numpy, predicted, 'b')



