#%%
import torch

x = torch.randn(3, requires_grad=True)  # --> create the computational graph
print(x)  # --> tensor([-1.5877, -0.2473,  1.5587], requires_grad=True)
y = x + 2
print(y)  # --> tensor([0.4123, 1.7527, 3.5587], grad_fn=<AddBackward0>)

z = y*y*2
print(z)  # --> tensor([ 0.3399,  6.1442, 25.3283], grad_fn=<MulBackward0>)
z = z.mean()
print(z)  # --> tensor(10.6041, grad_fn=<MeanBackward0>)

z.backward()  # dz/dx
print(x.grad)   # where the gradients are stored


z1 = y*y*2
# z1.backward()  # Error --> grad can be implicitly created only for scalar outputs

# if z is not a scalar value, then we must give it the jacobian vector to backward
z1.backward(torch.tensor([.1, .0, .001], dtype=torch.float32))

#%%  How to prevent x from tracking gradients
import torch

x = torch.randn(4, requires_grad=True)
print(x)

x.requires_grad_(False)  # _ --> in place operation
print(x)

x = torch.randn(4, requires_grad=True)
y = x.detach()
print(y)

x = torch.randn(4, requires_grad=True)
with torch.no_grad():
    y = x + 2
    print(x)
    print(y)

#%%
import torch

weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
    model_output = (weights*3).sum()
    print(model_output)

    model_output.backward()
    print(weights.grad)

    weights.grad.zero_()  # clear the accumulated gradient





















