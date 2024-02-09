# %%
import torch
import numpy as np
import torch.nn as nn

# %%
a = torch.tensor([[2, 3, 5], [1, 2, 9]])
b = np.array([[2, 3, 5], [1, 2, 9]])
print(a, b)
print(a.shape)

# %%
a = torch.rand(2, 2)
b = np.random.rand(2, 2)
print(a, b)

# %%
torch.manual_seed(1234)
a = torch.rand(2, 2)
b = torch.rand(2, 2)
print(a, b)

# %%
a = torch.tensor([[2, 3], [1, 2]])
b = torch.tensor([[2, 3], [1, 2]])

# %%
a * b  # element-wise multplication
# %%
torch.mul(a, b)  # element-wise multiplication
# %%
torch.matmul(a, b)  # matrix multiplication. https://pytorch.org/docs/stable/generated/torch.matmul.html
# %%
torch.zeros(2, 2)
# %%
torch.ones(2, 2)
# %%
torch.eye(2)
# %%
tensor = torch.ones(2)
print(f"Type of tensor: {type(tensor)}, tensor: {tensor}")
tensor = tensor.numpy()
print(f"Type of tensor: {type(tensor)}, tensor: {tensor}")
tensor = torch.from_numpy(tensor)
print(f"Type of tensor: {type(tensor)}, tensor: {tensor}")
# %%
print(torch.cuda.is_available())
tensor = tensor.to("cuda:0")
print(tensor.device)
tensor = tensor.to("cpu")
print(tensor.device)



# %%
# defining a neural network
# In PyTorch, a model is represented by a regular Python class that inherits from the Module class.
# __init__ defines the parts that make up the model
# forward() performs the actual computation
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 20)
        self.output = nn.Linear(20, 4)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.output(x)
        return x

input = torch.rand(10)
net = Net()
output = net(input)

# %%
