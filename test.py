import torch
x = torch.zeros(size=(5,10))
y = torch.rand(size=(10,))
print((x > y).shape)