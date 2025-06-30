import torch
x = torch.ones(size=(5,10))
y = torch.ones(size=(3,1,10))
print((x- y).shape)