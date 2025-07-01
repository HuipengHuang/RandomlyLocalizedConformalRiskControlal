import torch
x = torch.ones(size=(32,1050))
U, S, Vh = torch.linalg.svd(x, full_matrices=False)
print(U.shape, S.shape, Vh.shape)