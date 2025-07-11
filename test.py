import torch
import matplotlib.pyplot as plt

x = torch.randn(size=(20000, 5), device='cuda')  # Input: 100 points in 1024D

# Step 1: Compute squared L2 norms of each point
x_norm = (x ** 2).sum(dim=1)  # Shape: [100]

# Step 2: Compute pairwise dot products
dot_product = torch.mm(x, x.T)  # Shape: [100, 100]

# Step 3: Use identity ||a - b||² = ||a||² + ||b||² - 2<a, b>
pairwise_distances = x_norm.unsqueeze(1) + x_norm.unsqueeze(0) - 2 * dot_product  # Shape: [100, 100]

# Ensure numerical stability (avoid negative distances due to floating point errors)
pairwise_distances = torch.clamp(pairwise_distances, min=0.0)

# Optionally, take sqrt for true L2 distance
pairwise_l2 = torch.sqrt(pairwise_distances).view(-1)
pairwise_l2 = pairwise_l2[pairwise_l2 > 1e-3]
pairwise_l2 = (pairwise_l2 - torch.min(pairwise_l2)) / (torch.max(pairwise_l2) - torch.min(pairwise_l2))

plt.hist(pairwise_l2.cpu().detach().numpy(), bins=100)
plt.show()