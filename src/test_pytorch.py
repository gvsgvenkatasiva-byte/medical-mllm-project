import torch

print("PyTorch is working!")

# Create two tensors
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Add them
result = a + b

print("A:", a)
print("B:", b)
print("A + B:", result)

# Check device
print("Device:", result.device)