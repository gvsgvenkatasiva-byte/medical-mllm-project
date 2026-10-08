import torch
import torch.nn as nn
from torchvision import models


# Use CPU because our laptop does not have CUDA
device = torch.device("cpu")

# Load ResNet18
model = models.resnet18(weights="DEFAULT")

# Change the final layer for our 2 classes
num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 2)

# Move model to CPU
model = model.to(device)

# Print model information
print("Model: ResNet18")
print("Device:", device)
print("Number of classes:", 2)
print("Classes: NORMAL, PNEUMONIA")

# Test the model with a dummy image
dummy_input = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)