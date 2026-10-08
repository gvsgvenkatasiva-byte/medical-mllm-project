import cv2
import torch

# Read image
image = cv2.imread("data/test.jpg")

if image is None:
    print("Image not found!")
else:
    print("Original shape:", image.shape)

    # Resize
    image = cv2.resize(image, (224, 224))

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert pixel values from 0-255 to 0-1
    image = image / 255.0

    # Convert NumPy array to PyTorch tensor
    tensor = torch.tensor(image, dtype=torch.float32)

    print("Tensor shape:", tensor.shape)
    print("Tensor data type:", tensor.dtype)

    # Change from:
    # Height, Width, Channels
    # to:
    # Channels, Height, Width

    tensor = tensor.permute(2, 0, 1)

    print("Final tensor shape:", tensor.shape)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    print("Model input shape:", tensor.shape)