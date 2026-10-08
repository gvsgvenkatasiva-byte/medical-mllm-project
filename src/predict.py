import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# -----------------------------
# 1. Device
# -----------------------------

device = torch.device("cpu")


# -----------------------------
# 2. Image path
# -----------------------------

IMAGE_PATH = r"data\test.jpg"


# -----------------------------
# 3. Image preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# 4. Create ResNet18
# -----------------------------

model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 2)


# -----------------------------
# 5. Load trained model
# -----------------------------

model.load_state_dict(
    torch.load(
        "models/pneumonia_resnet18.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# -----------------------------
# 6. Load image
# -----------------------------

image = Image.open(IMAGE_PATH).convert("RGB")

print("Image loaded successfully!")


# -----------------------------
# 7. Preprocess image
# -----------------------------

input_tensor = transform(image)

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(device)


# -----------------------------
# 8. Make prediction
# -----------------------------

with torch.no_grad():

    output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)

    predicted_class = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][predicted_class].item()


# -----------------------------
# 9. Class names
# -----------------------------

classes = ["NORMAL", "PNEUMONIA"]

prediction = classes[predicted_class]


# -----------------------------
# 10. Display result
# -----------------------------

print("\n==============================")
print("X-RAY PREDICTION")
print("==============================")

print("Prediction:", prediction)
print(f"Confidence: {confidence * 100:.2f}%")

print("\nClass probabilities:")

print(
    f"NORMAL: {probabilities[0][0].item() * 100:.2f}%"
)

print(
    f"PNEUMONIA: {probabilities[0][1].item() * 100:.2f}%"
)

print("==============================")