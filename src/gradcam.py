import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image

import cv2
import numpy as np


# -----------------------------
# 1. Device
# -----------------------------

device = torch.device("cpu")

print("Using device:", device)


# -----------------------------
# 2. Image path
# -----------------------------

IMAGE_PATH = r"data\test.jpg"


# -----------------------------
# 3. Load improved model
# -----------------------------

model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 2)

model.load_state_dict(
    torch.load(
        "models/pneumonia_resnet18_improved.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Improved model loaded successfully!")


# -----------------------------
# 4. Preprocessing
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
# 5. Load image
# -----------------------------

image = Image.open(
    IMAGE_PATH
).convert("RGB")

input_tensor = transform(
    image
).unsqueeze(0)

input_tensor = input_tensor.to(device)

print("Image loaded successfully!")


# -----------------------------
# 6. Grad-CAM hooks
# -----------------------------

activations = []
gradients = []


def forward_hook(module, input, output):
    activations.append(output.detach())


def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0].detach())


target_layer = model.layer4[-1]

target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


# -----------------------------
# 7. Prediction
# -----------------------------

output = model(input_tensor)

probabilities = torch.softmax(
    output,
    dim=1
)

predicted_class = torch.argmax(
    probabilities,
    dim=1
).item()

confidence = (
    probabilities[0][predicted_class]
    .item()
)

classes = [
    "NORMAL",
    "PNEUMONIA"
]

prediction = classes[predicted_class]


print("\n==============================")
print("PREDICTION")
print("==============================")

print("Prediction:", prediction)

print(
    f"Confidence: {confidence * 100:.2f}%"
)


# -----------------------------
# 8. Backward pass
# -----------------------------

model.zero_grad()

score = output[
    0,
    predicted_class
]

score.backward()


# -----------------------------
# 9. Get activation and gradient
# -----------------------------

activation = activations[0][0]

gradient = gradients[0][0]


# -----------------------------
# 10. Calculate Grad-CAM
# -----------------------------

weights = gradient.mean(
    dim=(1, 2)
)

cam = torch.zeros(
    activation.shape[1:],
    dtype=torch.float32
)

for i in range(
    activation.shape[0]
):

    cam += (
        weights[i]
        * activation[i]
    )

cam = torch.relu(cam)

cam = cam.cpu().numpy()


# -----------------------------
# 11. Normalize CAM
# -----------------------------

cam = cv2.resize(
    cam,
    (224, 224)
)

cam = (
    cam - cam.min()
) / (
    cam.max() - cam.min() + 1e-8
)


# -----------------------------
# 12. Create heatmap
# -----------------------------

heatmap = np.uint8(
    255 * cam
)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)


# -----------------------------
# 13. Prepare original image
# -----------------------------

original = cv2.imread(
    IMAGE_PATH
)

original = cv2.resize(
    original,
    (224, 224)
)


# -----------------------------
# 14. Create overlay
# -----------------------------

overlay = cv2.addWeighted(
    original,
    0.6,
    heatmap,
    0.4,
    0
)


# -----------------------------
# 15. Add labels
# -----------------------------

original_labeled = original.copy()

overlay_labeled = overlay.copy()

cv2.putText(
    original_labeled,
    "Original X-Ray",
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

cv2.putText(
    overlay_labeled,
    "Grad-CAM",
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)


# -----------------------------
# 16. Side-by-side comparison
# -----------------------------

comparison = np.hstack(
    (
        original_labeled,
        overlay_labeled
    )
)


# -----------------------------
# 17. Save comparison
# -----------------------------

cv2.imwrite(
    "data/gradcam_comparison.jpg",
    comparison
)


print("\n==============================")
print("GRAD-CAM COMPLETED")
print("==============================")

print(
    "Saved: data/gradcam_comparison.jpg"
)