import torch
import torch.nn as nn

from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader


# -----------------------------
# 1. Device
# -----------------------------

device = torch.device("cpu")

print("Using device:", device)


# -----------------------------
# 2. Dataset path
# -----------------------------

DATA_PATH = r"C:\Users\HP\Downloads\archive\chest_xray"


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
# 4. Load test dataset
# -----------------------------

test_dataset = datasets.ImageFolder(
    DATA_PATH + r"\test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))
print("Classes:", test_dataset.classes)


# -----------------------------
# 5. Create ResNet18
# -----------------------------

model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 2)


# -----------------------------
# 6. Load trained model
# -----------------------------

model.load_state_dict(
    torch.load(
        "models/pneumonia_resnet18_improved.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Trained model loaded successfully!")


# -----------------------------
# 7. Evaluation
# -----------------------------

correct = 0
total = 0

# Confusion matrix values
true_normal = 0
false_normal = 0
true_pneumonia = 0
false_pneumonia = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        # ---------------------------------
        # NORMAL = class 0
        # PNEUMONIA = class 1
        # ---------------------------------

        for actual, prediction in zip(labels, predicted):

            actual = actual.item()
            prediction = prediction.item()

            if actual == 0 and prediction == 0:
                true_normal += 1

            elif actual == 0 and prediction == 1:
                false_pneumonia += 1

            elif actual == 1 and prediction == 1:
                true_pneumonia += 1

            elif actual == 1 and prediction == 0:
                false_normal += 1


# -----------------------------
# 8. Accuracy
# -----------------------------

accuracy = 100 * correct / total

normal_total = true_normal + false_pneumonia

pneumonia_total = true_pneumonia + false_normal

normal_accuracy = 100 * true_normal / normal_total

pneumonia_accuracy = 100 * true_pneumonia / pneumonia_total


# -----------------------------
# 9. Precision, Recall, F1
#    for PNEUMONIA
# -----------------------------

precision = (
    true_pneumonia /
    (true_pneumonia + false_pneumonia)
    if (true_pneumonia + false_pneumonia) > 0
    else 0
)

recall = (
    true_pneumonia /
    (true_pneumonia + false_normal)
    if (true_pneumonia + false_normal) > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)


# -----------------------------
# 10. Results
# -----------------------------

print("\n======================================")
print("           TEST RESULTS")
print("======================================")

print("Total test images:", total)

print(f"Overall Accuracy: {accuracy:.2f}%")

print(f"NORMAL Accuracy: {normal_accuracy:.2f}%")

print(f"PNEUMONIA Accuracy: {pneumonia_accuracy:.2f}%")

print("\n--------------------------------------")
print("CONFUSION MATRIX")
print("--------------------------------------")

print("                 Predicted")
print("                 NORMAL  PNEUMONIA")
print(
    f"Actual NORMAL    {true_normal:4d}     {false_pneumonia:4d}"
)
print(
    f"Actual PNEUMONIA {false_normal:4d}     {true_pneumonia:4d}"
)

print("\n--------------------------------------")
print("PNEUMONIA METRICS")
print("--------------------------------------")

print(f"Precision: {precision * 100:.2f}%")

print(f"Recall:    {recall * 100:.2f}%")

print(f"F1 Score:  {f1 * 100:.2f}%")

print("======================================")