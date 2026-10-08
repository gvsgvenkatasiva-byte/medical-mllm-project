import torch
import torch.nn as nn
import torch.optim as optim

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
# 3. Training preprocessing
# -----------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# 4. Validation preprocessing
# -----------------------------

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# 5. Load datasets
# -----------------------------

train_dataset = datasets.ImageFolder(
    DATA_PATH + r"\train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_PATH + r"\val",
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Classes:", train_dataset.classes)


# -----------------------------
# 6. Calculate class weights
# -----------------------------

class_counts = torch.bincount(
    torch.tensor(train_dataset.targets)
)

class_weights = len(train_dataset) / (
    2 * class_counts.float()
)

class_weights = class_weights.to(device)

print("Class counts:", class_counts.tolist())
print("Class weights:", class_weights.tolist())


# -----------------------------
# 7. Create ResNet18
# -----------------------------

model = models.resnet18(weights="DEFAULT")

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model = model.to(device)


# -----------------------------
# 8. Loss and optimizer
# -----------------------------

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# -----------------------------
# 9. Training
# -----------------------------

epochs = 3

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    print("\n==============================")
    print("Epoch", epoch + 1, "of", epochs)
    print("==============================")

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        if (batch_index + 1) % 100 == 0:

            accuracy = (
                100 * correct / total
            )

            print(
                f"Batch {batch_index + 1}/{len(train_loader)} "
                f"Loss: {loss.item():.4f} "
                f"Accuracy: {accuracy:.2f}%"
            )

    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    epoch_accuracy = (
        100 * correct / total
    )

    print("\nEpoch completed!")

    print(
        f"Average Loss: {epoch_loss:.4f}"
    )

    print(
        f"Training Accuracy: {epoch_accuracy:.2f}%"
    )


# -----------------------------
# 10. Save improved model
# -----------------------------

torch.save(
    model.state_dict(),
    "models/pneumonia_resnet18_improved.pth"
)

print("\n================================")
print("IMPROVED MODEL SAVED")
print("================================")

print(
    "models/pneumonia_resnet18_improved.pth"
)