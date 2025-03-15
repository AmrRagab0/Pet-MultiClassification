import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import os
import datetime
import glob
from preprocess import get_data_loaders

# Create directory for weights
os.makedirs("../weights", exist_ok=True)
# Load Data
train_loader, val_loader, _ = get_data_loaders("../data/")

# Load Pretrained ResNet34
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),  # Dropout added
    nn.Linear(num_ftrs, 37)  # 37 classes
)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9)  # Added momentum for better convergence
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=4)

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Track Metrics
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [],[]

# Training Loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation Phase
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Adjust Learning Rate
    scheduler.step(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# **Calculate Final Average Accuracy**
avg_train_acc = sum(train_accuracies) / len(train_accuracies)
avg_val_acc = sum(val_accuracies) / len(val_accuracies)

# **Generate Filename with Accuracy & Timestamp**
timestamp = datetime.datetime.now().strftime("%d-%H")  # Day-Hour
model_filename = f"weights/resnet34_{avg_val_acc:.2f}acc_{timestamp}.pth"

# Save Model
torch.save(model.state_dict(), model_filename)
print(f"Model saved as {model_filename}")

# **Keep Only Last 5 Models**
model_files = sorted(glob.glob("../weights/resnet34_*.pth"), key=os.path.getctime, reverse=True)
for old_file in model_files[5:]:  # Keep latest 5
    os.remove(old_file)
    print(f"Deleted old model: {old_file}")

# **Plot Accuracy & Loss**
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label="Train Accuracy", marker="o")
plt.plot(val_accuracies, label="Validation Accuracy", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training & Validation Accuracy")
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(train_losses, label="Train Loss", marker="o")
plt.plot(val_losses, label="Validation Loss", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

plt.show()
