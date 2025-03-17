import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import os
import datetime
import glob
from preprocess import get_data_loaders

# Ensure the weights directory exists
os.makedirs("../weights", exist_ok=True)

# Load training and validation data
train_loader, val_loader, _ = get_data_loaders("../data/")

# Initialize ResNet34 with pretrained weights
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features  # Get input features for the final layer

# Replace the final layer with a custom one for 37 classes
model.fc = nn.Sequential(
    nn.Dropout(0.3),  # Add dropout to prevent overfitting
    nn.Linear(num_ftrs, 37)  # Output layer for 37 pet classes
)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9)  # SGD with momentum
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=4)  # Reduce LR on plateau

# Set up device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to the selected device

# Lists to store training and validation metrics
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Training loop for specified number of epochs
num_epochs = 30
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss, correct, total = 0.0, 0, 0  # Track loss and accuracy for the epoch

    for images, labels in train_loader:  # Iterate over training batches
        images, labels = images.to(device), labels.to(device)  # Move data to device
        
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()  # Accumulate loss
        _, predicted = torch.max(outputs, 1)  # Get predicted classes
        correct += (predicted == labels).sum().item()  # Count correct predictions
        total += labels.size(0)  # Count total samples
    
    train_loss = running_loss / len(train_loader)  # Average training loss
    train_acc = 100 * correct / total  # Training accuracy percentage
    train_losses.append(train_loss)  # Store loss
    train_accuracies.append(train_acc)  # Store accuracy

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss, val_correct, val_total = 0.0, 0, 0  # Track validation metrics
    with torch.no_grad():  # No gradient computation during validation
        for images, labels in val_loader:  # Iterate over validation batches
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            val_loss += loss.item()  # Accumulate loss
            _, predicted = torch.max(outputs, 1)  # Get predicted classes
            val_correct += (predicted == labels).sum().item()  # Count correct predictions
            val_total += labels.size(0)  # Count total samples

    val_loss /= len(val_loader)  # Average validation loss
    val_acc = 100 * val_correct / val_total  # Validation accuracy percentage
    val_losses.append(val_loss)  # Store loss
    val_accuracies.append(val_acc)  # Store accuracy

    # Adjust learning rate based on validation loss
    scheduler.step(val_loss)

    # Print epoch summary
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# Calculate average accuracies over all epochs
avg_train_acc = sum(train_accuracies) / len(train_accuracies)
avg_val_acc = sum(val_accuracies) / len(val_accuracies)

# Generate a unique filename with accuracy and timestamp
timestamp = datetime.datetime.now().strftime("%d-%H")  # Day-Hour format
model_filename = f"weights/resnet34_{avg_val_acc:.2f}acc_{timestamp}.pth"

# Save the trained model weights
torch.save(model.state_dict(), model_filename)
print(f"Model saved as {model_filename}")

# Keep only the 5 most recent model files
model_files = sorted(glob.glob("../weights/resnet34_*.pth"), key=os.path.getctime, reverse=True)
for old_file in model_files[5:]:  # Delete files beyond the latest 5
    os.remove(old_file)
    print(f"Deleted old model: {old_file}")

# Plot training and validation metrics
plt.figure(figsize=(12, 5))  # Set figure size

# Plot accuracy
plt.subplot(1, 2, 1)  # First subplot (1 row, 2 columns, position 1)
plt.plot(train_accuracies, label="Train Accuracy", marker="o")  # Plot training accuracy
plt.plot(val_accuracies, label="Validation Accuracy", marker="o")  # Plot validation accuracy
plt.xlabel("Epochs")  # X-axis label
plt.ylabel("Accuracy (%)")  # Y-axis label
plt.title("Training & Validation Accuracy")  # Plot title
plt.legend()  # Add legend

# Plot loss
plt.subplot(1, 2, 2)  # Second subplot (1 row, 2 columns, position 2)
plt.plot(train_losses, label="Train Loss", marker="o")  # Plot training loss
plt.plot(val_losses, label="Validation Loss", marker="o")  # Plot validation loss
plt.xlabel("Epochs")  # X-axis label
plt.ylabel("Loss")  # Y-axis label
plt.title("Training & Validation Loss")  # Plot title
plt.legend()  # Add legend

plt.show()  # Display the plots