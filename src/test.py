import torch
import torch.nn as nn
import torchvision.models as models
from preprocess import get_data_loaders
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load test data from the dataset using the preprocess module
_, _, test_loader = get_data_loaders("../data/")  # Assuming test_loader is defined in preprocess.py

# Set up the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize ResNet34 model without pretraining for testing
model = models.resnet34(pretrained=False)  # No need for pretraining during testing
num_ftrs = model.fc.in_features  # Get the number of input features to the final layer

# Define the custom fully connected layer with dropout and 37 output classes
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Add dropout to reduce overfitting
    nn.Linear(num_ftrs, 37)  # 37 classes for the Oxford-IIIT Pet Dataset
)

# Load the trained model weights from file
model.load_state_dict(torch.load("../weights/best_model.pth", map_location=device))  # Load trained weights
model.to(device)  # Move the model to the selected device
model.eval()  # Set the model to evaluation mode (no gradient computation)

# Lists to store true and predicted labels for evaluation
y_true, y_pred = [], []

# Perform inference on the test set
with torch.no_grad():  # Disable gradient tracking for efficiency
    for images, labels in test_loader:  # Iterate over batches in the test loader
        images, labels = images.to(device), labels.to(device)  # Move data to device
        outputs = model(images)  # Forward pass to get model predictions
        _, predicted = torch.max(outputs, 1)  # Get the class with highest probability
        y_true.extend(labels.cpu().numpy())  # Append true labels to list (move to CPU)
        y_pred.extend(predicted.cpu().numpy())  # Append predicted labels to list (move to CPU)

# Print a detailed classification report with precision, recall, and F1-score
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))

# Generate and visualize the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)  # Compute confusion matrix
plt.figure(figsize=(12, 10))  # Set figure size for better readability
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(37), yticklabels=np.arange(37))  # Plot heatmap
plt.xlabel("Predicted Labels")  # Label for x-axis
plt.ylabel("True Labels")  # Label for y-axis
plt.title("Confusion Matrix")  # Title of the plot
plt.show()  # Display the plot