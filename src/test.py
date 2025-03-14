import torch
import torch.nn as nn
import torchvision.models as models
from preprocess import get_data_loaders
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load Test Data
_, _, test_loader = get_data_loaders("../data/")  # Assuming test_loader is defined in preprocess.py

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet34(pretrained=False)  # No need for pretraining during testing
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 37)  # 37 classes
)
model.load_state_dict(torch.load("resnet34_pets.pth", map_location=device))  # Load trained weights
model.to(device)
model.eval()

# Initialize Variables
y_true, y_pred = [], []

# Perform Testing
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))  # Bigger size
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(37), yticklabels=np.arange(37))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
