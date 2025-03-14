import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from preprocess import get_data_loaders

# Load Data
train_loader, val_loader = get_data_loaders("../data/")

# Load Pretrained ResNet34
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 37)  # 37 classes for pet dataset

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save Model
torch.save(model.state_dict(), "resnet34_pets.pth")
print("Training complete. Model saved as resnet34_pets.pth")
