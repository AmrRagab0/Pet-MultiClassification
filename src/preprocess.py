import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torchvision import transforms
from PIL import Image

class OxfordPetDataset(Dataset):
    """Custom dataset for Oxford-IIIT Pet Dataset."""
    
    def __init__(self, img_dir, annotation_file, transform=None):
        self.img_dir = img_dir  # Directory containing the images
        self.transform = transform  # Image transformations (e.g., resizing, normalization)
        self.image_labels = []  # List to store (image filename, label) pairs

        # Read annotation file (trainval.txt or test.txt) and populate image_labels
        with open(annotation_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()  # Split each line into components
                img_name = parts[0] + ".jpg"  # Construct image filename (e.g., "Abyssinian_1.jpg")
                label = int(parts[1]) - 1  # Convert label to zero-based indexing (1-37 becomes 0-36)
                self.image_labels.append((img_name, label))  # Add tuple to list

    def __len__(self):
        return len(self.image_labels)  # Return the total number of samples

    def __getitem__(self, idx):
        img_name, label = self.image_labels[idx]  # Get image filename and label at index
        img_path = os.path.join(self.img_dir, img_name)  # Construct full image path
        image = Image.open(img_path).convert("RGB")  # Load image and convert to RGB

        if self.transform:
            image = self.transform(image)  # Apply transformations if specified

        return image, label  # Return image tensor and label


def get_data_loaders(data_dir="../data/", batch_size=32, val_split=0.2, mock=False):
    if mock:
        # Create synthetic data for testing when mock mode is enabled
        mock_images = torch.randn(10, 3, 224, 224)  # 10 random images (3 channels, 224x224)
        mock_labels = torch.randint(0, 37, (10,))  # 10 random labels (0-36)
        dataset = TensorDataset(mock_images, mock_labels)  # Create a mock dataset
        return (
            DataLoader(dataset, batch_size=batch_size),  # Train loader
            DataLoader(dataset, batch_size=batch_size),  # Validation loader
            DataLoader(dataset, batch_size=batch_size)  # Test loader
        )
    else:
        """Loads the dataset using a custom dataset class."""
        # Define image transformations for data augmentation and normalization
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  # Randomly crop and resize to 224x224
            transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness/contrast
            transforms.RandomRotation(10),  # Rotate by up to 10 degrees
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])

        # Set paths for images and annotations
        img_dir = os.path.join(data_dir, "images")  # Directory with pet images
        annotation_file = os.path.join(data_dir, "annotations/trainval.txt")  # Training/validation annotations
        test_annotation_file = os.path.join(data_dir, "annotations/test.txt")  # Test annotations

        # Initialize datasets with the custom class
        dataset = OxfordPetDataset(img_dir, annotation_file, transform)  # Full train/val dataset
        test_dataset = OxfordPetDataset(img_dir, test_annotation_file, transform)  # Test dataset

        # Split train dataset into train and validation sets
        train_size = int((1 - val_split) * len(dataset))  # Calculate training set size
        val_size = len(dataset) - train_size  # Calculate validation set size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  # Split dataset

        # Create data loaders for training, validation, and testing
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle for training
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for validation
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for testing

        return train_loader, val_loader, test_loader  # Return the three loaders


if __name__ == "__main__":
    # Test the data loaders when running this file directly
    train_loader, val_loader, test_loader = get_data_loaders("../data/")
    print(f"Train size: {len(train_loader.dataset)}, Validation size: {len(val_loader.dataset)}")