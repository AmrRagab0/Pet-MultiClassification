import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class OxfordPetDataset(Dataset):
    """Custom dataset for Oxford-IIIT Pet Dataset."""
    
    def __init__(self, img_dir, annotation_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_labels = []

        # Read annotation file (trainval.txt or test.txt)
        with open(annotation_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                img_name = parts[0] + ".jpg"  # Image file name
                label = int(parts[1]) - 1  # Convert to zero-based indexing
                self.image_labels.append((img_name, label))

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_name, label = self.image_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")  # Open image

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(data_dir="../data/", batch_size=32, val_split=0.2):
    """Loads the dataset using a custom dataset class."""
    transform = transforms.Compose([
         transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_dir = os.path.join(data_dir, "images")
    annotation_file = os.path.join(data_dir, "annotations/trainval.txt") 

    dataset = OxfordPetDataset(img_dir, annotation_file, transform)

    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders("../data/")
    print(f"Train size: {len(train_loader.dataset)}, Validation size: {len(val_loader.dataset)}")
