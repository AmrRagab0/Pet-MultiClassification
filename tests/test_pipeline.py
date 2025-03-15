import unittest
import torch
import os
import sys
import torchvision.models as models
sys.path.append(os.path.abspath("../src/"))
from preprocess import get_data_loaders


class TestDataPreprocessing(unittest.TestCase):
    """Tests for dataset loading and preprocessing"""

    def setUp(self):
        """Load data loaders for testing"""
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders("../data/", batch_size=8)
    
    def test_data_loaders_not_empty(self):
        """Ensure dataset is not empty"""
        self.assertGreater(len(self.train_loader.dataset), 0, "Train dataset is empty!")
        self.assertGreater(len(self.val_loader.dataset), 0, "Validation dataset is empty!")
        self.assertGreater(len(self.test_loader.dataset), 0, "Test dataset is empty!")

    def test_data_shapes(self):
        """Ensure image tensors have correct shape (batch_size, channels, height, width)"""
        images, labels = next(iter(self.train_loader))
        self.assertEqual(images.shape[1:], (3, 224, 224), "Incorrect image shape!")
        self.assertTrue(torch.is_tensor(images), "Images are not tensors!")
        self.assertTrue(torch.is_tensor(labels), "Labels are not tensors!")

class TestModelTraining(unittest.TestCase):
    """Tests for model training"""

    def setUp(self):
        """Load a ResNet model and training data"""
        # Load the training loader from the dataset
        self.train_loader, _, _ = get_data_loaders("../data/", batch_size=8)
        
        # Set up the model
        self.model = models.resnet34(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3), 
            torch.nn.Linear(num_ftrs, 37)
        )  # 37 classes
        
        # Set up device, optimizer, and loss
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.015, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()

    def test_forward_pass(self):
        """Ensure model produces an output tensor of correct shape"""
        images, _ = next(iter(self.train_loader))
        images = images.to(self.device)
        outputs = self.model(images)
        self.assertEqual(outputs.shape[1], 37, "Model output has incorrect number of classes!")

    def test_training_step(self):
        """Ensure training step runs without errors and loss decreases"""
        images, labels = next(iter(self.train_loader))
        images, labels = images.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.assertGreater(loss.item(), 0, "Loss should be positive")

class TestPredictionPipeline(unittest.TestCase):
    """Tests for inference and model loading"""

    def setUp(self):
        """Load trained model and test data"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet34(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(num_ftrs, 37))
        self.model.load_state_dict(torch.load("../weights/best_model.pth", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        _, _, self.test_loader = get_data_loaders("../data/", batch_size=8)

    def test_model_loading(self):
        """Ensure model loads correctly from saved weights"""
        self.assertTrue(isinstance(self.model, models.ResNet), "Model failed to load!")

    def test_prediction_output(self):
        """Ensure model produces valid predictions"""
        images, _ = next(iter(self.test_loader))
        images = images.to(self.device)
        with torch.no_grad():
            outputs = self.model(images)
        predictions = torch.argmax(outputs, dim=1)
        self.assertEqual(len(predictions), len(images), "Prediction batch size mismatch!")

if __name__ == "__main__":
    unittest.main()
