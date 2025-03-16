import torch
import torchvision.models as models
import torch.nn as nn

# Load your trained model
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),  
    nn.Linear(num_ftrs, 37)  
)
model.load_state_dict(torch.load("../weights/best_model.pth", map_location=torch.device('cpu')))
model.eval()

# Dummy input (adjust shape as needed)
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx",
    export_params=True,        # Store trained parameters
    opset_version=11,          # Use opset 11 (adjust if needed)
    do_constant_folding=True,  # Optimize constant expressions
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Enable dynamic batching
)

print("Model successfully converted to ONNX!")
