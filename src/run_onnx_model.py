import torch
import torchvision.models as models
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import time

# Load the PyTorch model
model = models.resnet34(pretrained=False)  # Initialize without pretrained weights
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),  # Dropout rate as used in conversion; doesn't affect inference due to eval()
    nn.Linear(num_ftrs, 37)  # Assuming 37 output classes
)
model.load_state_dict(torch.load("../weights/best_model.pth", map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode (disables dropout)

# Load the ONNX model
session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])

# Generate dummy input (batch size 1, 3 channels, 224x224 image)
dummy_input = torch.randn(1, 3, 224, 224)
onnx_input = dummy_input.numpy()  # Convert to NumPy for ONNX Runtime

# Warm-up runs to stabilize performance
with torch.no_grad():
    pytorch_output = model(dummy_input)  # PyTorch warm-up
onnx_output = session.run(None, {"input": onnx_input})[0]  # ONNX warm-up, take first output

# Number of runs for timing
num_runs = 100

# Measure PyTorch inference time
start_time = time.time()
for _ in range(num_runs):
    with torch.no_grad():
        _ = model(dummy_input)
torch_time = (time.time() - start_time) / num_runs

# Measure ONNX inference time
start_time = time.time()
for _ in range(num_runs):
    _ = session.run(None, {"input": onnx_input})
onnx_time = (time.time() - start_time) / num_runs

# Check maximum output difference (optional, for correctness)
difference = np.abs(pytorch_output.numpy() - onnx_output).max()
print(f"Maximum output difference: {difference:.6f}")

# Print performance results
print(f"PyTorch Inference Time: {torch_time:.6f} seconds")
print(f"ONNX Inference Time: {onnx_time:.6f} seconds")
print(f"Speedup Factor (PyTorch/ONNX): {torch_time / onnx_time:.2f}x")