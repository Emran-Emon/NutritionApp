import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io

# Define the model architecture
def create_model():
    model = models.resnet18(weights=None)  # Create an uninitialized ResNet18
    model.fc = nn.Linear(model.fc.in_features, 101)  # Adjust the final layer to 101 classes
    return model

# Load model weights
def load_model():
    model = create_model()
    state_dict = torch.load("model/enhanced_trained_model.pth", map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Function to predict image class
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = outputs.max(1)
    return predicted.item()
