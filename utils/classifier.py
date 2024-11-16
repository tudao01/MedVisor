# utils/classifier.py

import torch
from PIL import Image
from torchvision import transforms

# Load models
binary_model = torch.load('models/binary_model.pth')
nonbinary_model = torch.load('models/nonbinary_model.pth')
binary_model.eval()
nonbinary_model.eval()

# Define preprocessing for ResNet
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def classify_disc(disc_image):
    # Preprocess image for model input
    input_tensor = preprocess(disc_image).unsqueeze(0)
    
    # Run binary model
    binary_prediction = binary_model(input_tensor).argmax().item()
    
    # Run non-binary model
    nonbinary_prediction = nonbinary_model(input_tensor).argmax().item()
    
    return binary_prediction, nonbinary_prediction
