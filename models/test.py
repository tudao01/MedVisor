import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

class UNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
        super(UNet, self).__init__()

        # Encoder - Using ResNet as the encoder
        self.encoder = self.get_encoder(encoder_name, encoder_weights, in_channels)
        # Decoder - Define U-Net style decoder (e.g., using transposed convolutions or upsampling)
        self.decoder = self.get_decoder()
        # Final segmentation layer
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)  # Modify this if necessary
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary segmentation

    def get_encoder(self, encoder_name, encoder_weights, in_channels):
        # Initialize a pre-trained encoder (e.g., ResNet34)
        if encoder_name == "resnet34":
            encoder = models.resnet34(weights="IMAGENET1K_V1" if encoder_weights == "imagenet" else None)
            # Modify the first conv layer to match the number of input channels
            encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
            # Remove the fully connected layers (classifier)
            encoder.fc = nn.Identity()  # So it won't interfere with the decoder

            # Optionally, select intermediate layers (e.g., after the last residual block)
            return nn.Sequential(*list(encoder.children())[:-2])  # Remove the fully connected layers

        # Add other encoder options as needed
        raise ValueError(f"Encoder {encoder_name} not supported")

    def get_decoder(self):
        # Define the decoder layers here
        return nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Forward pass through encoder
        encoder_out = self.encoder(x)
        # Forward pass through decoder
        decoder_out = self.decoder(encoder_out)
        # Final convolution to reduce channels to the number of classes
        x = self.final_conv(decoder_out)
        # Apply sigmoid activation to output
        return self.sigmoid(x)  # Use sigmoid for binary output

# Example usage
model = UNet()
model.eval()

# Define image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Adjust as needed
])

# Load and preprocess the image
image_path = '/Users/saiperam/Desktop/AIM-MedVisor/input/1-normal-lumbar-spine-mri-living-art-enterprises.jpg'
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Check the input tensor shape
print("Input tensor shape:", image_tensor.shape)  # Should be [1, 3, 256, 256]

# Inference
with torch.no_grad():
    output = model(image_tensor)

# Visualize the output
output_image = output.squeeze().cpu().numpy()  # Remove batch dimension and convert to numpy

# Optional: Apply a threshold for segmentation (e.g., 0.5 for binary segmentation)
output_image = (output_image > 0.5).astype(float)  # Thresholding

# Plot the output
plt.imshow(output_image, cmap='gray')
plt.show()
