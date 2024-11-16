import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os

class UNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
        super(UNet, self).__init__()
        self.encoder = self.get_encoder(encoder_name, encoder_weights, in_channels)
        self.decoder = self.get_decoder()
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

    def get_encoder(self, encoder_name, encoder_weights, in_channels):
        if encoder_name == "resnet34":
            encoder = models.resnet34(pretrained=encoder_weights == "imagenet")
            encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            return encoder
        raise ValueError(f"Encoder {encoder_name} not supported")

    def get_decoder(self):
        return nn.ConvTranspose2d(512, 64, kernel_size=2, stride=2)

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return self.final_conv(decoder_out)

def get_bounding_boxes(model, image_path, save_folder, device="cpu"):
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size

    # Preprocess image
    image = original_image.resize((256, 256))
    image_np = np.array(image) / 255.0
    image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        predicted_mask = torch.sigmoid(output).cpu().numpy()[0, 0]

    # Threshold and find contours
    binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    draw_original = ImageDraw.Draw(original_image)
    scale_x = original_size[0] / 256.0
    scale_y = original_size[1] / 256.0
    margin = 10

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < 100:
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            enlarged_x1 = max(0, x - margin)
            enlarged_y1 = max(0, y - margin)
            enlarged_x2 = min(original_size[0], x + w + margin)
            enlarged_y2 = min(original_size[1], y + h + margin)
            draw_original.rectangle([enlarged_x1, enlarged_y1, enlarged_x2, enlarged_y2], outline="red", width=5)

    final_results_folder = os.path.join(save_folder, "final_results")
    os.makedirs(final_results_folder, exist_ok=True)
    filename_original = f"original_with_boxes_{os.path.basename(image_path)}"
    save_path_original = os.path.join(final_results_folder, filename_original)
    original_image.save(save_path_original)

    return save_path_original
