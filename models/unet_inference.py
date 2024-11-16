from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # If needed for frontend access
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
import io
import base64
import os
from segmentation_models_pytorch import Unet

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)  # Optional, for frontend requests

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the segmentation model
def create_unet_model(num_classes=1, in_channels=3):
    model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes,
    )
    return model

def load_model(weights_path):
    model = create_unet_model()
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model.to(device)

model = load_model("/Users/saiperam/Desktop/AIM-MedVisor/models/unet_spine_segmentation.pth")

# Helper function to preprocess the image
def preprocess_image(image, target_size=(256, 256)):
    original_image = image.convert("RGB")
    resized_image = original_image.resize(target_size)
    image_np = np.array(resized_image) / 255.0
    return original_image, image_np

# Helper function to calculate bounding boxes
def calculate_bounding_boxes(binary_mask, original_size, scale_x, scale_y, margin=10):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < 100:
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            bounding_boxes.append((
                max(0, x - margin),
                max(0, y - margin),
                min(original_size[0], x + w + margin),
                min(original_size[1], y + h + margin)
            ))
    return bounding_boxes

# Inference and bounding box drawing
def infer_and_draw_boxes(model, image):
    original_image, image_np = preprocess_image(image)
    image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        predicted_mask = torch.sigmoid(output).cpu().numpy()[0, 0]
    binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255

    draw_original = ImageDraw.Draw(original_image)
    scale_x, scale_y = original_image.size[0] / 256.0, original_image.size[1] / 256.0
    for x1, y1, x2, y2 in calculate_bounding_boxes(binary_mask, original_image.size, scale_x, scale_y):
        draw_original.rectangle([x1, y1, x2, y2], outline="red", width=5)

    # Convert the result to a BytesIO object for easy transfer
    buffered = io.BytesIO()
    original_image.save(buffered, format="PNG")
    buffered.seek(0)
    return buffered

# Define the API endpoint
@app.route('/segment', methods=['POST'])
def segment():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    image = Image.open(file)
    processed_image = infer_and_draw_boxes(model, image)

    # Return the processed image as a response
    return send_file(processed_image, mimetype='image/png')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)