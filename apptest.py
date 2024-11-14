import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
from segmentation_models_pytorch import Unet
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Model setup
def create_unet_model(num_classes=1, in_channels=3):
    model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes,
    )
    return model

# Load the model and weights
def load_model(weights_path):
    model = create_unet_model()
    try:
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint, strict=False)
    except FileNotFoundError:
        print(f"Error: Model weights not found at {weights_path}")
        return None
    model.eval()
    return model

# Initialize device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("/Users/saiperam/Desktop/AIM-MedVisor/models/unet_spine_segmentation.pth")
if model:
    model = model.to(device)

# Image preprocessing and bounding box visualization
def preprocess_image(image_path, target_size=(256, 256)):
    try:
        original_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return None, None
    resized_image = original_image.resize(target_size)
    image_np = np.array(resized_image) / 255.0
    return original_image, image_np

def calculate_bounding_boxes(binary_mask, original_size, scale_x, scale_y, margin=10):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < 100:
            # Scale and enlarge bounding box
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            bounding_boxes.append((
                max(0, x - margin),
                max(0, y - margin),
                min(original_size[0], x + w + margin),
                min(original_size[1], y + h + margin)
            ))
    return bounding_boxes

# Inference and visualization
def infer_and_visualize(model, image_path, save_folder):
    original_image, image_np = preprocess_image(image_path)
    if original_image is None:
        return None
    
    # Prepare image tensor
    image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_mask = torch.sigmoid(output).cpu().numpy()[0, 0]
    binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255

    # Draw bounding boxes
    draw_original = ImageDraw.Draw(original_image)
    scale_x, scale_y = original_image.size[0] / 256.0, original_image.size[1] / 256.0
    for x1, y1, x2, y2 in calculate_bounding_boxes(binary_mask, original_image.size, scale_x, scale_y):
        draw_original.rectangle([x1, y1, x2, y2], outline="red", width=5)

    # Save and display results directly in static/output
    # Automate filename generation based on input image
    output_filename = f"processed_{os.path.basename(image_path)}"
    save_path = os.path.join(save_folder, output_filename)
    original_image.save(save_path)
    
    return save_path

# Flask route to handle requests
@app.route('/')
def index():
    image_url = None
    if model:
        # Retrieve image path from request or use a default for testing
        image_path = request.args.get('image_path', '/Users/saiperam/Desktop/AIM-MedVisor/static/input/Lumbar-Spine-MRI.png')  # Replace with actual image path
        output_folder = os.path.join(app.root_path, 'static', 'output')  # Dynamic output folder
        os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists
        
        output_image_path = infer_and_visualize(model, image_path, save_folder=output_folder)
        
        if output_image_path:
            # Construct URL to serve image dynamically
            image_url = f"/static/output/{os.path.basename(output_image_path)}"
        else:
            image_url = None
            print("Inference failed or no output saved.")
    
    return render_template('index.html', image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
