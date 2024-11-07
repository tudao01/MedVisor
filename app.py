from flask import Flask, request, jsonify, send_file
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import io

# Initialize Flask app
app = Flask(__name__)

# Load U-Net and ResNet-50 models
unet_model = tf.keras.models.load_model('models/unet_model.h5')   # Adjust based on your model type
resnet_model = tf.keras.models.load_model('models/resnet50_model.h5')

def preprocess_image(image):
    # Preprocess the image for U-Net model (e.g., resize, normalize)
    image = image.resize((256, 256))  # Example size
    image_np = np.array(image) / 255.0
    return image_np[np.newaxis, ...]  # Add batch dimension

def preprocess_for_resnet(segmented_output):
    # Resize and normalize U-Net output for ResNet model
    segmented_output = Image.fromarray((segmented_output * 255).astype(np.uint8))
    segmented_output = segmented_output.resize((224, 224))
    segmented_output_np = np.array(segmented_output) / 255.0
    return segmented_output_np[np.newaxis, ...]

def draw_bounding_boxes(image, segmentation_output):
    # Assume `segmentation_output` is binary (0 and 1) mask
    mask = (segmentation_output > 0.5).astype(np.uint8)  # Binarize output if needed
    mask_image = Image.fromarray(mask * 255).convert("RGB")
    draw = ImageDraw.Draw(mask_image)

    # Bounding box drawing logic
    # Example: Find contours and draw boxes around each detected region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

    return mask_image  # Return image with bounding boxes drawn

@app.route('/process', methods=['POST'])
def process():
    # Receive uploaded image
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    image = Image.open(image_file)

    # Step 1: Process image with U-Net
    preprocessed_image = preprocess_image(image)
    segmentation_output = unet_model.predict(preprocessed_image)
    segmented_image = (segmentation_output[0, :, :, 0] > 0.5).astype(np.uint8)

    # Step 2: Draw bounding boxes on U-Net output
    boxed_image = draw_bounding_boxes(image, segmented_image)

    # Step 3: Process the segmented output with ResNet-50
    preprocessed_segmentation = preprocess_for_resnet(segmented_image)
    classification_output = resnet_model.predict(preprocessed_segmentation)
    predicted_class = int(np.argmax(classification_output, axis=1)[0])  # Get the predicted class

    # Convert boxed image to byte stream to send it back as a file
    img_io = io.BytesIO()
    boxed_image.save(img_io, 'PNG')
    img_io.seek(0)

    # Return the classification and the modified image with bounding boxes
    return jsonify({'predicted_class': predicted_class, 'bounding_boxes_image': img_io})

if __name__ == '__main__':
    app.run(debug=True)