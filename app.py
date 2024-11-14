from flask import Flask, send_file, jsonify
from models.working import process_image  # Import the function to process images
import os

app = Flask(__name__)

# Define absolute paths for the test image and output folder
TEST_IMAGE_PATH = "/Users/saiperam/Desktop/AIM-MedVisor/static/1-normal-lumbar-spine-mri-living-art-enterprises.jpg"  # Replace with the actual image path
OUTPUT_FOLDER = "/Users/saiperam/Desktop/AIM-MedVisor/output"  # Folder to save processed images
WEIGHTS_PATH = "/Users/saiperam/Desktop/AIM-MedVisor/models/unet_spine_segmentation.pth"  # Absolute path to model weights

# Ensure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/process', methods=['GET'])
def process_fixed_image():
    try:
        # Print the current working directory for debugging
        print("Current working directory:", os.getcwd())
        
        # Process the image with bounding boxes
        process_image(TEST_IMAGE_PATH, OUTPUT_FOLDER, WEIGHTS_PATH)  # Ensure process_image accepts weights path
        
        # Define the output path based on the processed image name
        output_image_path = os.path.join(OUTPUT_FOLDER, "final_results", f"original_with_boxes_{os.path.basename(TEST_IMAGE_PATH)}")
        
        # Check if the output image exists and send it as the response
        if os.path.exists(output_image_path):
            return send_file(output_image_path, mimetype='image/jpeg')
        else:
            return jsonify({"error": "Failed to generate output image"}), 500
    except Exception as e:
        # Return a JSON error response if any exceptions occur
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
