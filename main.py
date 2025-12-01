from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# --- Configuration ---
try:
    model = load_model('model.h5')
    print("Model 'model.h5' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# IMPORTANT: Ensure this matches your training data order
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
IMAGE_SIZE = 128

# --- Helper Functions ---

def load_and_preprocess_image(image_path):
    """Loads and preprocesses the image for the model."""
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_tumor(image_array, model, class_labels):
    """Returns the prediction result with SAFETY CHECKS."""
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    # --- SAFETY BLOCK (Prevents 500 Error) ---
    if predicted_class_index >= len(class_labels):
        # Fallback if model predicts an index outside our list
        print(f"WARNING: Model predicted index {predicted_class_index}, but we only have {len(class_labels)} labels.")
        tumor_type = f"Unknown (Index {predicted_class_index})"
    else:
        tumor_type = class_labels[predicted_class_index]
    # -----------------------------------------
        
    return tumor_type, float(confidence_score)

# --- API Endpoints ---

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({"status": "ok", "message": "API is running!"})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for image prediction."""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    file_location = None
    if file:
        filename = f"{uuid.uuid4()}_{file.filename}"
        file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_location)
            img_array = load_and_preprocess_image(file_location)
            
            if img_array is None:
                return jsonify({"status": "error", "message": "Could not process image."}), 400

            tumor_type, confidence = predict_tumor(img_array, model, class_labels)
            
            # Format Result for UI
            if 'notumor' in tumor_type.lower():
                result_label = "CLEAN SCAN (No Tumor Detected)"
                clean_type = "NEGATIVE"
            elif 'unknown' in tumor_type.lower():
                result_label = f"Model Error: {tumor_type}"
                clean_type = "UNKNOWN"
            else:
                result_label = f"TUMOR DETECTED: {tumor_type.upper()}"
                clean_type = tumor_type.upper()
            
            return jsonify({
                "status": "success",
                "prediction": result_label,
                "tumor_type": clean_type,
                "confidence": confidence, 
                "confidence_percent": f"{confidence*100:.2f}%"
            })

        except Exception as e:
            print(f"Error processing API request: {e}")
            return jsonify({"status": "error", "message": f"Server Error: {e}"}), 500
        finally:
            if os.path.exists(file_location):
                 os.remove(file_location)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Using 5500 to match your frontend setup
    app.run(debug=False, port=5500)