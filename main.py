import os
import uuid
import logging
import gc  # Garbage Collector for Memory Management
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

# --- OPTIMIZATION 1: Force CPU & Reduce Memory ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Standard Imports
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- CONFIGURATION ---
class Config:
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', './uploads')
    MODEL_PATH = os.environ.get('MODEL_PATH', 'model.h5')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB Limit
    PORT = int(os.environ.get('PORT', 5500))

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App Initialization
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- OPTIMIZATION 2: GLOBAL VARIABLES (LAZY LOADING) ---
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']
model = None  # Model ko abhi load mat karo (Start fast hoga)

def get_model():
    """Loads model only when needed (Lazy Loading)."""
    global model
    if model is None:
        logger.info("⏳ Loading model into memory (First Request)...")
        try:
            model = load_model(app.config['MODEL_PATH'])
            logger.info("✅ Model loaded successfully!")
        except Exception as e:
            logger.critical(f"❌ Failed to load model: {e}")
            return None
    return model

# --- HELPER FUNCTIONS ---

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_and_preprocess_image(image_path):
    target_size = (128, 128)
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Image Error: {e}")
        return None

# --- API ENDPOINTS ---

@app.route('/status', methods=['GET'])
def get_status():
    # Check status without crashing memory
    status = "Model Loaded" if model else "Waiting for First Request"
    return jsonify({"status": "ok", "message": "Server is running", "model": status})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    # 1. Validation
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid file"}), 400

    file_location = None
    try:
        # 2. Save File
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4()}.{ext}"
        file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_location)

        # 3. Load Model (Lazy Load happens here)
        active_model = get_model()
        if active_model is None:
            return jsonify({"status": "error", "message": "Model could not be loaded on server."}), 500

        # 4. Preprocess
        img_array = load_and_preprocess_image(file_location)
        if img_array is None:
            return jsonify({"status": "error", "message": "Could not process image"}), 400

        # 5. Predict
        predictions = active_model.predict(img_array)
        idx = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions, axis=1)[0])

        # Safety Check
        if idx >= len(class_labels):
            result = f"Unknown (Index {idx})"
            clean_type = "UNKNOWN"
        else:
            tumor_type = class_labels[idx]
            clean_type = tumor_type.upper()
            result = "CLEAN SCAN (No Tumor)" if "notumor" in tumor_type.lower() else f"TUMOR DETECTED: {clean_type}"

        # 6. Return Response
        return jsonify({
            "status": "success",
            "prediction": result,
            "tumor_type": clean_type,
            "confidence": confidence,
            "confidence_percent": f"{confidence*100:.2f}%"
        })

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return jsonify({"status": "error", "message": "Server Error"}), 500

    finally:
        # 7. OPTIMIZATION 3: CLEANUP & MEMORY RELEASE
        if file_location and os.path.exists(file_location):
            os.remove(file_location)
        
        # Explicitly free up memory for Render Free Tier
        gc.collect() 

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config['PORT'], debug=False)