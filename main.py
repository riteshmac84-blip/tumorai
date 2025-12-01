import os
import logging
import numpy as np
import io  # <-- New: For In-Memory Processing
from PIL import Image # <-- New: Faster Image Handling
from flask import Flask, render_template, request, jsonify

# --- OPTIMIZATION 1: CPU Optimization ---
# TensorFlow logs aur GPU check ko disable karein taaki startup fast ho
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- CONFIGURATION ---
class Config:
    MODEL_PATH = os.environ.get('MODEL_PATH', 'model.h5')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB Limit
    PORT = int(os.environ.get('PORT', 5500))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# --- OPTIMIZATION 2: PRE-LOAD MODEL (NO LAZY LOADING) ---
# High Traffic ke liye Model pehle se Memory mein hona chahiye.
# Lazy loading 100 users aane par system crash kar dega.
try:
    logger.info("🚀 Loading AI Model into Memory...")
    model = load_model(app.config['MODEL_PATH'])
    logger.info("✅ Model Loaded & Ready for High Traffic!")
except Exception as e:
    logger.critical(f"❌ Model Load Failed: {e}")
    model = None

class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# --- HELPER FUNCTIONS (Optimized) ---

def process_image_in_memory(file_stream):
    """
    Reads image directly from RAM without saving to disk.
    Super fast for high concurrency.
    """
    try:
        # Open image from bytes
        image = Image.open(file_stream)
        
        # Ensure RGB (Handling PNG/Grayscale issues)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Resize using Pillow (Faster than Keras load_img)
        image = image.resize((128, 128))
        
        # Convert to Array
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Image Processing Error: {e}")
        return None

# --- API ENDPOINTS ---

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({"status": "ok", "message": "High Performance Server Running"})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    # 1. Fast Validation
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No filename"}), 400

    try:
        # --- OPTIMIZATION 3: DIRECT STREAM PROCESSING ---
        # file.save() hata diya gaya hai. Seedha RAM se read karein.
        img_array = process_image_in_memory(file.stream)
        
        if img_array is None:
            return jsonify({"error": "Invalid Image Format"}), 400

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # 4. Predict
        predictions = model.predict(img_array)
        idx = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions, axis=1)[0])

        # Safety Check
        if idx >= len(class_labels):
            result = "Unknown"
            clean_type = "UNKNOWN"
        else:
            tumor_type = class_labels[idx]
            clean_type = tumor_type.upper()
            result = "CLEAN SCAN (No Tumor)" if "notumor" in tumor_type.lower() else f"TUMOR DETECTED: {clean_type}"

        # 5. Fast Response
        return jsonify({
            "status": "success",
            "prediction": result,
            "tumor_type": clean_type,
            "confidence": confidence,
            "confidence_percent": f"{confidence*100:.2f}%"
        })

    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({"status": "error", "message": "Processing Failed"}), 500

# Frontend Route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config['PORT'], debug=False)