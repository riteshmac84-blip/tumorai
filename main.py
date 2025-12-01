# main.py
import os
import uuid
import json
import subprocess # <-- New Hero
from flask import Flask, render_template, request, jsonify

class Config:
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', './uploads')
    MODEL_PATH = os.environ.get('MODEL_PATH', 'model.h5')
    PORT = int(os.environ.get('PORT', 5500))

app = Flask(__name__)
app.config.from_object(Config)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No file"}), 400

    filename = f"{uuid.uuid4()}.jpg"
    file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_location)

    try:
        # --- Run Prediction in Separate Process ---
        # This prevents RAM buildup in the main server
        result = subprocess.check_output(
            ["python", "predict.py", file_location, app.config['MODEL_PATH']],
            stderr=subprocess.STDOUT
        )
        
        # Parse Result
        output = json.loads(result.decode('utf-8').strip().split('\n')[-1])
        
        if "error" in output:
            return jsonify({"status": "error", "message": output["error"]}), 500

        idx = output["idx"]
        confidence = output["confidence"]

        if idx >= len(class_labels):
            clean_type = "UNKNOWN"
            result_text = "Unknown"
        else:
            tumor_type = class_labels[idx]
            clean_type = tumor_type.upper()
            result_text = "CLEAN SCAN (No Tumor)" if "notumor" in tumor_type.lower() else f"TUMOR DETECTED: {clean_type}"

        return jsonify({
            "status": "success",
            "prediction": result_text,
            "tumor_type": clean_type,
            "confidence": confidence,
            "confidence_percent": f"{confidence*100:.2f}%"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config['PORT'], debug=False)