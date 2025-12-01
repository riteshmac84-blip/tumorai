# predict.py
import sys
import os
import json

# Suppress TF Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def run_prediction(image_path, model_path):
    try:
        # Load Model
        model = load_model(model_path)
        
        # Load Image
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        idx = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions, axis=1)[0])
        
        return {"idx": idx, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Command line arguments: python predict.py <img_path> <model_path>
    img_path = sys.argv[1]
    model_path = sys.argv[2]
    
    result = run_prediction(img_path, model_path)
    print(json.dumps(result))