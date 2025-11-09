import os
import time
import requests
import cv2
import numpy as np
import speech_recognition as sr
import langdetect
import winsound

def load_object_detection_model():
    """Load MobileNet-SSD if available; skip download if not."""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    PROTOTXT_PATH = os.path.join(models_dir, "MobileNetSSD_deploy.prototxt")
    MODEL_PATH = os.path.join(models_dir, "MobileNetSSD_deploy.caffemodel")

    # Skip download if both files are already there
    print("[Debug] Current working directory:", os.getcwd())
    print("[Debug] Looking for model files in:", os.path.abspath(models_dir))
    if os.path.exists(PROTOTXT_PATH) and os.path.exists(MODEL_PATH):
        try:
            net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
            print("[Model] MobileNet-SSD loaded successfully.")
            return net
        except Exception as e:
            print(f"[Model load error] {e}")
            return None
    else:
        print("Model files not found. Please download and place them in /models/")
        return None
load_object_detection_model()