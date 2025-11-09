"""
Utility for installing language and recognition models
"""
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

MODEL_URLS = {
    # Vosk ASR models
    "vosk": {
        "en": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "hi": "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
        "kn": "https://alphacephei.com/vosk/models/vosk-model-small-kn-0.22.zip"
    },
    # YOLO models
    "yolo": {
        "n": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "s": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        "m": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"
    }
}

def download_yolo_model(model_size='n', models_dir=None, progress_callback=None):
    """Download a YOLO model if not already present.
    
    Args:
        model_size (str): Size of model ('n', 's', 'm')
        models_dir (str): Path to models directory
        progress_callback (callable): Optional callback for progress updates
    
    Returns:
        Path: Path to downloaded model file
    """
    if model_size not in MODEL_URLS["yolo"]:
        raise ValueError(f"Invalid YOLO model size: {model_size}")
        
    if models_dir is None:
        models_dir = Path.cwd() / "models"
    else:
        models_dir = Path(models_dir)
        
    os.makedirs(models_dir, exist_ok=True)
    model_path = models_dir / f"yolov8{model_size}.pt"
    
    if not model_path.exists():
        if progress_callback:
            progress_callback(f"Downloading YOLOv8{model_size} model...")
        try:
            urllib.request.urlretrieve(MODEL_URLS["yolo"][model_size], model_path)
            if progress_callback:
                progress_callback("YOLO model downloaded successfully")
        except Exception as e:
            if os.path.exists(model_path):
                os.remove(model_path)
            raise RuntimeError(f"Failed to download YOLO model: {e}")
            
    return model_path

def install_vosk_model(lang_code, models_dir, progress_callback=None):
    """Install a Vosk model for the specified language.
    
    Args:
        lang_code (str): Language code ('en', 'hi', 'kn')
        models_dir (str): Path to models directory
        progress_callback (callable): Optional callback for progress updates
    
    Returns:
        bool: True if successful, False otherwise
    """
    if lang_code not in MODEL_URLS["vosk"]:
        if progress_callback:
            progress_callback(f"Language {lang_code} not supported")
        return False
        
    model_url = MODEL_URLS["vosk"][lang_code]
    model_dir = Path(models_dir)
    zip_path = model_dir / f"vosk-model-{lang_code}.zip"
    install_dir = model_dir / f"vosk-model-small-{lang_code}"
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Download the model
        if progress_callback:
            progress_callback(f"Downloading {lang_code} model...")
        urllib.request.urlretrieve(model_url, zip_path)
        
        # Extract the model
        if progress_callback:
            progress_callback("Extracting model files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        
        # Clean up
        os.remove(zip_path)
        if progress_callback:
            progress_callback(f"Model for {lang_code} installed successfully")
        return True
        
    except Exception as e:
        print(f"Error installing model for {lang_code}: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(install_dir):
            shutil.rmtree(install_dir)
        if progress_callback:
            progress_callback(f"Error installing model: {e}")
        return False

def get_installed_models(models_dir):
    """Get list of installed Vosk models.
    
    Args:
        models_dir (str): Path to models directory
        
    Returns:
        set: Set of installed language codes
    """
    installed = {"en"}  # English is required/default
    model_dir = Path(models_dir)
    
    if not model_dir.exists():
        return installed
        
    for item in model_dir.iterdir():
        if item.is_dir() and item.name.startswith("vosk-model-small-"):
            lang = item.name.split("-")[-1]
            if lang in ["hi", "kn"]:
                installed.add(lang)
                
    return installed