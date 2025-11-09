import os
import datetime
import time
import requests
import cv2
import numpy as np
import speech_recognition as sr
import langdetect
import winsound
import sys
import random
import time
from utils.speech_utils import speak, listen_for_speech
# import requests
import pyttsx3
# import speech_recognition as sr
from datetime import datetime, timedelta
import re
import threading
import queue
import urllib.parse
import webbrowser
import psutil
import openrouteservice
from geopy.geocoders import Nominatim
import math
import json
import torch
import torch.nn.functional as F
from ultralytics import YOLO
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except Exception as _imp_err:
    BlipProcessor = None
    BlipForConditionalGeneration = None
from PIL import Image
import pytesseract
from pytesseract import Output
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[EasyOCR Warning] EasyOCR not available. Handwriting recognition will be limited.")
import sounddevice as sd
import asyncio, edge_tts
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[MediaPipe Warning] MediaPipe not available. Pose detection will be limited.")

# Make asyncio.run tolerant when given a non-coroutine (e.g., speak("..."))
_original_asyncio_run = asyncio.run
def _safe_asyncio_run(maybe_coro):
    try:
        if asyncio.iscoroutine(maybe_coro):
            return _original_asyncio_run(maybe_coro)
        # If it's not a coroutine, just return; the side-effect already happened
        return None
    except Exception:
        return None
asyncio.run = _safe_asyncio_run

# -------------------------
# Initialize global models
# -------------------------
# engine = pyttsx3.init('sapi5')
local_model = YOLO("yolov8n.pt")  # local detection model
try:
    if BlipProcessor is not None and BlipForConditionalGeneration is not None:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    else:
        processor = None
        model_blip = None
except Exception as _blip_err_top:
    processor = None
    model_blip = None
    print("[BLIP Load Warning]", _blip_err_top)

depth_model = None
depth_transform = None
depth_device = None

# engine.setProperty('rate', 180)     # speaking speed
# engine.setProperty('volume', 1.0)  


def detect_scream(audio_data, sample_rate):
    """
    Detects if the audio contains a potential scream.
    Logic: checks for high loudness + high-frequency energy.
    """
    # Compute loudness (RMS)
    rms = np.sqrt(np.mean(audio_data ** 2))
    
    # Compute frequency content (FFT)
    freqs = np.fft.rfftfreq(len(audio_data), 1 / sample_rate)
    spectrum = np.abs(np.fft.rfft(audio_data))
    
    # Energy in high frequencies (> 2 kHz)
    high_freq_energy = np.sum(spectrum[freqs > 2000])
    total_energy = np.sum(spectrum)
    high_freq_ratio = high_freq_energy / (total_energy + 1e-9)
    
    # Thresholds — tweak for your environment
    if rms > 0.6 and high_freq_ratio > 0.25:
        return True
    return False


def listen_for_audio(duration=2, sample_rate=44100):
    """Records a short audio clip from the microphone."""
    print("🎤 Listening for scream or loud noise...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return recording.flatten()


def start_emergency_mode():
    """The actual emergency routine."""
    print("🚨 EMERGENCY MODE ACTIVATED 🚨")
    print("Calling emergency services...")
    print("Sending location and alert to contacts...")
    # Add your actual emergency handling logic here


def run_emergency_mode():
    """
    Monitors for text or scream-based emergency triggers.
    """
    emergency_words = [
        "emergency", "emergency mode", "help me", "i need help",
        "sos", "activate emergency mode", "trigger emergency",
        "emergency please", "start emergency", "call emergency",
        "guide this is an emergency"
    ]

    while True:
        # Step 1: Check for voice-based distress
        audio_data = listen_for_audio()
        scream_detected = detect_scream(audio_data, 44100)

        # Step 2: Ask for user text input (optional fallback)
        mode_input = input("Type or say command (or 'exit' to quit): ").lower().strip()
        if mode_input == "exit":
            print("Exiting emergency listener.")
            break

        # Step 3: Check if emergency words were said
        emergency_triggered = any(word in mode_input for word in emergency_words)

        # Step 4: Decision logic
        if emergency_triggered or scream_detected:
            if scream_detected and not emergency_triggered:
                confirm = input("⚠️ I heard a scream. Should I start emergency mode? (yes/no): ").lower()
                if confirm in ["yes", "y"]:
                    start_emergency_mode()
                else:
                    print("Okay, staying in normal mode.")
            else:
                start_emergency_mode()
        else:
            print("No emergency detected. Monitoring continues...\n")
            time.sleep(1)  # small delay to prevent overload

engine = pyttsx3.init()

# ===============================
# DOOR AND WALL DETECTION FUNCTIONS
# ===============================

def detect_door(frame):
    """
    Detect doors in the frame using computer vision techniques.
    Returns: dict with 'detected' (bool), 'position' (str), 'distance' (str), 'bbox' (list)
    """
    h, w = frame.shape[:2]
    result = {
        'detected': False,
        'position': None,
        'distance': None,
        'bbox': None,
        'confidence': 0.0
    }
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            # Find vertical lines (potential door frame)
            vertical_lines = []
            horizontal_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Vertical lines (door frame sides)
                if angle > 75 and angle < 105:
                    vertical_lines.append((x1, y1, x2, y2))
                # Horizontal lines (door frame top/bottom)
                elif angle < 15 or angle > 165:
                    horizontal_lines.append((x1, y1, x2, y2))
            
            # Check for rectangular door frame pattern
            if len(vertical_lines) >= 2 and len(horizontal_lines) >= 1:
                # Find vertical line pairs that could form door sides
                vertical_lines.sort(key=lambda x: x[0])  # Sort by x position
                
                # Look for parallel vertical lines (door sides)
                door_candidates = []
                for i in range(len(vertical_lines)):
                    for j in range(i + 1, len(vertical_lines)):
                        x1_1, y1_1, x2_1, y2_1 = vertical_lines[i]
                        x1_2, y1_2, x2_2, y2_2 = vertical_lines[j]
                        
                        # Check if lines are roughly parallel and form a rectangle
                        avg_x1 = (x1_1 + x2_1) / 2
                        avg_x2 = (x1_2 + x2_2) / 2
                        width = abs(avg_x2 - avg_x1)
                        
                        # Typical door width is 0.6-1.0m, in image terms roughly 15-30% of frame width
                        if 0.15 * w < width < 0.35 * w:
                            # Find top and bottom y coordinates
                            top_y = min(y1_1, y2_1, y1_2, y2_2)
                            bottom_y = max(y1_1, y2_1, y1_2, y2_2)
                            height = bottom_y - top_y
                            
                            # Typical door height is 2.0-2.5m, in image terms roughly 40-70% of frame height
                            if 0.3 * h < height < 0.8 * h:
                                left_x = min(avg_x1, avg_x2)
                                right_x = max(avg_x1, avg_x2)
                                door_candidates.append({
                                    'bbox': [int(left_x), int(top_y), int(right_x), int(bottom_y)],
                                    'width': width,
                                    'height': height,
                                    'center_x': (left_x + right_x) / 2
                                })
                
                if door_candidates:
                    # Select the most prominent door (largest, most centered)
                    best_door = max(door_candidates, key=lambda d: d['width'] * d['height'])
                    result['detected'] = True
                    result['bbox'] = best_door['bbox']
                    result['confidence'] = 0.7
                    
                    # Determine position
                    center_x = best_door['center_x']
                    if center_x < w * 0.33:
                        result['position'] = 'left'
                    elif center_x > w * 0.67:
                        result['position'] = 'right'
                    else:
                        result['position'] = 'center'
                    
                    # Estimate distance based on door size in frame
                    door_area = best_door['width'] * best_door['height']
                    frame_area = w * h
                    area_ratio = door_area / frame_area
                    
                    if area_ratio > 0.3:
                        result['distance'] = 'very close'
                    elif area_ratio > 0.15:
                        result['distance'] = 'close'
                    elif area_ratio > 0.08:
                        result['distance'] = 'moderate'
                    else:
                        result['distance'] = 'far'
        
        # Also try using YOLO model for door detection
        try:
            results = local_model(frame, verbose=False)
            for result_yolo in results:
                for box in result_yolo.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    obj_name = local_model.names[cls]
                    
                    # Check for door-related objects
                    if conf > 0.4 and ("door" in obj_name.lower() or "doorframe" in obj_name.lower()):
                        if conf > result['confidence']:
                            result['detected'] = True
                            result['confidence'] = conf
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            result['bbox'] = [int(x1), int(y1), int(x2), int(y2)]
                            
                            # Determine position
                            center_x = (x1 + x2) / 2
                            if center_x < w * 0.33:
                                result['position'] = 'left'
                            elif center_x > w * 0.67:
                                result['position'] = 'right'
                            else:
                                result['position'] = 'center'
                            
                            # Estimate distance
                            door_area = (x2 - x1) * (y2 - y1)
                            frame_area = w * h
                            area_ratio = door_area / frame_area
                            
                            if area_ratio > 0.3:
                                result['distance'] = 'very close'
                            elif area_ratio > 0.15:
                                result['distance'] = 'close'
                            elif area_ratio > 0.08:
                                result['distance'] = 'moderate'
                            else:
                                result['distance'] = 'far'
        except Exception as e:
            print(f"[Door Detection YOLO Error]: {e}")
        
        # Additional check: Look for rectangular contours that could be doors
        if not result['detected']:
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5000:  # Minimum area threshold
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check if it's roughly rectangular (4 corners)
                    if len(approx) >= 4:
                        x, y, w_rect, h_rect = cv2.boundingRect(approx)
                        aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0
                        
                        # Door aspect ratio is typically 0.3-0.5 (taller than wide)
                        if 0.25 < aspect_ratio < 0.6 and h_rect > 0.3 * h:
                            if not result['detected'] or result['confidence'] < 0.5:
                                result['detected'] = True
                                result['confidence'] = 0.5
                                result['bbox'] = [x, y, x + w_rect, y + h_rect]
                                
                                center_x = x + w_rect / 2
                                if center_x < w * 0.33:
                                    result['position'] = 'left'
                                elif center_x > w * 0.67:
                                    result['position'] = 'right'
                                else:
                                    result['position'] = 'center'
                                
                                # Estimate distance
                                door_area = w_rect * h_rect
                                frame_area = w * h
                                area_ratio = door_area / frame_area
                                
                                if area_ratio > 0.3:
                                    result['distance'] = 'very close'
                                elif area_ratio > 0.15:
                                    result['distance'] = 'close'
                                elif area_ratio > 0.08:
                                    result['distance'] = 'moderate'
                                else:
                                    result['distance'] = 'far'
                                break
        
    except Exception as e:
        print(f"[Door Detection Error]: {e}")
    
    return result


def detect_wall(frame):
    """
    Detect walls in the frame using depth analysis and edge detection.
    Returns: dict with 'detected' (bool), 'position' (str), 'distance' (str), 'direction' (str)
    """
    h, w = frame.shape[:2]
    result = {
        'detected': False,
        'position': None,
        'distance': None,
        'direction': None,  # 'ahead', 'left', 'right'
        'confidence': 0.0
    }
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Analyze texture - walls typically have low texture variation
        # Divide frame into regions
        center_region = gray[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
        left_region = gray[int(h*0.3):int(h*0.7), :int(w*0.3)]
        right_region = gray[int(h*0.3):int(h*0.7), int(w*0.7):]
        
        center_std = np.std(center_region)
        left_std = np.std(left_region)
        right_std = np.std(right_region)
        
        # Low texture variation suggests a wall
        texture_threshold = 15.0
        
        # Check center region (wall ahead)
        if center_std < texture_threshold:
            # Additional check: look for vertical edges (wall boundaries)
            edges = cv2.Canny(gray, 50, 150)
            vertical_lines = []
            
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    # Vertical lines
                    if angle > 70 and angle < 110:
                        vertical_lines.append((x1, y1, x2, y2))
            
            # If we have many vertical lines in center, it's likely a wall
            center_vertical_count = sum(1 for line in vertical_lines 
                                      if int(w*0.3) < (line[0] + line[2])/2 < int(w*0.7))
            
            if center_vertical_count > 3 or center_std < 10:
                result['detected'] = True
                result['direction'] = 'ahead'
                result['confidence'] = 0.8 if center_std < 10 else 0.6
                
                # Estimate distance based on texture uniformity and edge density
                if center_std < 8:
                    result['distance'] = 'very close'
                elif center_std < 12:
                    result['distance'] = 'close'
                else:
                    result['distance'] = 'moderate'
        
        # Check left region
        if left_std < texture_threshold and not result['detected']:
            result['detected'] = True
            result['direction'] = 'left'
            result['confidence'] = 0.6
            if left_std < 10:
                result['distance'] = 'close'
            else:
                result['distance'] = 'moderate'
        
        # Check right region
        if right_std < texture_threshold and not result['detected']:
            result['detected'] = True
            result['direction'] = 'right'
            result['confidence'] = 0.6
            if right_std < 10:
                result['distance'] = 'close'
            else:
                result['distance'] = 'moderate'
        
        # Additional check: Look for large uniform regions (walls)
        # Apply adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
        
        # Find large uniform regions
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_uniform_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0.1 * w * h:  # Large region (at least 10% of frame)
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                # Check if region is relatively uniform (low variance)
                region = gray[y:y+h_rect, x:x+w_rect]
                if region.size > 0:
                    region_std = np.std(region)
                    if region_std < texture_threshold:
                        large_uniform_regions.append({
                            'bbox': [x, y, x+w_rect, y+h_rect],
                            'std': region_std,
                            'center_x': x + w_rect/2
                        })
        
        # If we found large uniform regions, it's likely a wall
        if large_uniform_regions and not result['detected']:
            best_region = min(large_uniform_regions, key=lambda r: r['std'])
            result['detected'] = True
            result['confidence'] = 0.7
            
            center_x = best_region['center_x']
            if center_x < w * 0.33:
                result['direction'] = 'left'
            elif center_x > w * 0.67:
                result['direction'] = 'right'
            else:
                result['direction'] = 'ahead'
            
            if best_region['std'] < 8:
                result['distance'] = 'very close'
            elif best_region['std'] < 12:
                result['distance'] = 'close'
            else:
                result['distance'] = 'moderate'
        
    except Exception as e:
        print(f"[Wall Detection Error]: {e}")
    
    return result


def run_walking_mode():
    """Interactive Real-Time Walking Assistant for Blind Users."""

    # === Initialize Core Components ===
    engine = pyttsx3.init()
    recognizer = sr.Recognizer()
    command_queue = queue.Queue()
    stop_threads = threading.Event()

    # Use global speak() from utils.speech_utils

    # === Model Initialization ===
    try:
        speak("Activating walking mode. Please hold on while I load the detection model.")
        from transformers import DetrImageProcessor, DetrForObjectDetection
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        speak("Walking mode ready. Start walking slowly, I will guide you safely.")
    except Exception as e:
        speak("I could not load my vision model. Please check your internet connection.")
        print("Model load error:", e)
        return

    # === Listen to User in Background Thread ===
    def continuous_listen():
        """Continuously listen for commands and push to queue."""
        while not stop_threads.is_set():
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
                    text = recognizer.recognize_google(audio).lower()
                    command_queue.put(text)
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                print("[Speech Error]", e)
                continue

    listener_thread = threading.Thread(target=continuous_listen, daemon=True)
    listener_thread.start()

    # === Object Detection ===
    def detect_objects(frame):
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

            detected = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > 0.7:
                    detected.append(model.config.id2label[label.item()])
            return detected
        except Exception as e:
            print("[Detection Error]", e)
            return []

    # === Describe Scene ===
    def describe_scene(objects):
        if not objects:
            return "The way ahead looks clear."
        else:
            objects = list(set(objects))
            if len(objects) == 1:
                return f"I see a {objects[0]} ahead."
            else:
                return "I can see " + ", ".join(objects[:-1]) + " and " + objects[-1] + " ahead."

    # === Directional Guidance ===
    def give_guidance(objects):
        danger_objects = {"car", "bus", "truck", "bicycle", "person", "wall", "pole", "chair"}
        safe = True
        for obj in objects:
            if obj in danger_objects:
                safe = False
                speak(f"Warning! {obj} detected ahead. Please slow down.")
                return
        if safe:
            speak("Path is clear, you may continue straight.")

    # === Walking Guidance Loop ===
    def process_camera_feed():
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                speak("Camera not found. Please reconnect and try again.")
            return

            last_scene_time = 0
            last_detected_objects = []
            idle_time = 0
            last_door_announcement = 0
            last_wall_announcement = 0
            last_door_result = None
            last_wall_result = None
            last_orientation_check = 0
            last_orientation_announcement = 0
            frame_count = 0
            orientation = {'is_aligned': True, 'severity': 'none'}  # Initialize
            
            while not stop_threads.is_set():
                ret, frame = cap.read()
                if not ret:
                    idle_time += 1
                    if idle_time > 10:
                        speak("Camera feed lost. Please check the camera.")
                        break
                    continue

                idle_time = 0
                frame_count += 1
                
                # Check camera orientation every 2 seconds
                current_time = time.time()
                if current_time - last_orientation_check > 2.0:
                    orientation = check_camera_orientation(frame)
                    last_orientation_check = current_time
                    
                    if not orientation['is_aligned']:
                        # Provide audio feedback
                        if orientation['severity'] == 'major':
                            if current_time - last_orientation_announcement > 5.0:
                                speak(orientation['message'])
                                last_orientation_announcement = current_time
                        elif orientation['severity'] == 'minor':
                            if current_time - last_orientation_announcement > 8.0:
                                speak(orientation['message'])
                                last_orientation_announcement = current_time
                
                # Skip detection if camera is too tilted
                if orientation.get('severity') == 'major':
                    time.sleep(0.3)
                    continue
                
                detected = detect_objects(frame)
                
                # Detect doors and walls in real-time
                door_result = detect_door(frame)
                wall_result = detect_wall(frame)
                
                current_time = time.time()
                
                # Announce door detection (rate-limited to avoid spam)
                if door_result['detected'] and door_result['confidence'] > 0.5:
                    # Only announce if it's a new detection or significant change
                    if (last_door_result is None or 
                        last_door_result['position'] != door_result['position'] or
                        last_door_result['distance'] != door_result['distance'] or
                        current_time - last_door_announcement > 5):
                        
                        position_text = door_result['position']
                        distance_text = door_result['distance']
                        announcement = f"Door detected {distance_text} on your {position_text}."
                        speak(announcement)
                        last_door_announcement = current_time
                        last_door_result = door_result.copy()
                
                # Announce wall detection (rate-limited)
                if wall_result['detected'] and wall_result['confidence'] > 0.5:
                    # Only announce if it's a new detection or significant change
                    if (last_wall_result is None or
                        last_wall_result['direction'] != wall_result['direction'] or
                        last_wall_result['distance'] != wall_result['distance'] or
                        current_time - last_wall_announcement > 5):
                        
                        direction_text = wall_result['direction']
                        distance_text = wall_result['distance']
                        announcement = f"Wall detected {distance_text} {direction_text}."
                        speak(announcement)
                        last_wall_announcement = current_time
                        last_wall_result = wall_result.copy()

                # Dynamic scene updates
                if time.time() - last_scene_time > 8:
                    message = describe_scene(detected)
                    speak(message)
                    give_guidance(detected)
                    last_scene_time = time.time()
                    last_detected_objects = detected

                # Handle user voice commands dynamically
                while not command_queue.empty():
                    cmd = command_queue.get()
                    print(f"[USER SAID]: {cmd}")

                    if any(word in cmd for word in ["what", "see", "front", "ahead"]):
                        speak(describe_scene(last_detected_objects))
                        # Also mention doors and walls if detected
                        if last_door_result and last_door_result['detected']:
                            speak(f"Door detected {last_door_result['distance']} on your {last_door_result['position']}.")
                        if last_wall_result and last_wall_result['detected']:
                            speak(f"Wall detected {last_wall_result['distance']} {last_wall_result['direction']}.")
                    elif any(word in cmd for word in ["door", "doors"]):
                        # Check for door in current frame
                        ret_check, frame_check = cap.read()
                        if ret_check:
                            door_check = detect_door(frame_check)
                            if door_check['detected']:
                                speak(f"Door detected {door_check['distance']} on your {door_check['position']}.")
                            else:
                                speak("No door detected in the current view. Try turning slowly to scan for doors.")
                    elif any(word in cmd for word in ["wall", "walls"]):
                        # Check for wall in current frame
                        ret_check, frame_check = cap.read()
                        if ret_check:
                            wall_check = detect_wall(frame_check)
                            if wall_check['detected']:
                                speak(f"Wall detected {wall_check['distance']} {wall_check['direction']}.")
                            else:
                                speak("No wall detected in the current view.")
                    elif any(word in cmd for word in ["clear", "safe", "obstacle"]):
                        give_guidance(last_detected_objects)
                        # Warn about walls if very close
                        if last_wall_result and last_wall_result['detected'] and last_wall_result['distance'] in ['very close', 'close']:
                            speak(f"Caution: Wall is {last_wall_result['distance']} {last_wall_result['direction']}.")
                    elif any(word in cmd for word in ["left", "right", "turn"]):
                        speak("Turn slowly and I'll scan your surroundings.")
                    elif any(word in cmd for word in ["describe", "around", "surrounding"]):
                        speak(describe_scene(last_detected_objects))
                        # Include door and wall information
                        if last_door_result and last_door_result['detected']:
                            speak(f"Door detected {last_door_result['distance']} on your {last_door_result['position']}.")
                        if last_wall_result and last_wall_result['detected']:
                            speak(f"Wall detected {last_wall_result['distance']} {last_wall_result['direction']}.")
                    elif any(word in cmd for word in ["repeat", "again"]):
                        speak(describe_scene(last_detected_objects))
                        if last_door_result and last_door_result['detected']:
                            speak(f"Door detected {last_door_result['distance']} on your {last_door_result['position']}.")
                        if last_wall_result and last_wall_result['detected']:
                            speak(f"Wall detected {last_wall_result['distance']} {last_wall_result['direction']}.")
                    elif is_exit_command(cmd):
                        speak("Walking mode deactivated. Take care and walk safely.")
                        stop_threads.set()
                        break
                    else:
                        speak("Sorry, I didn't catch that. Please repeat.")

                if stop_threads.is_set():
                    break

            cap.release()
            cv2.destroyAllWindows()
            speak("Exiting walking mode.")
        except Exception as e:
            print("Walking mode error:", traceback.format_exc())
            speak("An error occurred while processing walking mode.")

    # === Start Guidance Loop ===
    process_camera_feed()
    stop_threads.set()
    listener_thread.join(timeout=1)

# Initialize Text-to-Speech
engine = pyttsx3.init()

# Initialize Hugging Face caption model (optional)
# This will be loaded in a separate thread during startup
processor = None
model_blip = None
blip_loading_thread = None

def load_blip_model_async():
    """Load BLIP model asynchronously with periodic feedback."""
    global processor, model_blip
    try:
        if BlipProcessor is not None and BlipForConditionalGeneration is not None:
            print("[BLIP] Loading BLIP model in background...")
            
            # Start feedback thread
            feedback_stop = threading.Event()
            feedback_thread = threading.Thread(
                target=provide_loading_feedback,
                args=(feedback_stop, "Loading vision models", "BLIP"),
                daemon=True
            )
            feedback_thread.start()
            
            # Load models
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # Stop feedback
            feedback_stop.set()
            feedback_thread.join(timeout=1)
            
            print("[BLIP] BLIP model loaded successfully.")
            speak("Vision models loaded successfully.")
    except Exception as e:
        print(f"[BLIP Load Warning]: {e}")

# Initialize EasyOCR for handwriting recognition (optional)
easyocr_reader = None
easyocr_loading_thread = None

def load_easyocr_async():
    """Load EasyOCR model asynchronously with periodic feedback."""
    global easyocr_reader
    if EASYOCR_AVAILABLE:
        try:
            print("[EasyOCR] Initializing EasyOCR reader (this may take a moment on first run)...")
            
            # Start feedback thread
            feedback_stop = threading.Event()
            feedback_thread = threading.Thread(
                target=provide_loading_feedback,
                args=(feedback_stop, "Loading handwriting recognition", "EasyOCR"),
                daemon=True
            )
            feedback_thread.start()
            
            # Load model
            easyocr_reader = easyocr.Reader(['en'], gpu=False)  # Use GPU if available
            
            # Stop feedback
            feedback_stop.set()
            feedback_thread.join(timeout=1)
            
            print("[EasyOCR] EasyOCR initialized successfully.")
            speak("Handwriting recognition model loaded.")
        except Exception as e:
            print(f"[EasyOCR Warning] Could not initialize EasyOCR: {e}")
            easyocr_reader = None

# Start loading models in background threads (will be triggered during system_boot)
battery_monitor_stop_event = None


def provide_loading_feedback(stop_event, model_name, model_type):
    """
    Provide periodic audio feedback (beeps and voice) while models are loading.
    Plays beep every 2 seconds with occasional voice updates.
    """
    feedback_count = 0
    messages = [
        f"Loading {model_name}. Please wait.",
        f"Still loading {model_name}.",
        f"{model_name} is loading. This may take a moment.",
        f"Almost done loading {model_name}.",
    ]
    
    # Initial voice message
    try:
        speak(f"Loading {model_name}. Please wait.")
    except Exception as e:
        print(f"[Speech Error during loading]: {e}")
    
    # Play first beep immediately
    try:
        winsound.Beep(600, 200)  # Lower pitch, shorter beep
    except Exception as e:
        print(f"[Beep Error]: {e}")
    
    feedback_count = 1
    
    while not stop_event.is_set():
        # Wait 2 seconds before next beep
        if stop_event.wait(2):
            break  # Stop event was set, exit loop
        
        # Play beep sound every 2 seconds
        try:
            winsound.Beep(600, 200)  # Lower pitch, shorter beep
        except Exception as e:
            print(f"[Beep Error]: {e}")
        
        feedback_count += 1
        
        # Every 3 beeps (6 seconds), provide a voice message
        if feedback_count % 3 == 0:
            message_index = min((feedback_count // 3) % len(messages), len(messages) - 1)
            try:
                speak(messages[message_index])
            except Exception as e:
                print(f"[Speech Error during loading]: {e}")

# Configure tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ------------------------
# Basic Helper Functions
# ------------------------

def listen_for_speech():
    """Replace with real mic input. For now, use console input."""
    return input("[You said]: ").lower().strip()

# ------------------------
# Reading Mode Core Logic
# ------------------------

engine = pyttsx3.init()

# ===============================
# ADVANCED IMAGE PREPROCESSING FOR TEXT DETECTION
# ===============================

def preprocess_image_for_ocr(frame):
    """
    Advanced image preprocessing to improve OCR accuracy.
    Applies denoising, contrast enhancement, deskewing, and binarization.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Apply adaptive thresholding for better text separation
    # Try multiple thresholding methods and pick the best
    thresh1 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    
    # Otsu's thresholding as fallback
    _, thresh3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up text
    kernel = np.ones((2, 2), np.uint8)
    thresh1_clean = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    thresh2_clean = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    
    # Return multiple preprocessed versions for different OCR engines
    return {
        'original': gray,
        'enhanced': enhanced,
        'adaptive_gaussian': thresh1_clean,
        'adaptive_mean': thresh2_clean,
        'otsu': thresh3
    }


def detect_text_regions(frame):
    """
    Detect regions in the image that likely contain text.
    Returns bounding boxes of text regions.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply preprocessing
    preprocessed = preprocess_image_for_ocr(frame)
    
    # Use morphological operations to find text regions
    # Horizontal kernel to connect text characters
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(preprocessed['adaptive_gaussian'], cv2.MORPH_CLOSE, horizontal_kernel)
    
    # Find contours
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions = []
    h, w = gray.shape
    
    for contour in contours:
        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filter by size (text regions should be reasonably sized)
        if area > 100 and w_rect > 20 and h_rect > 10:
            # Filter by aspect ratio (text is usually wider than tall)
            aspect_ratio = w_rect / float(h_rect) if h_rect > 0 else 0
            if 0.5 < aspect_ratio < 20:
                text_regions.append({
                    'x': x,
                    'y': y,
                    'w': w_rect,
                    'h': h_rect,
                    'area': area
                })
    
    # Sort by y-coordinate (top to bottom)
    text_regions.sort(key=lambda r: r['y'])
    
    return text_regions


def extract_text_with_tesseract(image, config='--oem 3 --psm 6'):
    """
    Extract text using Tesseract OCR with optimized configuration.
    Tries multiple PSM modes for better accuracy.
    """
    texts = []
    confidences = []
    
    # Try different PSM modes
    psm_modes = [
        ('--oem 3 --psm 6', 'uniform_block'),  # Assume uniform block of text
        ('--oem 3 --psm 11', 'sparse_text'),  # Sparse text
        ('--oem 3 --psm 7', 'single_line'),   # Single text line
        ('--oem 3 --psm 8', 'single_word'),   # Single word
        ('--oem 3 --psm 13', 'raw_line'),     # Raw line
    ]
    
    for psm_config, mode_name in psm_modes:
        try:
            data = pytesseract.image_to_data(image, config=psm_config, output_type=Output.DICT)
            
            # Extract text with confidence scores
            text_parts = []
            conf_scores = []
            for i, text in enumerate(data['text']):
                if len(text.strip()) > 0:
                    conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0
                    if conf > 30:  # Minimum confidence threshold
                        text_parts.append(text.strip())
                        conf_scores.append(conf)
            
            if text_parts:
                combined_text = ' '.join(text_parts)
                avg_conf = np.mean(conf_scores) if conf_scores else 0
                texts.append(combined_text)
                confidences.append(avg_conf)
        except Exception as e:
            print(f"[Tesseract Error - {mode_name}]: {e}")
            continue
    
    # Return the text with highest average confidence
    if texts and confidences:
        best_idx = np.argmax(confidences)
        return texts[best_idx], confidences[best_idx]
    
    return "", 0


def extract_text_with_easyocr(image):
    """
    Extract text using EasyOCR (better for handwriting.
    """
    if not EASYOCR_AVAILABLE or easyocr_reader is None:
        return "", 0
    
    try:
        # EasyOCR expects numpy array
        results = easyocr_reader.readtext(image)
        
        if not results:
            return "", 0
        
        # Combine all detected text
        texts = []
        confidences = []
        
        for (bbox, text, conf) in results:
            if conf > 0.3:  # Minimum confidence threshold
                texts.append(text.strip())
                confidences.append(conf)
        
        if texts:
            combined_text = ' '.join(texts)
            avg_conf = np.mean(confidences)
            return combined_text, avg_conf
        
        return "", 0
    except Exception as e:
        print(f"[EasyOCR Error]: {e}")
        return "", 0


def extract_text_combined(frame):
    """
    Extract text using multiple OCR engines and combine results.
    Returns the best result from Tesseract and EasyOCR.
    """
    preprocessed = preprocess_image_for_ocr(frame)
    
    all_results = []
    
    # Try Tesseract on different preprocessed versions
    for name, img in preprocessed.items():
        if name != 'original':  # Skip original, use enhanced versions
            text, conf = extract_text_with_tesseract(img)
            if text and conf > 40:
                all_results.append({
                    'text': text,
                    'confidence': conf,
                    'method': f'Tesseract-{name}'
                })
    
    # Try EasyOCR (better for handwriting)
    if EASYOCR_AVAILABLE and easyocr_reader is not None:
        # EasyOCR works better on original or enhanced image
        for name in ['enhanced', 'adaptive_gaussian', 'otsu']:
            text, conf = extract_text_with_easyocr(preprocessed[name])
            if text and conf > 0.3:
                all_results.append({
                    'text': text,
                    'confidence': conf * 100,  # Convert to 0-100 scale
                    'method': f'EasyOCR-{name}'
                })
    
    # Return the result with highest confidence
    if all_results:
        best_result = max(all_results, key=lambda x: x['confidence'])
        return best_result['text'], best_result['confidence'], best_result['method']
    
    return "", 0, "none"


def assess_image_quality(frame):
    """
    Assess image quality and provide feedback to user.
    Returns quality score and suggestions.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Check brightness
    mean_brightness = np.mean(gray)
    brightness_score = 1.0 - abs(mean_brightness - 127) / 127.0
    
    # Check contrast
    std_contrast = np.std(gray)
    contrast_score = min(std_contrast / 50.0, 1.0)
    
    # Check blur (using Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(laplacian_var / 100.0, 1.0)
    
    # Overall quality score
    quality_score = (brightness_score * 0.3 + contrast_score * 0.4 + blur_score * 0.3)
    
    suggestions = []
    if mean_brightness < 80:
        suggestions.append("Image is too dark. Move to better lighting.")
    elif mean_brightness > 180:
        suggestions.append("Image is too bright. Reduce lighting or move away from light source.")
    
    if std_contrast < 20:
        suggestions.append("Low contrast detected. Ensure text is clearly visible.")
    
    if laplacian_var < 50:
        suggestions.append("Image appears blurry. Hold camera steady and ensure focus.")
    
    return quality_score, suggestions


def run_reading_mode():
    """
    Enhanced Reading Mode — Continuously scans text from live camera feed and reads it aloud.
    Supports both printed text and handwriting recognition.
    Listens for voice commands in parallel until the user says 'exit', 'stop', or 'quit'.
    """
    speak("Reading Mode Activated. I will read visible text in real time, including handwriting.")
    speak("Say 'pause reading', 'resume', 'check quality', or 'exit' to stop.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not available. Please check the connection.")
        return
    
    running = True
    reading_enabled = True
    last_text = ""
    last_confidence = 0
    frame_count = 0
    quality_check_counter = 0
    consecutive_no_text = 0

    # --- Voice command listener in a separate thread ---
    def listen_for_commands():
        nonlocal running, reading_enabled
        recognizer = sr.Recognizer()
        while running:
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    # Use longer timeout and only listen when there's actual audio
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
                    user_input = recognizer.recognize_google(audio).lower().strip()
                    
                    if is_exit_command(user_input):
                        speak("Exiting Reading Mode.")
                        running = False
                        break

                    elif "pause" in user_input:
                        reading_enabled = False
                        speak("Reading paused.")

                    elif "resume" in user_input:
                        reading_enabled = True
                        speak("Resuming real-time reading.")

                    elif "describe" in user_input:
                        speak("Describing what I see.")
                        describe_document_scene()

                    elif "repeat" in user_input and last_text:
                        speak("Repeating last detected text.")
                        speak(last_text)

                    elif "quality" in user_input or "check" in user_input:
                        # Will be handled in main loop
                        pass
            except sr.WaitTimeoutError:
                # Timeout is normal - just continue listening
                continue
            except sr.UnknownValueError:
                # Could not understand audio - continue listening
                continue
            except sr.RequestError as e:
                print(f"[Speech Recognition Error]: {e}")
                # Wait a bit before retrying
                time.sleep(2)
                continue
            except Exception as e:
                print(f"[Listener Error]: {e}")
                time.sleep(1)
                continue

    # Start listening for voice commands in parallel
    listener_thread = threading.Thread(target=listen_for_commands, daemon=True)
    listener_thread.start()

    # --- Live OCR Loop ---
    while running:
        ret, frame = cap.read()
        if not ret:
            consecutive_no_text += 1
            if consecutive_no_text > 10:
                speak("Camera feed lost. Please check the connection.")
                break
            continue

        consecutive_no_text = 0
        frame_count += 1

        if reading_enabled:
            # Check image quality periodically
            if frame_count % 30 == 0:  # Every 30 frames (~15 seconds at 0.5s interval)
                try:
                    quality_score, suggestions = assess_image_quality(frame)
                    if quality_score < 0.5:
                        speak("Image quality is low. " + " ".join(suggestions[:2]))
                except Exception as e:
                    print(f"[Quality Check Error]: {e}")
            
            # Extract text using combined OCR methods
            try:
                detected_text, confidence, method = extract_text_combined(frame)
            except Exception as e:
                print(f"[OCR Error]: {e}")
                detected_text = ""
                confidence = 0
                method = "none"
            
            # Only announce if we have meaningful text with reasonable confidence
            if detected_text and detected_text.strip():
                # Check if text is significantly different from last text
                text_similarity = 0
                if last_text:
                    # Simple similarity check
                    words_new = set(detected_text.lower().split())
                    words_old = set(last_text.lower().split())
                    if words_old:
                        text_similarity = len(words_new & words_old) / len(words_old)
                
                # Announce if:
                # 1. New text with good confidence (>50)
                # 2. Significantly different text (similarity < 0.7)
                # 3. Much better confidence than last time
                should_announce = False
                
                if confidence > 50 and (text_similarity < 0.7 or not last_text):
                    should_announce = True
                elif confidence > last_confidence + 15 and confidence > 40:
                    should_announce = True
                elif not last_text and confidence > 30:  # First detection with low confidence
                    should_announce = True
                
                if should_announce:
                    # Clean up text
                    cleaned_text = ' '.join(detected_text.split())  # Remove extra whitespace
                    
                    if len(cleaned_text) > 3:  # Only announce if meaningful length
                        last_text = cleaned_text
                        last_confidence = confidence
                        
                        # Determine if it's likely handwriting (lower confidence, different patterns)
                        is_handwriting = "EasyOCR" in method or confidence < 60
                        
                        if is_handwriting:
                            speak(f"I see handwritten text: {cleaned_text}")
                        else:
                            speak(f"I see text: {cleaned_text}")
                        
                        # If confidence is low, mention it
                        if confidence < 50:
                            speak("Note: Confidence is moderate. Please ensure text is clearly visible.")
            
            # Provide feedback if no text detected for a while
            elif frame_count % 20 == 0 and not detected_text:
                consecutive_no_text += 1
                if consecutive_no_text >= 3:
                    speak("No text detected. Please ensure the document is well-lit and clearly visible in the camera.")
                    consecutive_no_text = 0
            else:
                consecutive_no_text = 0

        # Note: Removed cv2.imshow() and cv2.waitKey() as they cause issues on Windows
        # and are not needed for blind users (no visual display needed)
        # Exit is handled via voice commands only
        
        time.sleep(0.5)  # adjust scanning rate (every half second)

    # --- Cleanup ---
    try:
        cap.release()
    except Exception as e:
        print(f"[Camera Release Error]: {e}")
    
    # Note: Removed cv2.destroyAllWindows() as it's not needed and causes issues on Windows
    speak("Reading Mode deactivated.")


# ------------------------
# Subfunctions
# ------------------------

engine = pyttsx3.init()

def capture_and_read_text():
    """Capture a frame from camera and extract text + headings using improved OCR."""
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        speak("Camera error while capturing image.")
        return "", []

    # Use improved OCR extraction
    detected_text, confidence, method = extract_text_combined(frame)
    
    # If no text found with combined method, try traditional Tesseract with line structure
    if not detected_text or len(detected_text.strip()) < 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        preprocessed = preprocess_image_for_ocr(frame)
        enhanced = preprocessed['enhanced']
        
        try:
            data = pytesseract.image_to_data(enhanced, output_type=Output.DICT, config='--oem 3 --psm 6')
            
            # Preserve line breaks by grouping text by line number
            lines_dict = {}
            for i, text in enumerate(data['text']):
                if len(text.strip()) > 0:
                    conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0
                    if conf > 30:  # Minimum confidence
                        line_num = data['line_num'][i]
                        if line_num not in lines_dict:
                            lines_dict[line_num] = []
                        lines_dict[line_num].append(text)
            
            # Join words on same line with spaces, then join lines with newlines
            detected_text = '\n'.join([' '.join(lines_dict[line_num]) for line_num in sorted(lines_dict.keys())]).strip()
            
            # Fallback to original method if no line structure found
            if not detected_text:
                detected_text = " ".join([t for t in data['text'] if len(t.strip()) > 0]).strip()
        except Exception as e:
            print(f"[Tesseract Error in capture_and_read_text]: {e}")
    
    # Extract headings (uppercase text with high confidence)
    headings = []
    if detected_text:
        # Try to identify headings from the text
        lines = detected_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and line.isupper() and len(line) > 2 and len(line) < 50:
                headings.append(line)
    
    # Also try Tesseract data extraction for headings if available
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        preprocessed = preprocess_image_for_ocr(frame)
        enhanced = preprocessed['enhanced']
        data = pytesseract.image_to_data(enhanced, output_type=Output.DICT, config='--oem 3 --psm 6')
        
        for i, text in enumerate(data['text']):
            if len(text.strip()) > 0:
                conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0
                if conf > 70 and text.isupper() and text not in headings:
                    headings.append(text)
    except:
        pass

    if headings:
        speak(f"I detected {len(headings)} headings: {', '.join(headings[:5])}.")
    elif detected_text:
        speak("Text detected, but no clear headings found.")
    else:
        speak("No text detected. Please ensure the document is clearly visible and well-lit.")

    return detected_text, headings

engine = pyttsx3.init()

def describe_document_scene():
    """Use BLIP model to describe what’s in front of the camera."""
    if processor is None or model_blip is None:
        speak("Vision caption model not available right now.")
        return
    speak("Analyzing your document surroundings...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        speak("Unable to access camera for scene analysis.")
        return

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    try:
        inputs = processor(img, return_tensors="pt")
        out = model_blip.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        speak(f"It looks like {caption}.")
    except Exception as _cap_err:
        print("[Caption Error]", _cap_err)
        speak("Sorry, I couldn't analyze the scene right now.")

engine = pyttsx3.init()

def read_specific_heading(user_input, headings):
    """Read text corresponding to a specific heading."""
    try:
        words = user_input.split()
        for word in words:
            if word.isdigit():
                idx = int(word) - 1
                if 0 <= idx < len(headings):
                    speak(f"Reading text under heading {headings[idx]}.")
                    # In a real OCR system, you could extract specific region around that heading.
                    speak(f"Heading: {headings[idx]}")
                    return
        speak("I couldn't find that heading number.")
    except Exception as e:
        print("[Error reading heading]", e)
        speak("Something went wrong while reading the heading.")

engine = pyttsx3.init()

def navigate_headings(headings, direction="next"):
    """Simple heading navigation placeholder."""
    if not headings:
        speak("No headings detected to navigate.")
        return
    if direction == "next":
        speak(f"Next heading is {headings[min(1, len(headings)-1)]}.")
    else:
        speak(f"Previous heading is {headings[0]}.")

engine = pyttsx3.init()

# -------------------------
# Basic system functions
# -------------------------
# Note: speak() and listen_for_speech() are imported from utils.speech_utils
# Don't redefine them here or they will shadow the import

# -------------------------
# Core object detection logic
# -------------------------

engine = pyttsx3.init()

def run_object_detection_mode():
    speak("Object Detection Mode Activated. You can ask me to find any object around you.")
    speak("Say, for example, 'Find a bottle', or 'Describe the scene'.")

    last_detection = None
    cap = None

    while True:
        user_input = listen_for_speech().lower().strip()

        if not user_input:
            continue

        # Exit condition - run continuously until exit is said
        if is_exit_command(user_input):
            speak("Exiting Object Detection Mode and returning to the main system.")
            if cap:
                cap.release()
            break

        # User asks to describe surroundings
        elif "describe" in user_input or "what do you see" in user_input:
            describe_scene()
            continue

        # User asks to repeat
        elif "repeat" in user_input and last_detection:
            speak(f"Repeating last detection: {last_detection}")
            continue

        # Detect specific object
        elif "find" in user_input or "detect" in user_input or "locate" in user_input:
            object_name = extract_object_name(user_input)
            if not object_name:
                speak("I didn’t catch the object name. Please say it again.")
                continue
            last_detection = detect_object(object_name)
            continue

        # Retry search
        elif "search again" in user_input or "try again" in user_input:
            if not last_detection:
                speak("You haven’t searched for anything yet. Please say an object name.")
            else:
                last_detection = detect_object(last_detection)
            continue

        else:
            speak(f"I heard '{user_input}', but I’m not sure what to do. Try saying 'Find bottle' or 'Describe scene'.")


# -------------------------
# Helper functions
# -------------------------
def extract_object_name(command):
    keywords = ["find", "detect", "locate", "search for", "show me", "look for"]
    for key in keywords:
        if key in command:
            name = command.replace(key, "").strip()
            return name
    return None


def detect_object(object_name):
    speak(f"Searching for {object_name}. Please hold still while I scan your surroundings.")
    cap = cv2.VideoCapture(0)
    found_local = False
    similar_objects = set()

    start_time = time.time()
    rotation_prompts = [
        "Please turn a little to your left.",
        "Now turn slightly to your right.",
        "Try facing behind you.",
        "Look a bit upwards or downwards."
    ]
    rotation_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = local_model(frame, verbose=False)
        detections = results[0].boxes
        labels = [local_model.names[int(cls)] for cls in detections.cls]

        # Check for requested object
        if object_name.lower() in [l.lower() for l in labels]:
            found_local = True
            speak(f"I found a {object_name} nearby.")
            break
        else:
            similar_objects.update(labels)

        # Rotate suggestion if not found
        if time.time() - start_time > 4 and rotation_index < len(rotation_prompts):
            speak(rotation_prompts[rotation_index])
            rotation_index += 1
            start_time = time.time()

        # After full rotation
        if rotation_index >= len(rotation_prompts):
            break

    cap.release()

    # If not found locally, use online Hugging Face
    if not found_local:
        speak(f"I couldn’t find {object_name} nearby. Let me check online vision.")
        caption = run_online_detection()

        if object_name.lower() in caption.lower():
            speak(f"I think I found a {object_name}. {caption}.")
        else:
            speak(f"Sorry, I couldn’t find any {object_name}. {caption}.")

        if similar_objects:
            speak(f"I did notice some similar objects like {', '.join(similar_objects)} around you.")
        else:
            speak("No other similar objects detected nearby.")

        return object_name

    else:
        return object_name


def run_online_detection():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        speak("Camera error while capturing frame.")
        return ""

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(img, return_tensors="pt")
    out = model_blip.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"[Online Caption]: {caption}")
    return caption


def describe_scene():
    """
    Enhanced scene description with conversational flow matching the examples.
    Captures scenes from multiple directions with natural turn-by-turn guidance.
    """
    global user_name
    
    # First, determine if indoor or outdoor with a quick scan
    speak(f"Sure, {user_name}. Let me analyze your surroundings.")
    
    # Quick initial scan to determine environment
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, initial_frame = cap.read()
        if ret:
            try:
                results = local_model(initial_frame, verbose=False)
                initial_detections = []
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        if conf > 0.5:
                            obj_name = local_model.names[cls]
                            initial_detections.append(obj_name)
                
                # Determine environment
                outdoor_indicators = ["car", "bus", "truck", "traffic light", "road", "sidewalk"]
                indoor_indicators = ["door", "window", "table", "chair", "computer", "bed", "sofa", "tvmonitor"]
                is_outdoor = any(obj in outdoor_indicators for obj in initial_detections)
                is_indoor = any(obj in indoor_indicators for obj in initial_detections)
            except:
                is_indoor = True  # Default to indoor
                is_outdoor = False
        else:
            is_indoor = True
            is_outdoor = False
        cap.release()
    else:
        is_indoor = True
        is_outdoor = False
    
    # Indoor flow: left → center → right (and optionally back)
    if is_indoor and not is_outdoor:
        speak(f"To help me describe everything around you, please slowly turn to your left until I say stop.")
        
        # Capture while user turns left
        left_captured = False
        start_time = time.time()
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            while time.time() - start_time < 5:  # Give user time to turn
                ret, frame = cap.read()
                if ret:
                    # Capture multiple frames while turning
                    time.sleep(0.5)
                    if not left_captured:
                        left_caption, left_objects, left_traffic, left_road = capture_and_analyze_direction("left")
                        left_captured = True
                        break
            cap.release()
        
        speak("Stop. Great, now please turn a little to your right, back to your center position.")
        time.sleep(3)  # Give time to turn
        
        # Capture center/front
        front_caption, front_objects, front_traffic, front_road = capture_and_analyze_direction("front")
        
        speak("Perfect. Now turn slightly more to your right, until I say stop.")
        
        # Capture while user turns right
        right_captured = False
        start_time = time.time()
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            while time.time() - start_time < 5:
                ret, frame = cap.read()
                if ret:
                    time.sleep(0.5)
                    if not right_captured:
                        right_caption, right_objects, right_traffic, right_road = capture_and_analyze_direction("right")
                        right_captured = True
                        break
            cap.release()
        
        speak(f"Stop. Thank you, {user_name}. I've analyzed the surroundings. Here's what I see:")
        
        # Capture back view (optional)
        speak("Let me also check behind you.")
        back_caption, back_objects, back_traffic, back_road = capture_and_analyze_direction("back")
        
    else:  # Outdoor flow: panoramic turn
        speak(f"Please slowly turn around so I can capture your surroundings in all directions.")
        
        # Capture front
        front_caption, front_objects, front_traffic, front_road = capture_and_analyze_direction("front")
        time.sleep(1)
        
        # Capture left
        speak("Keep turning slowly...")
        left_caption, left_objects, left_traffic, left_road = capture_and_analyze_direction("left")
        time.sleep(1)
        
        # Capture back
        speak("Continue turning...")
        back_caption, back_objects, back_traffic, back_road = capture_and_analyze_direction("back")
        time.sleep(1)
        
        # Capture right
        speak("Almost done...")
        right_caption, right_objects, right_traffic, right_road = capture_and_analyze_direction("right")
        
        speak(f"Perfect, I've got a full view. Here's what I see:")
    
    # Build comprehensive description
    description = build_comprehensive_description(
        front_caption, front_objects, front_traffic, front_road,
        left_caption, left_objects, left_traffic, left_road,
        right_caption, right_objects, right_traffic, right_road,
        back_caption, back_objects, back_traffic, back_road
    )
    
    speak(description)
    
    # Ask follow-up questions
    if is_indoor and not is_outdoor:
        speak("Would you like me to describe specific objects in more detail — for example, what's on the table or near the door?")
    else:
        # Check if traffic signal was detected
        if front_traffic == "red":
            speak("When the signal turns green, I'll guide you safely across. Would you like me to keep monitoring traffic for you?")
        else:
            speak("Would you like me to keep monitoring your surroundings or help you navigate somewhere?")


def capture_and_analyze_direction(direction):
    """
    Capture a frame from the camera and analyze it using both object detection and image captioning.
    Returns: (caption, objects_list)
    """
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak(f"Camera error while capturing {direction} view.")
            return "", [], None, None
        
        # Capture frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            speak(f"Could not capture {direction} view.")
            return "", [], None, None
        
        # Convert to PIL Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Get image caption using BLIP
        caption = ""
        if processor is not None and model_blip is not None:
            try:
                inputs = processor(img, return_tensors="pt")
                out = model_blip.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                print(f"[{direction.capitalize()} Caption]: {caption}")
            except Exception as e:
                print(f"[Caption Error for {direction}]: {e}")
        
        # Get object detection using YOLO
        objects = []
        try:
            results = local_model(frame, verbose=False)
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf > 0.5:  # Confidence threshold
                        obj_name = local_model.names[cls]
                        objects.append(obj_name)
            objects = list(set(objects))  # Remove duplicates
            print(f"[{direction.capitalize()} Objects]: {objects}")
        except Exception as e:
            print(f"[Object Detection Error for {direction}]: {e}")
        
        # Analyze for traffic signals, road conditions, etc.
        traffic_info = analyze_traffic_signals(frame)
        road_info = analyze_road_conditions(frame, objects)
        
        return caption, objects, traffic_info, road_info
        
    except Exception as e:
        print(f"[Error capturing {direction}]: {e}")
        return "", [], None, None


def analyze_traffic_signals(frame):
    """Analyze frame for traffic signals (red, green, yellow lights)."""
    try:
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for traffic lights
        # Red range
        red_lower = np.array([0, 50, 50])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        red_pixels = cv2.countNonZero(red_mask)
        
        # Green range
        green_lower = np.array([50, 50, 50])
        green_upper = np.array([70, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_pixels = cv2.countNonZero(green_mask)
        
        # Yellow range
        yellow_lower = np.array([20, 50, 50])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        
        # Determine dominant color
        total = red_pixels + green_pixels + yellow_pixels
        if total > 100:  # Threshold to avoid noise
            if red_pixels > green_pixels and red_pixels > yellow_pixels:
                return "red"
            elif green_pixels > red_pixels and green_pixels > yellow_pixels:
                return "green"
            elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
                return "yellow"
        
        return None
    except Exception as e:
        print(f"[Traffic Signal Analysis Error]: {e}")
        return None


def analyze_road_conditions(frame, objects):
    """Analyze road conditions based on detected objects and scene."""
    try:
        # Count vehicles and people
        vehicle_count = sum(1 for obj in objects if obj in ["car", "bus", "truck", "motorcycle", "bicycle"])
        person_count = sum(1 for obj in objects if obj == "person")
        
        # Determine road condition
        if vehicle_count == 0 and person_count == 0:
            return "empty"
        elif vehicle_count <= 2 and person_count <= 3:
            return "little busy"
        elif vehicle_count <= 5 and person_count <= 8:
            return "moderately busy"
        else:
            return "too crowded"
    except Exception as e:
        print(f"[Road Condition Analysis Error]: {e}")
        return None


def load_depth_estimator():
    """Lazy-load MiDaS depth model for edge and drop-off detection."""
    global depth_model, depth_transform, depth_device
    if depth_model is not None and depth_transform is not None:
        return depth_model, depth_transform, depth_device

    try:
        depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        depth_transform = transforms.small_transform
        depth_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        depth_model.to(depth_device)
        depth_model.eval()
        print("[Depth] MiDaS_small model loaded for navigation analysis.")
    except Exception as e:
        print("[Depth Model Warning]", e)
        depth_model = None
        depth_transform = None
        depth_device = None

    return depth_model, depth_transform, depth_device


def analyze_depth_for_navigation(frame):
    """Return safety warnings and texture status using depth analysis."""
    model, transform, device = load_depth_estimator()
    if model is None or transform is None:
        return [], False

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    h, w = depth_map.shape
    messages = []

    # Regions
    middle = depth_map[int(h * 0.4): int(h * 0.65), int(w * 0.25): int(w * 0.75)]
    bottom = depth_map[int(h * 0.75):, int(w * 0.25): int(w * 0.75)]
    left_bottom = depth_map[int(h * 0.65):, : int(w * 0.3)]
    right_bottom = depth_map[int(h * 0.65):, int(w * 0.7):]
    top_region = depth_map[: int(h * 0.2), int(w * 0.25): int(w * 0.75)]

    mid_mean = np.mean(middle)
    bottom_mean = np.mean(bottom)
    left_mean = np.mean(left_bottom)
    right_mean = np.mean(right_bottom)
    top_mean = np.mean(top_region)

    # Depth heuristics (MiDaS: higher value ≈ closer object)
    drop_threshold = 0.18
    close_threshold = 0.6

    if bottom_mean < max(mid_mean - 0.18, drop_threshold):
        messages.append("Caution, it looks like the floor dips or drops ahead. Please stop and scan carefully before moving.")

    if left_mean < max(mid_mean - 0.2, drop_threshold) and mid_mean > close_threshold:
        messages.append("There's a low edge on your left. Shift a little to the right.")

    if right_mean < max(mid_mean - 0.2, drop_threshold) and mid_mean > close_threshold:
        messages.append("There's a low edge on your right. Adjust slightly to the left.")

    # Large flat obstacle / wall detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    center_gray = gray[:, int(w * 0.25): int(w * 0.75)]
    texture_std = float(np.std(center_gray))
    low_texture_flag = texture_std < 7.5

    if texture_std < 12 and mid_mean > close_threshold:
        messages.append("Flat surface detected ahead. It may be a wall or door. Approach slowly and feel for a handle.")

    # Sky / open space detection
    if top_mean < max(mid_mean - 0.25, drop_threshold) and mid_mean > close_threshold:
        messages.append("There's open space ahead. Make sure you're not approaching a balcony or roof edge.")

    # Remove duplicate messages
    unique_messages = []
    for msg in messages:
        if msg not in unique_messages:
            unique_messages.append(msg)

    return unique_messages, low_texture_flag

def build_comprehensive_description(front_caption, front_objects, front_traffic, front_road,
                                     left_caption, left_objects, left_traffic, left_road,
                                     right_caption, right_objects, right_traffic, right_road,
                                     back_caption, back_objects, back_traffic, back_road):
    """
    Build a comprehensive natural language description from all captured directions.
    """
    description_parts = []
    
    # Determine if indoor or outdoor
    all_objects = front_objects + left_objects + right_objects + back_objects
    outdoor_indicators = ["car", "bus", "truck", "traffic light", "road", "sidewalk", "street"]
    indoor_indicators = ["door", "window", "table", "chair", "computer", "bed", "room"]
    
    is_outdoor = any(obj in outdoor_indicators for obj in all_objects)
    is_indoor = any(obj in indoor_indicators for obj in all_objects) or not is_outdoor
    
    if is_indoor:
        # Determine room size based on objects detected
        total_objects = len(all_objects)
        if total_objects > 8:
            room_size = "large"
        elif total_objects > 4:
            room_size = "medium-sized"
        else:
            room_size = "small"
        
        description_parts.append(f"You are in a {room_size} room.")
        
        # Left description - prioritize doors and windows
        left_items_list = []
        door_on_left = False
        window_on_left = False
        
        if left_objects:
            for obj in left_objects:
                if obj == "door":
                    door_on_left = True
                elif obj == "window":
                    window_on_left = True
                else:
                    left_items_list.append(obj)
        
        if door_on_left:
            description_parts.append("There's a closed door on your left.")
        if window_on_left:
            description_parts.append("There's a window on your left letting in some light.")
        if left_items_list:
            left_items = ", ".join(left_items_list[:3])
            description_parts.append(f"On your left, there's {left_items}.")
        
        # Right description - prioritize windows
        right_items_list = []
        door_on_right = False
        window_on_right = False
        
        if right_objects:
            for obj in right_objects:
                if obj == "door":
                    door_on_right = True
                elif obj == "window":
                    window_on_right = True
                else:
                    right_items_list.append(obj)
        
        if window_on_right:
            description_parts.append("There's a window on your right letting in some light.")
        if door_on_right:
            description_parts.append("There's a door on your right.")
        if right_items_list:
            right_items = ", ".join(right_items_list[:3])
            description_parts.append(f"On your right, there's {right_items}.")
        
        # Front description - prioritize table and computer
        front_items_list = []
        has_table = False
        has_computer = False
        
        if front_objects:
            for obj in front_objects:
                if obj in ["table", "diningtable"]:
                    has_table = True
                elif obj in ["laptop", "computer", "tvmonitor"]:
                    has_computer = True
                else:
                    front_items_list.append(obj)
        
        if has_table and has_computer:
            description_parts.append("There's a table with a computer in front of you.")
        elif has_table:
            description_parts.append("There's a table in front of you.")
            if front_items_list:
                front_items = ", ".join(front_items_list[:2])
                description_parts.append(f"On the table, there's {front_items}.")
        elif front_objects:
            front_items = ", ".join(front_objects[:3])
            description_parts.append(f"In front of you, there's {front_items}.")
        
        # Back description
        if back_objects:
            if "person" in back_objects:
                person_count = back_objects.count("person")
                if person_count == 1:
                    description_parts.append("Behind you, there's a person sitting on a chair.")
                else:
                    description_parts.append(f"Behind you, there are {person_count} people.")
            if "chair" in back_objects and "person" not in back_objects:
                description_parts.append("Behind you, there's a chair.")
            if "bookshelf" in back_objects or "shelf" in back_objects:
                description_parts.append("There's a bookshelf against the wall.")
            back_other = [obj for obj in back_objects if obj not in ["person", "chair", "bookshelf", "shelf"]]
            if back_other:
                back_items = ", ".join(back_other[:2])
                description_parts.append(f"Behind you, there's also {back_items}.")
    
    else:  # Outdoor
        # Determine position (footpath, road, etc.)
        if "sidewalk" in all_objects or "footpath" in all_objects:
            description_parts.append("You're standing on a footpath.")
        elif "road" in all_objects or "street" in all_objects:
            description_parts.append("You're standing near a road.")
        else:
            description_parts.append("You're standing outdoors.")
        
        # Road direction and conditions
        road_direction = None
        road_condition = None
        
        if left_objects and any(obj in ["car", "bus", "truck", "road"] for obj in left_objects):
            road_direction = "left"
            road_condition = left_road if left_road else front_road
        elif right_objects and any(obj in ["car", "bus", "truck", "road"] for obj in right_objects):
            road_direction = "right"
            road_condition = right_road if right_road else front_road
        else:
            road_condition = front_road if front_road else (left_road or right_road)
        
        if road_direction:
            if road_condition:
                if road_condition == "empty":
                    description_parts.append(f"The road is on your {road_direction}, and it looks completely empty.")
                elif road_condition == "little busy":
                    description_parts.append(f"The road is on your {road_direction}, and it looks a little busy — a few cars are passing by.")
                elif road_condition == "moderately busy":
                    description_parts.append(f"The road is on your {road_direction}, and it looks moderately busy — several cars are passing by.")
                else:
                    description_parts.append(f"The road is on your {road_direction}, and it looks too crowded.")
            else:
                description_parts.append(f"The road is on your {road_direction}.")
        elif road_condition:
            if road_condition == "empty":
                description_parts.append("The road looks completely empty.")
            elif road_condition == "little busy":
                description_parts.append("The road looks a little busy — a few cars are passing by.")
            elif road_condition == "moderately busy":
                description_parts.append("The road looks moderately busy — several cars are passing by.")
            else:
                description_parts.append("The road looks too crowded.")
        
        # Traffic signals
        if front_traffic:
            signal_status = "red" if front_traffic == "red" else "green" if front_traffic == "green" else "yellow"
            if signal_status == "red":
                description_parts.append("There's a traffic signal in front of you, currently red, so please wait before crossing.")
            elif signal_status == "green":
                description_parts.append("There's a traffic signal in front of you, currently green, so you can proceed.")
            else:
                description_parts.append("There's a traffic signal in front of you, currently yellow, so please be cautious.")
        
        # Objects around - natural language
        if right_objects:
            right_items = [obj for obj in right_objects if obj not in ["car", "bus", "truck", "road"]]
            if right_items:
                right_desc = ", ".join(right_items[:2])
                description_parts.append(f"To your right, there's {right_desc}.")
        
        if back_objects:
            back_items = [obj for obj in back_objects if obj not in ["car", "bus", "truck", "road"]]
            if back_items:
                back_desc = ", ".join(back_items[:2])
                if "tree" in back_items:
                    description_parts.append(f"There's a tree just behind you providing some shade.")
                else:
                    description_parts.append(f"Behind you, there's {back_desc}.")
    
    # Combine all parts
    final_description = " ".join(description_parts)
    return final_description if final_description else "I've analyzed your surroundings, but couldn't detect specific details."

# You can add APIs later
# import requests
# import pyowm   # For weather
# import playsound  # For music

reminders = []

# Global user name storage
user_name = "friend"  # Default name, will be updated during system boot


def speak_available_modes_overview():
    """Speak the list of available modes in a friendly tone."""
    speak(
        "You can choose from assistant mode, navigation mode, object detection mode, walking mode, emergency mode, reading mode, security mode, medical mode, fire mode, and police mode."
    )


def explain_assistant_capabilities():
    """Explain what the assistant can help with."""
    speak(
        "I can tell you the time or date, check the weather, set reminders, play music, describe the scene through the camera, guide you indoors or outdoors, read documents, and help during emergencies. Just let me know what you need."
    )


def prompt_navigation_support():
    """Offer to start smart camera navigation."""
    speak("I'll turn on the camera and guide you step by step. Ready?")
    response = listen_for_speech(timeout=6, phrase_time_limit=4)
    if response and any(word in response.lower() for word in ["yes", "ready", "ok", "okay", "sure", "start"]):
        start_smart_camera_navigation()
    else:
        speak("No problem. If you change your mind, just say 'guide me out' or 'start navigation'.")


def handle_smalltalk(command: str) -> bool:
    """Handle simple conversational phrases. Returns True if handled."""
    if any(phrase in command for phrase in ["it seems good", "that's good", "sounds good", "great"]):
        speak("I'm glad to hear that! Let me know if you need anything else.")
        return True
    if command.strip() in {"thanks", "thank you", "thankyou"}:
        speak("You're welcome. I'm here whenever you need me.")
        return True
    if command.startswith("tell me") and "joke" not in command and "story" not in command:
        speak("Sure. Tell me what you’d like to hear about—time, weather, your surroundings, or navigation help?")
        return True
    if "repeat" in command and "modes" in command:
        speak_available_modes_overview()
        return True
    if "i'm good" in command or "im good" in command:
        speak("Great! If you need anything, just ask.")
        return True
    return False


def get_time():
    now = datetime.now()
    return now.strftime("%I:%M %p")


def get_date():
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y")


def get_period():
    hour = datetime.now().hour
    if hour < 12:
        return "morning"
    elif hour < 18:
        return "afternoon"
    else:
        return "evening"


def set_alarm():
    """
    Voice-driven, robust alarm setter.
    - Asks the user for date/time (in any spoken order).
    - Understands flexible natural language like 'tomorrow 6 am' or 'June fifth evening'.
    - Confirms before setting.
    - Rings by speaking aloud.
    - Exits only after user confirmation.
    """

    CONFIRM_YES = {"yes","yep","yeah","yup","correct","right","confirm","confirmed",
                   "sure","okay","ok","that is correct","that's right","done","set",
                   "affirmative","please do","please","sounds good","looks good"}
    CONFIRM_NO = {"no","nope","not","change","edit","cancel","don't","dont","wrong",
                  "not yet","try again","again","nevermind","never mind","stop"}

    engine = pyttsx3.init()
    recognizer = sr.Recognizer()

    def speak(text):
        print("SPEAK:", text)
        engine.say(text)
        engine.runAndWait()

    def listen(prompt=None):
        if prompt:
            speak(prompt)
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("Heard:", text)
            return text
        except sr.UnknownValueError:
            speak("Sorry, I didn’t catch that. Could you repeat?")
            return listen()
        except sr.RequestError:
            speak("Sorry, my speech recognition service is unavailable.")
            return None

    try:
        import dateparser
        HAS_DATEPARSER = True
    except ImportError:
        HAS_DATEPARSER = False
        speak("For better understanding of natural dates, please install the dateparser library later.")

    def parse_datetime(text):
        now = datetime.now()
        if not text:
            return None
        text = text.lower().strip()
        if HAS_DATEPARSER:
            parsed = dateparser.parse(text, settings={'PREFER_DATES_FROM': 'future'})
            if parsed and parsed < now:
                parsed += timedelta(days=1)
            return parsed
        # fallback
        m = re.search(r"(\d{1,2})(:\d{2})?\s*(am|pm)?", text)
        if not m:
            return None
        hour = int(m.group(1))
        minute = int(m.group(2)[1:]) if m.group(2) else 0
        if "pm" in text and hour < 12:
            hour += 12
        if "am" in text and hour == 12:
            hour = 0
        dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if "tomorrow" in text or dt < now:
            dt += timedelta(days=1)
        return dt

    def set_alarm(alarm_dt):
        delay = (alarm_dt - datetime.now()).total_seconds()
        if delay < 0:
            delay = 0.1

        def ring():
            for _ in range(3):
                speak("Your alarm time is here. Wake up!")
                time.sleep(1)

        t = threading.Timer(delay, ring)
        t.start()

    # Conversation
    speak("Let's set your alarm. You can say the date and time together or separately.")
    while True:
        spoken = listen("Please tell me the date and time for the alarm.")
        if not spoken:
            speak("I didn’t hear anything. Please say the alarm time again.")
            continue

        alarm_dt = parse_datetime(spoken)
        if not alarm_dt:
            speak("I couldn’t understand that time. Please say it again, like 'tomorrow at 6 a.m.'")
            continue

        human = alarm_dt.strftime("%I:%M %p on %A, %B %d, %Y")
        speak(f"I understood {human}. Is that correct?")
        confirm = listen()
        if confirm and any(word in confirm.lower() for word in CONFIRM_YES):
            schedule_alarm(alarm_dt)
            speak(f"Your alarm is confirmed and set for {human}.")
            speak("Thank you for using the alarm service. Have a great day!")
            return
        elif confirm and any(word in confirm.lower() for word in CONFIRM_NO):
            speak("Okay, please tell me the correct date and time.")
            continue
        else:
            speak("I didn’t catch that. Do you want to confirm or change it?")


def set_reminder():
    """
    Voice-driven reminder setter.
    - Understands natural phrases like 'remind me to take medicine tomorrow morning'
      or 'meeting in evening' or 'remind me in 10 minutes'.
    - Infers AM/PM based on words: morning = AM, afternoon/evening/night = PM.
    - Handles confirmation and conversation smoothly.
    - Speaks everything aloud for accessibility.
    - Ends with a thank-you message.
    """

    CONFIRM_YES = {"yes","yep","yeah","yup","correct","right","confirm","confirmed",
                   "sure","okay","ok","that is correct","that's right","done","set",
                   "affirmative","please do","please","sounds good","looks good"}
    CONFIRM_NO = {"no","nope","not","change","edit","cancel","don't","dont","wrong",
                  "not yet","try again","again","nevermind","never mind","stop"}

    engine = pyttsx3.init()
    recognizer = sr.Recognizer()

    def speak(text):
        print("SPEAK:", text)
        engine.say(text)
        engine.runAndWait()

    def listen(prompt=None):
        if prompt:
            speak(prompt) 
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("Heard:", text)
            return text
        except sr.UnknownValueError:
            speak("Sorry, I didn’t catch that. Please say it again.")
            return listen()
        except sr.RequestError:
            speak("Sorry, my speech service is unavailable right now.")
            return None

    # Try to use dateparser for natural language date/time parsing
    try:
        import dateparser
        HAS_DATEPARSER = True
    except ImportError:
        HAS_DATEPARSER = False
        speak("For more natural time understanding, you can install the dateparser library later.")

    def infer_time_of_day(text):
        """Infer AM/PM or default hours from words like 'morning', 'afternoon', etc."""
        text = text.lower()
        if "morning" in text:
            return 8  # 8 AM default
        elif "noon" in text:
            return 12  # 12 PM
        elif "afternoon" in text:
            return 15  # 3 PM
        elif "evening" in text:
            return 19  # 7 PM
        elif "night" in text:
            return 21  # 9 PM
        return None

    def parse_datetime(text):
        """Extract datetime with contextual AM/PM understanding."""
        now = datetime.now()
        if not text:
            return None
        text = text.lower().strip()

        inferred_hour = infer_time_of_day(text)

        if HAS_DATEPARSER:
            parsed = dateparser.parse(text, settings={'PREFER_DATES_FROM': 'future'})
            if parsed:
                if inferred_hour is not None:
                    parsed = parsed.replace(hour=inferred_hour, minute=0, second=0, microsecond=0)
                if parsed < now:
                    parsed += timedelta(days=1)
                return parsed

        # Fallback manual parsing
        match = re.search(r"(\d{1,2})(:\d{2})?\s*(am|pm)?", text)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)[1:]) if match.group(2) else 0
            period = match.group(3)
            if not period:
                # Infer based on context words
                if inferred_hour is not None:
                    hour = inferred_hour
                elif hour < 8:
                    period = "am"
                elif hour >= 8 and hour < 12:
                    period = "am"
                else:
                    period = "pm"
            if period == "pm" and hour < 12:
                hour += 12
            if period == "am" and hour == 12:
                hour = 0
            dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if "tomorrow" in text or dt < now:
                dt += timedelta(days=1)
            return dt

        # Handle relative phrases like “in 10 minutes” or “in 2 hours”
        rel = re.search(r"in\s+(\d+)\s*(minute|minutes|min|hour|hours|hr|hrs)", text)
        if rel:
            n = int(rel.group(1))
            unit = rel.group(2)
            if "hour" in unit:
                return now + timedelta(hours=n)
            else:
                return now + timedelta(minutes=n)

        # Only context like “morning” or “evening”
        if inferred_hour is not None:
            dt = now.replace(hour=inferred_hour, minute=0, second=0, microsecond=0)
            if "tomorrow" in text or dt < now:
                dt += timedelta(days=1)
            return dt

        return None

    def schedule_reminder(reminder_dt, message):
        delay = (reminder_dt - datetime.now()).total_seconds()
        if delay < 0:
            delay = 0.1

        def remind():
            for _ in range(3):
                speak(f"This is your reminder: {message}")
                time.sleep(1)

        t = threading.Timer(delay, remind)
        t.start()

    # ---------------------------
    # Conversational interaction
    # ---------------------------
    speak("Let's set a reminder. You can say things like 'Remind me to take medicine tomorrow morning' or 'Remind me to call John at 6 PM'.")

    while True:
        user_input = listen("Please tell me what I should remind you about and when.")
        if not user_input:
            speak("I didn’t catch that. Please try again.")
            continue

        reminder_dt = parse_datetime(user_input)
        message_text = re.sub(r"(remind me to|remind me|at\s+\d.*|tomorrow.*|in\s+\d+.*|today.*|morning|afternoon|evening|night)", "", user_input, flags=re.I).strip()
        
        if not message_text:
            speak("What would you like me to remind you about?")
            message_text = listen()

        if not reminder_dt:
            speak("When should I remind you?")
            time_input = listen()
            reminder_dt = parse_datetime(time_input)

        if not reminder_dt:
            speak("Sorry, I couldn’t understand the time. Please say it again.")
            continue

        human_time = reminder_dt.strftime("%I:%M %p on %A, %B %d, %Y")
        speak(f"I will remind you to {message_text} at {human_time}. Is that correct?")
        confirm = listen()

        if confirm and any(word in confirm.lower() for word in CONFIRM_YES):
            schedule_reminder(reminder_dt, message_text)
            speak(f"Your reminder to {message_text} is set for {human_time}.")
            speak("Thank you for using the reminder service. Have a wonderful day!")
            return
        elif confirm and any(word in confirm.lower() for word in CONFIRM_NO):
            speak("Alright, let's try again. Please tell me the new reminder details.")
            continue
        else:
            speak("I didn’t get that. Would you like to confirm or change it?")
            continue

def show_reminders():
    """
    Speaks reminders for today. If user asks to add a missing one,
    it redirects to set_reminder().
    """
    from datetime import datetime

    now = datetime.now()
    today = now.date()

    if not reminders:
        speak("You have no reminders for today.")
    else:
        # Filter today's reminders
        todays_reminders = [r for r in reminders if r["time"].date() == today]
        if not todays_reminders:
            speak("You don’t have any reminders scheduled for today.")
        else:
            speak("Here are your reminders for today:")
            for r in sorted(todays_reminders, key=lambda x: x["time"]):
                time_str = r["time"].strftime("%I:%M %p")
                speak(f"{r['message']} at {time_str}")

    # After listing, ask if user wants to add more
    response = listen("Would you like to add a new reminder or make changes?")

    if not response:
        speak("I didn’t catch that. Let’s try again later.")
        return

    response = response.lower()

    add_keywords = {"add", "new", "one more", "another", "set", "create", "missing", "forgot"}
    no_keywords = {"no", "nope", "not now", "cancel", "stop", "done", "that’s all", "that's it"}

    if any(word in response for word in add_keywords):
        speak("Sure, let's add a new reminder.")
        set_reminder()
    elif any(word in response for word in no_keywords):
        speak("Alright, no changes made to your reminders.")
    else:
        speak("Sorry, I didn’t understand. I’ll take that as no changes for now.")



def clear_reminders():
    reminders.clear()
    speak("All reminders for today are canceled.")


def get_weather():
    """
    Fully voice-driven weather assistant (no API key needed):
    - Detects your city automatically using IP.
    - Asks if you want another city.
    - Fetches current weather from wttr.in.
    - Speaks results naturally.
    - Tells if you need an umbrella.
    """

    engine = pyttsx3.init()
    recognizer = sr.Recognizer()

    def speak(text):
        print("SPEAK:", text)
        engine.say(text)
        engine.runAndWait()

    def listen(prompt=None):
        if prompt:
            speak(prompt)  
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("Heard:", text)
            return text
        except sr.UnknownValueError:
            speak("Sorry, I didn’t catch that. Could you repeat?")
            return listen()
        except sr.RequestError:
            speak("Sorry, the speech service is unavailable right now.")
            return None

    def fetch_weather(city=None):
        """Fetches weather data for given city (no API key)."""
        try:
            if not city:
                # Auto-detect user location
                loc = requests.get("https://ipinfo.io/json", timeout=5).json()
                city = loc.get("city", "your location")
            url = f"https://wttr.in/{city}?format=j1"
            data = requests.get(url, timeout=5).json()

            area = data["nearest_area"][0]["areaName"][0]["value"]
            condition = data["current_condition"][0]["weatherDesc"][0]["value"]
            temp = data["current_condition"][0]["temp_C"]
            feels_like = data["current_condition"][0]["FeelsLikeC"]
            humidity = data["current_condition"][0]["humidity"]

            speak(f"In {area}, it’s currently {condition.lower()} with a temperature of {temp}°C, feeling like {feels_like}°C.")

            # Simple umbrella suggestion logic
            if any(word in condition.lower() for word in ["rain", "shower", "drizzle", "storm"]):
                speak("It might rain soon. You should take an umbrella.")
            elif int(humidity) > 85:
                speak("Humidity is high — possible light rain, better carry an umbrella just in case.")
            else:
                speak("No signs of rain right now. You probably don’t need an umbrella.")

            return area

        except Exception as e:
            print("Error fetching weather:", e)
            speak("Sorry, I couldn’t fetch the weather right now. Please try again later.")
            return None

    # Step 1: Get current location weather
    speak("Let me check the weather for your current location.")
    current_city = fetch_weather()

    # Step 2: Ask if user wants another city
    response = listen("Would you like to check the weather for another city?")
    if not response:
        speak("I didn’t catch that. Let’s stop here for now.")
        return

    response = response.lower()
    yes_words = {"yes", "yeah", "yep", "ok", "okay", "sure", "please", "why not", "of course"}
    no_words = {"no", "nope", "nah", "not now", "cancel", "done", "stop"}

    if any(word in response for word in yes_words):
        city_name = listen("Sure, please tell me the city name.")
        if city_name:
            fetch_weather(city_name)
            speak("Thank you for using the weather service. Stay safe and have a nice day!")
        else:
            speak("Sorry, I didn’t get the city name.")
    elif any(word in response for word in no_words):
        speak("Alright, no problem. Thank you for using the weather service. Have a great day!")
    else:
        speak("I’ll take that as a no. Have a wonderful day!")


def get_temperature():
    # Placeholder
    speak("It’s a bit cool — around 22 degrees Celsius.")


def need_umbrella():
    """Voice-driven weather assistant for checking rain forecasts."""
    def get_location():
        """Fetch city and coordinates via IP geolocation."""
        try:
            res = requests.get("https://ipinfo.io/json", timeout=5)
            data = res.json()
            # Return both city and coordinates if available
            loc_str = data.get("loc", "")
            if loc_str:
                parts = loc_str.split(",")
                if len(parts) == 2:
                    return {
                        "city": data.get("city", None),
                        "coords": (float(parts[0]), float(parts[1]))
                    }
            return {"city": data.get("city", None), "coords": None}
        except Exception:
            return {"city": None, "coords": None}

    def check_rain_today(city):
        """Check if it's likely to rain today in the given city."""
        try:
            res = requests.get(f"https://wttr.in/{city}?format=j1", timeout=5)
            data = res.json()
            
            # Check current condition
            current_condition = data["current_condition"][0]["weatherDesc"][0]["value"].lower()
            is_raining_now = any(word in current_condition for word in ["rain", "shower", "drizzle", "storm"])
            
            # Check hourly forecast for today (next 12 hours)
            hourly = data["weather"][0]["hourly"][:12]  # next 12 hours (roughly today)
            will_rain_today = False
            max_rain_chance = 0
            
            for hour in hourly:
                chance = int(hour.get("chanceofrain", 0))
                max_rain_chance = max(max_rain_chance, chance)
                if chance > 50:
                    will_rain_today = True
            
            # Get today's date for context
            today_date = data["weather"][0]["date"]
            
            return {
                "is_raining_now": is_raining_now,
                "will_rain_today": will_rain_today,
                "max_rain_chance": max_rain_chance,
                "current_condition": current_condition,
                "date": today_date
            }
        except Exception as e:
            print(f"[Rain Check Error]: {e}")
            return None

    # Get current location and check rain
    location_data = get_location()
    city = location_data.get("city") if isinstance(location_data, dict) else location_data
    
    if not city:
        # Try to get location using the main function
        coords = get_current_location()
        if coords:
            # Reverse geocode to get city name
            try:
                geo_url = "https://nominatim.openstreetmap.org/reverse"
                r = requests.get(
                    geo_url,
                    params={"lat": coords[0], "lon": coords[1], "format": "json"},
                    headers={"User-Agent": "blind-nav"},
                    timeout=4
                )
                city = r.json().get("address", {}).get("city") or r.json().get("address", {}).get("town") or "your location"
            except:
                city = "your location"
        else:
            speak("Sorry, I couldn't determine your location. Please try again.")
            return

    speak(f"Let me check the rain forecast for today in {city}. Please wait.")
    rain_info = check_rain_today(city)

    if rain_info is None:
        speak("Sorry, I couldn't fetch the weather information right now. Please check your internet connection and try again.")
        return

    # Provide detailed rain information
    if rain_info["is_raining_now"]:
        speak(f"Yes, it's currently raining in {city}. You should definitely take an umbrella if you're going out.")
    elif rain_info["will_rain_today"]:
        speak(f"Yes, there's a chance of rain today in {city}. The maximum chance of rain is {rain_info['max_rain_chance']} percent. I recommend taking an umbrella with you.")
    else:
        if rain_info["max_rain_chance"] > 20:
            speak(f"There's a low chance of rain today in {city}, about {rain_info['max_rain_chance']} percent. You probably don't need an umbrella, but keep an eye on the weather.")
        else:
            speak(f"No, it doesn't look like it will rain today in {city}. The weather forecast shows clear conditions. You don't need an umbrella.")


def tell_joke():
    jokes = [
        "Why did the blind man fall into the well? Because he couldn’t see that well!",
        "I told my computer I needed a break, and it said — 'You seem stressed, shall I crash for you?'",
        "Why was the math book sad? Because it had too many problems!"
    ]
    speak(random.choice(jokes))


def play_music():
    """
    Voice-driven music assistant.
    - Asks user for song name and platform (YouTube, Spotify, or Google).
    - Opens song automatically in browser.
    """

    engine = pyttsx3.init()
    recognizer = sr.Recognizer()

    def speak(text):
        print("SPEAK:", text)
        engine.say(text)
        engine.runAndWait()

    def listen(prompt=None):
        if prompt:
            speak(prompt)
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("Heard:", text)
            return text.lower()
        except sr.UnknownValueError:
            speak("Sorry, I didn’t catch that. Please repeat.")
            return listen()
        except sr.RequestError:
            speak("Speech recognition is unavailable right now.")
            return None

    # Step 1: Ask what song or artist to play
    song_query = listen("What song or artist would you like to listen to?")
    if not song_query:
        speak("I didn’t hear the song name. Please try again later.")
        return

    # Step 2: Ask where to play it
    platform = listen("Where would you like to play it? You can say YouTube, Spotify, or Google.")

    if not platform:
        platform = "youtube"  # default
        speak("I’ll play it on YouTube by default.")

    platform = platform.lower()
    encoded_query = urllib.parse.quote(song_query)

    # Step 3: Open appropriate platform
    if "spotify" in platform:
        url = f"https://open.spotify.com/search/{encoded_query}"
        speak(f"Playing {song_query} on Spotify.")
        webbrowser.open(url)
    elif "google" in platform:
        url = f"https://www.google.com/search?q={encoded_query}+song"
        speak(f"Searching {song_query} on Google.")
        webbrowser.open(url)
    else:
        # Default YouTube
        url = f"https://www.youtube.com/results?search_query={encoded_query}"
        speak(f"Playing {song_query} on YouTube.") 
        webbrowser.open(url)

    speak("Enjoy your music!")


def stop_music():
    """
    Stops music playback, whether it’s local or web-based.
    If music was opened via a browser tab or local player,
    it tries to close or stop it gracefully.
    """

    engine = pyttsx3.init()

    def speak(text):
        print("SPEAK:", text)
        engine.say(text)
        engine.runAndWait()

    # --- 1️⃣ Attempt to close browser music tabs (YouTube, Spotify, etc.)
    try:
        for proc in psutil.process_iter(["pid", "name"]):
            name = proc.info["name"].lower()
            if "chrome" in name or "firefox" in name or "edge" in name:
                # kill browser tabs playing music
                proc.send_signal(signal.SIGTERM)
                speak("Music stopped by closing browser playback.")
                return
    except Exception:
        pass

    # --- 2️⃣ Attempt to stop local playback processes (like VLC or playsound)
    try:
        for proc in psutil.process_iter(["pid", "name"]):
            if any(x in proc.info["name"].lower() for x in ["vlc", "mplayer", "wmplayer"]):
                proc.send_signal(signal.SIGTERM)
                speak("Music player stopped.")
                return
    except Exception:
        pass

    # --- 3️⃣ Fallback
    speak("Music stopped.")


def personalized_greeting():
    speak("Good morning! You have two reminders today — take medicine at 9 a.m. and meeting at 4 p.m.")


def get_battery_status():
    """
    Fetches real-time battery percentage and charging status,
    and speaks the result.
    """
    engine = pyttsx3.init()

    def speak(text):
        print("SPEAK:", text)
        engine.say(text)
        engine.runAndWait()

    try:
        battery = psutil.sensors_battery()
        if battery is None:
            speak("Sorry, I couldn’t detect a battery on this device.")
            return

        percent = int(battery.percent)
        charging = battery.power_plugged

        if charging:
            speak(f"Your device is charging and the battery is at {percent} percent.")
            if percent >= 95:
                speak("You might want to unplug it soon to preserve battery health.")
        else:
            speak(f"Your device battery is at {percent} percent.")
            if percent <= 20:
                speak("Battery is low. Please plug in your charger soon.")
            elif percent <= 50:
                speak("Battery is below half. You may want to keep an eye on it.") 
            elif percent >= 90:
                speak("Battery is almost full. Great job keeping it charged!")

    except Exception as e:
        print("Error reading battery status:", e)
        speak("Sorry, I couldn’t read the battery information right now.")


def change_voice():
    speak("Voice changed to a female tone. (Simulated)")


def speak_slower():
    speak("Okay, I’ll speak a little slower from now on. (Simulated)")


silent_mode = False  # global flag to track silence state

def stop_talking():
    """
    Puts the assistant into silent listening mode.
    It stops responding verbally until the user calls it back.
    """

    global silent_mode
    silent_mode = True

    engine = pyttsx3.init()
    recognizer = sr.Recognizer()

    def speak(text):
        if not silent_mode:
            print("SPEAK:", text)
            engine.say(text)
            engine.runAndWait()
        else:
            print(f"(Silent mode) {text}")

    def listen(prompt=None):
        if prompt and not silent_mode:
            speak(prompt)
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("Heard:", text)
            return text.lower()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""

    speak("Okay, I’ll stay quiet until you call me again.")
    print("Assistant is now silent... Listening for wake word.")

    wake_words = ["wake up", "you can talk", "hello assistant", "resume", "talk again", "start speaking"]

    while silent_mode:
        command = listen()
        if any(word in command for word in wake_words):
            silent_mode = False
            speak("I'm back. How can I help you?")
            break
        time.sleep(1)


# ---------------- MAIN MODE LOGIC ----------------

def run_assistant_mode():
    speak("Assistant mode activated.")

    while True:
        command = listen_for_speech()
        if not command:
            continue

        command = command.lower()

        # Handle bus stop queries BEFORE exit check (to prevent false exit triggers)
        if any(phrase in command for phrase in [
            "nearest bus stop", "find bus stop", "where is bus stop",
            "bus stop", "nearest stop", "find stop", "show me bus stop",
            "locate bus stop", "bus stop near me", "closest bus stop"
        ]):
            try:
                speak("Finding the nearest bus stop...")
                start_coords = get_current_location()
                if start_coords:
                    bus_stop = find_nearby_bus_stops(start_coords[0], start_coords[1])
                    if bus_stop:
                        speak(f"The nearest bus stop is {bus_stop['name']}, approximately {bus_stop['distance']} meters away.")
                    else:
                        speak("I couldn't find any nearby bus stops. Please try a different location or check your internet connection.")
                else:
                    speak("I couldn't determine your current location. Please enable location services or tell me your address.")
            except Exception as e:
                print(f"[Bus Stop Query Error]: {e}")
                speak("I encountered an error while searching for bus stops. Please try again.")
            continue  # Continue to next iteration, don't check for exit
        
        # Exit condition - run continuously until exit is said
        if is_exit_command(command):
            speak("Exiting assistant mode. Returning to main system.")
            break  # Exit the loop

        # Capability overview / help phrases
        elif any(
            phrase in command
            for phrase in [
                "how can you assist",
                "how can you help",
                "how do you assist",
                "what can you do",
                "what should i do",
                "how can i help",
                "how can you serve",
                "what should i know",
                "help me understand",
            ]
        ):
            explain_assistant_capabilities()

        elif "help me" in command and "describe" not in command and "navigate" not in command:
            speak("I'm here. You can ask for time, weather, reminders, scene descriptions, or even navigation help. What would you like me to do?")

        # Weather and climate - Check BEFORE date/time to avoid false matches
        # Check for rain queries first (including "will it rain today")
        elif "rain" in command or "umbrella" in command or "will it rain" in command:
            need_umbrella()

        elif "weather" in command:
            get_weather()

        elif "temperature" in command:
            get_temperature()

        # Time and date - Check AFTER weather to avoid matching "today" in weather queries
        elif "time" in command:
            speak(f"The time is {get_time()}.")

        # Only match "today" if it's clearly a date query, not a weather query
        elif ("date" in command or ("today" in command and "rain" not in command and "weather" not in command)):
            speak(f"Today is {get_date()}.")   

        elif "morning" in command or "afternoon" in command or "evening" in command:
            speak(f"It's {get_period()} right now.")

        elif "wake me up" in command or "set alarm" in command:
            set_alarm("7 a.m. tomorrow")

        # Reminders
        elif "remind me" in command:
            add_reminder("take your medicine", "9 p.m.")

        elif "what reminders" in command or "show reminders" in command:
            show_reminders()

        elif "cancel reminders" in command:
            clear_reminders()

        # Entertainment / Chat
        elif "play music" in command:
            play_music()

        elif "stop music" in command:
            stop_music()

        elif "joke" in command:
            tell_joke()

        # Misc
        elif "good morning" in command:
            personalized_greeting()

        elif "battery" in command:
            get_battery_status()

        elif "change your voice" in command or "change voice" in command:
            change_voice()

        elif "speak slower" in command:
            speak_slower()

        elif "stop talking" in command:
            stop_talking()

        # Scene description
        elif "describe" in command and "scene" in command:
            describe_scene()

        # Navigation support requests
        elif any(
            phrase in command
            for phrase in [
                "indoor navigation",
                "guide me out",
                "help me go out",
                "need to go out",
                "go outside",
                "take me outside",
                "help me outside",
                "help me navigate",
                "start indoor guidance",
            ]
        ):
            speak("I'll help you with navigation.")
            prompt_navigation_support()

        elif "start navigation" in command or "start smart navigation" in command:
            prompt_navigation_support()

        # Mode switching (future integration)
        elif "switch" in command or "mode" in command:
            speak("Mode switching is managed by the main system. Please say 'activate navigation mode' or another mode name in main system.")
            break

        elif any(phrase in command for phrase in ["please shut down", "shut down", "power off", "turn off", "quit"]):
            speak("Alright, I'll take you back to the main menu. Say 'exit' there if you want to close the system.")
            break

        elif handle_smalltalk(command):
            continue

        elif "repeat" in command and "modes" in command:
            speak_available_modes_overview()

        else:
            speak(
                "I'm here to help with time, weather, reminders, scene descriptions, indoor or outdoor guidance, and emergencies. Could you rephrase or ask for help to hear what I can do?"
            )

    # End of while loop — now returning to main system
    return  # <-- This ensures control returns to the main program



# # Add the directory to sys.path
# if module_path not in sys.path:
#     sys.path.append(module_path)

# ===== Optional Text-to-Speech =====
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False


# ===============================
# CONFIGURATION
# ===============================
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImMyMDQxMDc5ZWE4MTQwZWU4MGE3NTUwNDdlODA2NDE3IiwiaCI6Im11cm11cjY0In0="
client = openrouteservice.Client(key=ORS_API_KEY)
geolocator = Nominatim(user_agent="blind_navigation_assistant")

# Note: speak() is imported from utils.speech_utils - don't redefine it here

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        audio = r.listen(source, phrase_time_limit=5)
    try:
        command = r.recognize_google(audio)
        print(f"👂 You said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        speak("Sorry, I didn’t catch that. Please repeat.")
        return listen()
    except sr.RequestError:
        speak("Voice service error. Try again.")
        return ""

# ===============================
# GEOLOCATION FUNCTIONS
# ===============================
def get_current_location():
    """
    Automatically detect current location using multiple methods in parallel.
    Returns precise coordinates (latitude, longitude) for accurate navigation.
    """
    speak("Detecting your exact location using multiple sources for better accuracy...")
    
    location_results = []
    threads = []
    results_lock = threading.Lock()
    
    def try_ipinfo():
        """Try ipinfo.io for location."""
        try:
            response = requests.get("https://ipinfo.io/json", timeout=6)
            if response.status_code == 200:
                data = response.json()
                loc_str = data.get("loc", "")
                if loc_str:
                    parts = loc_str.split(",")
                    if len(parts) == 2:
                        lat = float(parts[0])
                        lon = float(parts[1])
                        with results_lock:
                            location_results.append({
                                "coords": (lat, lon),
                                "city": data.get("city", "Unknown"),
                                "region": data.get("region", ""),
                                "method": "ipinfo",
                                "accuracy": "city-level"
                            })
        except Exception as e:
            print("[Location Detection Error - ipinfo]", e)
    
    def try_ipapi():
        """Try ipapi.co for location."""
        try:
            response = requests.get("https://ipapi.co/json/", timeout=6)
            if response.status_code == 200:
                data = response.json()
                lat = data.get("latitude")
                lon = data.get("longitude")
                if lat and lon:
                    with results_lock:
                        location_results.append({
                            "coords": (float(lat), float(lon)),
                            "city": data.get("city", "Unknown"),
                            "region": data.get("region", ""),
                            "method": "ipapi",
                            "accuracy": "city-level"
                        })
        except Exception as e:
            print("[Location Detection Error - ipapi]", e)
    
    def try_ip_api():
        """Try ip-api.com for location."""
        try:
            response = requests.get("http://ip-api.com/json/", timeout=6)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    lat = data.get("lat")
                    lon = data.get("lon")
                    if lat and lon:
                        with results_lock:
                            location_results.append({
                                "coords": (float(lat), float(lon)),
                                "city": data.get("city", "Unknown"),
                                "region": data.get("regionName", ""),
                                "method": "ip-api",
                                "accuracy": "city-level"
                            })
        except Exception as e:
            print("[Location Detection Error - ip-api]", e)
    
    def try_geojs():
        """Try geojs.io for location."""
        try:
            response = requests.get("https://get.geojs.io/v1/ip/geo.json", timeout=6)
            if response.status_code == 200:
                data = response.json()
                lat = data.get("latitude")
                lon = data.get("longitude")
                if lat and lon:
                    with results_lock:
                        location_results.append({
                            "coords": (float(lat), float(lon)),
                            "city": data.get("city", "Unknown"),
                            "region": data.get("region", ""),
                            "method": "geojs",
                            "accuracy": "city-level"
                        })
        except Exception as e:
            print("[Location Detection Error - geojs]", e)
    
    def try_ipgeolocation():
        """Try ipgeolocation.io for more accurate location."""
        try:
            response = requests.get("https://api.ipgeolocation.io/ipgeo?apiKey=free", timeout=6)
            if response.status_code == 200:
                data = response.json()
                lat = data.get("latitude")
                lon = data.get("longitude")
                if lat and lon:
                    with results_lock:
                        location_results.append({
                            "coords": (float(lat), float(lon)),
                            "city": data.get("city", "Unknown"),
                            "region": data.get("state_prov", ""),
                            "method": "ipgeolocation",
                            "accuracy": "city-level"
                        })
        except Exception as e:
            print("[Location Detection Error - ipgeolocation]", e)
    
    # Run all location services in parallel
    t1 = threading.Thread(target=try_ipinfo, daemon=True)
    t2 = threading.Thread(target=try_ipapi, daemon=True)
    t3 = threading.Thread(target=try_ip_api, daemon=True)
    t4 = threading.Thread(target=try_geojs, daemon=True)
    t5 = threading.Thread(target=try_ipgeolocation, daemon=True)
    
    threads = [t1, t2, t3, t4, t5]
    for t in threads:
        t.start()
    
    # Wait for all threads to complete (with timeout)
    for t in threads:
        t.join(timeout=6)
    
    # Process results
    if location_results:
        # Calculate average coordinates from all successful results for better accuracy
        if len(location_results) > 1:
            avg_lat = sum(r["coords"][0] for r in location_results) / len(location_results)
            avg_lon = sum(r["coords"][1] for r in location_results) / len(location_results)
            
            # Use the most common city name
            cities = [r["city"] for r in location_results]
            most_common_city = max(set(cities), key=cities.count)
            regions = [r["region"] for r in location_results if r.get("region")]
            most_common_region = max(set(regions), key=regions.count) if regions else ""
            
            speak(f"Detected your location: {most_common_city}, {most_common_region}")
            speak(f"Coordinates: {avg_lat:.6f}, {avg_lon:.6f}. Using averaged location from {len(location_results)} sources for better accuracy.")
            return (avg_lat, avg_lon)
        else:
            # Single result
            result = location_results[0]
            speak(f"Detected your location: {result['city']}, {result['region']}")
            speak(f"Coordinates: {result['coords'][0]:.6f}, {result['coords'][1]:.6f}")
            return result["coords"]
    
    # If all methods failed, try one more time with a single method
    speak("Trying alternative location detection method...")
    try:
        response = requests.get("https://ipapi.co/json/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            lat = data.get("latitude")
            lon = data.get("longitude")
            if lat and lon:
                city = data.get("city", "Unknown")
                region = data.get("region", "")
                speak(f"Detected your location: {city}, {region}")
                speak(f"Coordinates: {lat:.6f}, {lon:.6f}")
                return (float(lat), float(lon))
    except Exception as e:
        print("[Location Detection Error - final attempt]", e)
    
    speak("Could not automatically detect your exact location. For more accurate navigation, please provide your address or enable GPS on your device.")
    speak("Note: IP-based location detection provides city-level accuracy (within 1-5 kilometers). For precise navigation, GPS coordinates are recommended.")
    return None

def get_coordinates(location_name):
    try:
        loc = geolocator.geocode(location_name, timeout=3)
        if loc:
            return (loc.latitude, loc.longitude)
    except Exception as e:
        print("[Geocode Error]", e)
    return None

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def calculate_route(start_coords, dest_coords, mode="foot-walking"):
    """Calculate route using OSRM for step-by-step navigation."""
    try:
        # Map mode to OSRM profile
        profile_map = {
            "foot-walking": "walking",
            "driving-car": "driving",
            "cycling-regular": "driving"  # OSRM uses driving for bikes
        }
        profile = profile_map.get(mode, "walking")
        
        # Use OSRM for detailed step-by-step routing
        url = f"http://router.project-osrm.org/route/v1/{profile}/{start_coords[1]},{start_coords[0]};{dest_coords[1]},{dest_coords[0]}?overview=full&steps=true"
        response = requests.get(url)
        route_data = response.json()
        
        if not route_data.get("routes"):
            # Fallback to OpenRouteService
            route = client.directions(
                coordinates=[(start_coords[1], start_coords[0]), (dest_coords[1], dest_coords[0])],
                profile=mode,
                format="geojson"
            )
            distance = route["features"][0]["properties"]["summary"]["distance"] / 1000
            duration = route["features"][0]["properties"]["summary"]["duration"] / 60
            return {"distance": round(distance, 2), "duration": round(duration, 1), "raw": route, "steps": None}
        
        # Extract route info from OSRM
        route_info = route_data["routes"][0]
        distance_km = route_info["distance"] / 1000
        duration_min = route_info["duration"] / 60
        steps = route_info["legs"][0]["steps"] if route_info.get("legs") else []
        
        return {
            "distance": round(distance_km, 2),
            "duration": round(duration_min, 1),
            "raw": route_data,
            "steps": steps
        }
    except Exception as e:
        speak("I couldn't calculate the route.")
        print(f"[Route Error]: {e}")
        return None

def find_nearby_bus_stops(lat, lon, radius=1000):
    """
    Find nearby bus stops using Overpass API.
    Uses exact coordinates for precise distance calculation.
    Returns the nearest bus stop with distance in meters.
    """
    try:
        overpass_url = "https://overpass-api.de/api/interpreter"
        # Increased radius to 1000m (1km) for better coverage
        query = f'[out:json];node(around:{radius},{lat},{lon})[highway=bus_stop];out;'
        response = requests.get(overpass_url, params={'data': query}, timeout=10)
        data = response.json()
        
        if data.get("elements"):
            # Find the closest bus stop by calculating distance to all
            nearest_stop = None
            min_distance = float('inf')
            
            for stop in data["elements"]:
                if "lat" in stop and "lon" in stop:
                    stop_lat = float(stop["lat"])
                    stop_lon = float(stop["lon"])
                    distance = haversine(lat, lon, stop_lat, stop_lon)
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_stop = {
                            "name": stop.get("tags", {}).get("name", "Bus Stop"),
                            "lat": stop_lat,
                            "lon": stop_lon,
                            "distance": int(distance)
                        }
            
            if nearest_stop:
                return nearest_stop
        
        return None
    except Exception as e:
        print(f"[Bus Stop Error]: {e}")
        return None

def parse_location_coordinates(text):
    """Parse location from text, supporting both place names and coordinates."""
    # Try to extract coordinates if user says numbers like "twelve point nine seven one six"
    try:
        # Replace common words
        text_lower = text.lower().replace("point", ".").replace(",", " ")
        # Try to extract numbers
        parts = text_lower.split()
        numbers = []
        for p in parts:
            try:
                num = float(p)
                numbers.append(num)
            except ValueError:
                continue
        
        if len(numbers) >= 2:
            return (numbers[0], numbers[1])
    except:
        pass
    
    # Otherwise try geocoding
    return None

# ===============================
# EXIT DETECTION HELPER
# ===============================
def is_exit_command(text):
    """
    Check if user wants to exit the current mode.
    Context-aware to avoid false positives like "bus stop", "train stop", etc.
    """
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Exclude common phrases that contain "stop" but aren't exit commands
    exclusion_patterns = [
        "bus stop", "train stop", "metro stop", "subway stop",
        "stop sign", "stop light", "traffic stop", "next stop",
        "nearest stop", "find stop", "where is stop", "show stop",
        "stop here", "stop there", "stop at", "stop near",
        "stop location", "stop address", "stop directions"
    ]
    
    # Check if text contains any exclusion pattern - if so, it's NOT an exit command
    for pattern in exclusion_patterns:
        if pattern in text_lower:
            return False
    
    # Specific exit phrases (more reliable)
    specific_exit_phrases = [
        "exit mode", "stop mode", "quit mode", "end mode",
        "exit navigation", "stop navigation", "exit reading", "stop reading",
        "exit walking", "stop walking", "exit detection", "stop detection",
        "exit assistant", "stop assistant", "exit system", "stop system",
        "quit system", "shut down", "turn off", "close system",
        "go back", "return to main", "back to main", "main menu",
        "exit now", "stop now", "quit now", "end session",
        "deactivate mode", "disable mode"
    ]
    
    # Check for specific exit phrases first (most reliable)
    for phrase in specific_exit_phrases:
        if phrase in text_lower:
            return True
    
    # Check for standalone exit words (only if they appear as complete words or at sentence boundaries)
    # This prevents "bus stop" from matching
    exit_words = ["exit", "quit", "end", "cancel", "deactivate"]
    
    # Check if exit word appears as a standalone word (not part of another word)
    import re
    for word in exit_words:
        # Match word boundaries to ensure it's not part of another word
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, text_lower):
            return True
    
    # For "stop", require it to be at the start, end, or with context words
    # This prevents "bus stop" from matching
    if "stop" in text_lower:
        # Check if "stop" appears with exit context
        stop_exit_contexts = [
            "stop the", "stop this", "stop it", "stop everything",
            "stop system", "stop mode", "stop navigation", "stop reading",
            "stop walking", "stop detection", "stop assistant",
            "please stop", "i want to stop", "let me stop"
        ]
        
        for context in stop_exit_contexts:
            if context in text_lower:
                return True
        
        # If "stop" is at the very beginning or end of the sentence, it might be an exit command
        # But only if it's not part of a location phrase
        text_stripped = text_lower.strip()
        if text_stripped.startswith("stop") and len(text_stripped.split()) <= 3:
            # Short phrases starting with "stop" might be exit commands
            # But exclude if it contains location words
            location_words = ["stop", "here", "there", "at", "near", "to", "the"]
            words = text_stripped.split()
            if len(words) == 1 or (len(words) == 2 and words[1] in location_words):
                # Could be "stop" or "stop here" - check if it's clearly an exit
                if text_stripped in ["stop", "stop it", "stop this", "stop now"]:
                    return True
    
    return False

# ===============================
# REAL-TIME CAMERA NAVIGATION
# ===============================
def _load_mobilenet_ssd():
    """Load MobileNet-SSD model for object detection."""
    try:
        prototxt_path = "models/MobileNetSSD_deploy.prototxt.txt"
        caffemodel_path = "models/MobileNetSSD_deploy.caffemodel"
        net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        return net
    except Exception as e:
        print(f"[Model Load Error]: {e}")
        return None

CLASSES_NAV = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

def _detect_environment(frame, detections):
    """Detect if user is indoors or outdoors based on camera analysis."""
    h, w = frame.shape[:2]
    
    # Check for outdoor indicators
    outdoor_indicators = ["car", "bus", "motorbike", "bicycle", "train"]
    vehicle_count = sum(1 for det in detections if det.get("label") in outdoor_indicators)
    
    # Analyze brightness and color distribution (outdoors usually brighter, more varied)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    std_dev = np.std(gray)
    
    # Indoor typically has: furniture (chair, sofa, diningtable), lower brightness variance
    indoor_indicators = ["chair", "sofa", "diningtable", "tvmonitor"]
    indoor_count = sum(1 for det in detections if det.get("label") in indoor_indicators)
    
    # Decision logic
    if vehicle_count >= 2 or (brightness > 100 and std_dev > 40 and vehicle_count >= 1):
        return "outdoor"
    elif indoor_count >= 2 or (brightness < 80 and std_dev < 30):
        return "indoor"
    else:
        return "unknown"

def _detect_road_crossing(frame, detections, environment):
    """Detect if user is approaching or at a road crossing - only outdoors."""
    # Only detect road crossings if we're outdoors
    if environment != "outdoor":
        return False
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    roi = gray[int(h*0.6):, :]
    
    # Check for vehicles first (road crossing needs vehicles nearby)
    vehicle_count = sum(1 for det in detections if det.get("label") in ["car", "bus", "motorbike", "bicycle", "train"])
    if vehicle_count == 0:
        return False  # No vehicles = likely not a road
    
    # Edge detection for road markings
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=15)
    
    if lines is not None and len(lines) > 0:
        horizontal_lines = 0
        long_horizontal = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Horizontal lines (road markings)
            if 85 < angle < 95 or angle < 5 or angle > 175:
                horizontal_lines += 1
                if length > w * 0.3:  # Long horizontal lines (road markings)
                    long_horizontal += 1
        
        # Require multiple indicators: vehicles + horizontal lines + long lines
        if horizontal_lines >= 5 and long_horizontal >= 2 and vehicle_count >= 1:
            return True
    
    return False

def _analyze_path_clearance(frame, detections):
    """Analyze which direction (left/right/forward) is clearest."""
    h, w = frame.shape[:2]
    left_score = 0.0
    right_score = 0.0
    center_score = 0.0
    
    for det in detections:
        x_center = (det["startX"] + det["endX"]) / 2.0
        ratio = det["ratio"]
        
        if x_center < w * 0.33:
            left_score += ratio
        elif x_center > w * 0.67:
            right_score += ratio
        else:
            center_score += ratio
    
    if center_score < 0.1 and left_score < 0.1 and right_score < 0.1:
        return "forward"
    elif left_score < right_score and left_score < center_score:
        return "left"
    elif right_score < left_score and right_score < center_score:
        return "right"
    else:
        return "forward"


def check_camera_orientation(frame):
    """
    Check camera orientation and detect tilting issues.
    Returns: dict with 'is_aligned' (bool), 'tilt_direction' (str), 'tilt_angle' (float), 'message' (str)
    """
    h, w = frame.shape[:2]
    result = {
        'is_aligned': True,
        'tilt_direction': None,  # 'up', 'down', 'left', 'right', 'too_tilted'
        'tilt_angle': 0.0,
        'message': None,
        'severity': 'none'  # 'none', 'minor', 'major'
    }
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Detect horizon using edge detection and Hough lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Horizontal lines (should be ~0 or 180 degrees)
                if angle < 10 or angle > 170:
                    horizontal_lines.append((x1, y1, x2, y2, angle))
                # Vertical lines (should be ~90 degrees)
                elif 80 < angle < 100:
                    vertical_lines.append((x1, y1, x2, y2, angle))
        
        # Method 2: Analyze feature distribution
        # If camera is tilted up, more features/texture in top half
        # If camera is tilted down, more features/texture in bottom half
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        
        top_variance = np.var(top_half)
        bottom_variance = np.var(bottom_half)
        left_variance = np.var(left_half)
        right_variance = np.var(right_half)
        
        # Method 3: Check vertical alignment using vertical lines
        vertical_tilt = 0.0
        horizontal_tilt = 0.0
        
        if len(vertical_lines) > 3:
            # Calculate average vertical line position
            avg_x = np.mean([(l[0] + l[2]) / 2 for l in vertical_lines])
            center_x = w / 2
            horizontal_tilt = (avg_x - center_x) / w  # Normalized tilt
        
        if len(horizontal_lines) > 3:
            # Calculate average horizontal line position and angle
            avg_y = np.mean([(l[1] + l[3]) / 2 for l in horizontal_lines])
            center_y = h / 2
            vertical_tilt = (avg_y - center_y) / h  # Normalized tilt
            
            # Check if horizontal lines are actually horizontal
            avg_angle = np.mean([l[4] for l in horizontal_lines])
            if avg_angle > 10 and avg_angle < 170:
                # Lines are not horizontal - camera is rotated
                if avg_angle < 90:
                    horizontal_tilt = -abs(horizontal_tilt)  # Tilted left
                else:
                    horizontal_tilt = abs(horizontal_tilt)  # Tilted right
        
        # Determine tilt direction and severity
        tilt_threshold_minor = 0.15  # 15% tilt
        tilt_threshold_major = 0.30  # 30% tilt
        
        # Vertical tilt (up/down)
        if abs(vertical_tilt) > tilt_threshold_major:
            result['is_aligned'] = False
            result['severity'] = 'major'
            if vertical_tilt > 0:
                result['tilt_direction'] = 'down'
                result['message'] = "Camera facing too low. Please tilt up."
            else:
                result['tilt_direction'] = 'up'
                result['message'] = "Camera facing too high. Please tilt down."
            result['tilt_angle'] = abs(vertical_tilt) * 100
        elif abs(vertical_tilt) > tilt_threshold_minor:
            result['is_aligned'] = False
            result['severity'] = 'minor'
            if vertical_tilt > 0:
                result['tilt_direction'] = 'down'
                result['message'] = "Camera is slightly low. Please tilt up a bit."
            else:
                result['tilt_direction'] = 'up'
                result['message'] = "Camera is slightly high. Please tilt down a bit."
            result['tilt_angle'] = abs(vertical_tilt) * 100
        
        # Horizontal tilt (left/right) - less critical but still important
        if abs(horizontal_tilt) > tilt_threshold_major:
            if not result['is_aligned']:
                result['tilt_direction'] = 'too_tilted'
                result['message'] = "Camera is tilted. Please align it straight and level."
            else:
                result['is_aligned'] = False
                result['severity'] = 'major'
                if horizontal_tilt > 0:
                    result['tilt_direction'] = 'right'
                    result['message'] = "Camera is tilted to the right. Please rotate left."
                else:
                    result['tilt_direction'] = 'left'
                    result['message'] = "Camera is tilted to the left. Please rotate right."
        elif abs(horizontal_tilt) > tilt_threshold_minor and result['is_aligned']:
            result['is_aligned'] = False
            result['severity'] = 'minor'
            if horizontal_tilt > 0:
                result['tilt_direction'] = 'right'
                result['message'] = "Camera is slightly tilted right. Please adjust."
            else:
                result['tilt_direction'] = 'left'
                result['message'] = "Camera is slightly tilted left. Please adjust."
        
        # Additional check: If top half has much more variance than bottom (camera pointing up)
        if top_variance > bottom_variance * 1.5 and len(horizontal_lines) < 3:
            if result['is_aligned']:
                result['is_aligned'] = False
                result['severity'] = 'minor'
                result['tilt_direction'] = 'up'
                result['message'] = "Camera appears to be facing upward. Please tilt down to see the path ahead."
        
        # If bottom half has much more variance (camera pointing down)
        elif bottom_variance > top_variance * 1.5 and len(horizontal_lines) < 3:
            if result['is_aligned']:
                result['is_aligned'] = False
                result['severity'] = 'minor'
                result['tilt_direction'] = 'down'
                result['message'] = "Camera appears to be facing downward. Please tilt up to see ahead."
        
    except Exception as e:
        print(f"[Camera Orientation Error]: {e}")
        result['is_aligned'] = True  # Assume aligned on error
    
    return result

def _start_realtime_navigation(route, dest_name, dest_coords, start_coords):
    """Real-time camera-based navigation with obstacle detection and path analysis."""
    speak(f"Starting real-time navigation to {dest_name}.")
    speak("I will guide you using the camera. Keep the camera facing forward.")
    
    # Load detection model
    net = _load_mobilenet_ssd()
    if not net:
        speak("Could not load vision model. Using basic navigation.")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not available. Please check the connection.")
        return
    
    last_said = {}
    last_instruction = ""
    route_steps = route.get("steps", [])
    current_step_idx = 0
    distance_remaining = route.get("distance", 0) * 1000  # Convert to meters
    empty_scene_counter = 0
    low_texture_counter = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                _speak_rate_limited(last_said, "cam", "Camera feed lost. Trying again...", 3.0)
                time.sleep(0.5)
                continue
            
            h, w = frame.shape[:2]
            
            # Check camera orientation first
            orientation = check_camera_orientation(frame)
            camera_too_tilted = False
            
            if not orientation['is_aligned']:
                # Provide audio feedback about camera orientation
                if orientation['severity'] == 'major':
                    _speak_rate_limited(last_said, "camera_orientation", orientation['message'], 5.0)
                    camera_too_tilted = True  # Pause obstacle detection if too tilted
                elif orientation['severity'] == 'minor':
                    _speak_rate_limited(last_said, "camera_orientation_minor", orientation['message'], 8.0)
            
            # Object detection (skip if camera is too tilted)
            if camera_too_tilted:
                # Pause obstacle detection but continue checking orientation
                _speak_rate_limited(last_said, "detection_paused", "Obstacle detection paused due to camera misalignment. Please adjust the camera.", 10.0)
                time.sleep(0.5)
                continue
            
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            
            closest_obstacle = None
            det_list = []
            vehicles_detected = []
            
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < 0.4:
                    continue
                
                idx = int(detections[0, 0, i, 1])
                if idx < 0 or idx >= len(CLASSES_NAV):
                    continue
                
                label = CLASSES_NAV[idx]
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                height_ratio = (endY - startY) / float(h)
                
                # Track obstacles
                if label in ["person", "car", "bus", "bicycle", "motorbike", "train", "dog", "chair", "sofa", "diningtable"]:
                    if closest_obstacle is None or height_ratio > closest_obstacle["ratio"]:
                        closest_obstacle = {
                            "label": label,
                            "startX": startX,
                            "startY": startY,
                            "endX": endX,
                            "endY": endY,
                            "ratio": height_ratio
                        }
                    det_list.append({
                        "label": label,
                        "startX": startX,
                        "startY": startY,
                        "endX": endX,
                        "endY": endY,
                        "ratio": height_ratio
                    })
                
                # Track vehicles for road crossing
                if label in ["car", "bus", "motorbike", "bicycle", "train"]:
                    vehicles_detected.append({"label": label, "ratio": height_ratio})
            
            depth_messages, low_texture_flag = analyze_depth_for_navigation(frame)
            depth_alert_active = False
            for idx_msg, msg in enumerate(depth_messages):
                _speak_rate_limited(last_said, f"depth_nav_{idx_msg}", msg, 6.0)
                if any(keyword in msg.lower() for keyword in ["caution", "edge", "wall", "stop"]):
                    depth_alert_active = True

            if low_texture_flag:
                low_texture_counter += 1
            else:
                low_texture_counter = 0

            if low_texture_counter >= 8:
                _speak_rate_limited(
                    last_said,
                    "nav_lowtexture",
                    "The camera view seems flat. Please tilt or move the camera slightly so I can scan better.",
                    10.0,
                )
                low_texture_counter = 0

            if not det_list and not depth_messages:
                empty_scene_counter += 1
            else:
                empty_scene_counter = 0

            if empty_scene_counter >= 10:
                _speak_rate_limited(
                    last_said,
                    "nav_empty_scene",
                    "I can't detect clear features. Please adjust the camera angle or turn slowly so I can rescan.",
                    12.0,
                )
                empty_scene_counter = 0

            # Detect environment (indoor/outdoor) for context-aware guidance
            environment = _detect_environment(frame, det_list)
            
            # Provide context-aware guidance based on environment
            if environment == "indoor":
                # Indoor-specific guidance
                if closest_obstacle is not None:
                    if closest_obstacle["ratio"] > 0.55:
                        last_instruction = f"Stop! {closest_obstacle['label']} very close ahead. Move around it carefully."
                        _speak_rate_limited(last_said, "stop", last_instruction, 2.0)
                    elif closest_obstacle["ratio"] > 0.35:
                        preferred_dir = _analyze_path_clearance(frame, det_list)
                        if preferred_dir == "left":
                            last_instruction = f"Move left to avoid the {closest_obstacle['label']} ahead."
                        elif preferred_dir == "right":
                            last_instruction = f"Move right to avoid the {closest_obstacle['label']} ahead."
                        else:
                            last_instruction = f"Careful, {closest_obstacle['label']} ahead. Continue slowly."
                        _speak_rate_limited(last_said, "avoid", last_instruction, 2.5)
                    else:
                        last_instruction = "Path clear. Continue forward."
                        _speak_rate_limited(last_said, "clear", last_instruction, 3.0)
                else:
                    last_instruction = "Path clear. Continue forward."
                    _speak_rate_limited(last_said, "clear", last_instruction, 3.0)
                # Skip road crossing detection indoors
                if depth_alert_active:
                    _speak_rate_limited(last_said, "indoor_depth", "Something ahead looks unsafe. Please stay put and adjust the camera so I can rescan.", 8.0)
                    continue
                is_road_crossing = False
            else:
                # Outdoor guidance - includes road crossing detection
                is_road_crossing = _detect_road_crossing(frame, det_list, environment)
            
            # Road crossing detection - only outdoors and with proper indicators
            if is_road_crossing:
                if vehicles_detected:
                    vehicle_count = len(vehicles_detected)
                    avg_ratio = sum(v["ratio"] for v in vehicles_detected) / vehicle_count
                    if avg_ratio > 0.3:
                        _speak_rate_limited(last_said, "road_stop", "Stop! Vehicles detected on the road. Wait for them to pass.", 3.0)
                    else:
                        _speak_rate_limited(last_said, "road_cross", "Road crossing detected. Look both ways and cross when clear.", 4.0)
                else:
                    _speak_rate_limited(last_said, "road_clear", "Road crossing ahead. Path looks clear. You can cross.", 4.0)
            
            # Obstacle avoidance guidance (for outdoor or when not handled by indoor-specific logic)
            if environment != "indoor" and closest_obstacle is not None:
                if depth_alert_active:
                    closest_obstacle = None
                else:
                    ox_center = int((closest_obstacle["startX"] + closest_obstacle["endX"]) / 2)
                    ratio = closest_obstacle["ratio"]

                    # Determine preferred direction
                    preferred_dir = _analyze_path_clearance(frame, det_list)

                    if ratio > 0.55:
                        last_instruction = f"Stop! {closest_obstacle['label']} very close ahead. Move {preferred_dir} to avoid."
                        _speak_rate_limited(last_said, "stop", last_instruction, 2.0)
                    elif ratio > 0.35:
                        if preferred_dir == "left":
                            last_instruction = f"Move left to avoid the {closest_obstacle['label']} ahead."
                        elif preferred_dir == "right":
                            last_instruction = f"Move right to avoid the {closest_obstacle['label']} ahead."
                        else:
                            last_instruction = f"Careful, {closest_obstacle['label']} ahead. Continue slowly."
                        _speak_rate_limited(last_said, "avoid", last_instruction, 2.5)
                    else:
                        # Provide directional guidance based on route
                        if current_step_idx < len(route_steps):
                            step = route_steps[current_step_idx]
                            instruction = step.get("maneuver", {}).get("instruction", "")
                            if "left" in instruction.lower():
                                last_instruction = "Path clear. Prepare to turn left ahead."
                            elif "right" in instruction.lower():
                                last_instruction = "Path clear. Prepare to turn right ahead."
                            else:
                                last_instruction = "Path clear. Continue forward."
                        else:
                            last_instruction = "Path clear. Continue forward."
                        _speak_rate_limited(last_said, "clear", last_instruction, 3.0)
                    closest_obstacle = None

            if depth_alert_active:
                _speak_rate_limited(last_said, "depth_hold", "Let's stay put until the path is safe. Adjust the camera slightly and wait for my cue.", 8.0)
                continue

            if environment != "indoor" and closest_obstacle is not None:
                ox_center = int((closest_obstacle["startX"] + closest_obstacle["endX"]) / 2)
                ratio = closest_obstacle["ratio"]
                
                # Determine preferred direction
                preferred_dir = _analyze_path_clearance(frame, det_list)
                
                if ratio > 0.55:
                    last_instruction = f"Stop! {closest_obstacle['label']} very close ahead. Move {preferred_dir} to avoid."
                    _speak_rate_limited(last_said, "stop", last_instruction, 2.0)
                elif ratio > 0.35:
                    if preferred_dir == "left":
                        last_instruction = f"Move left to avoid the {closest_obstacle['label']} ahead."
                    elif preferred_dir == "right":
                        last_instruction = f"Move right to avoid the {closest_obstacle['label']} ahead."
                    else:
                        last_instruction = f"Careful, {closest_obstacle['label']} ahead. Continue slowly."
                    _speak_rate_limited(last_said, "avoid", last_instruction, 2.5)
                else:
                    # Provide directional guidance based on route
                    if current_step_idx < len(route_steps):
                        step = route_steps[current_step_idx]
                        instruction = step.get("maneuver", {}).get("instruction", "")
                        if "left" in instruction.lower():
                            last_instruction = "Path clear. Prepare to turn left ahead."
                        elif "right" in instruction.lower():
                            last_instruction = "Path clear. Prepare to turn right ahead."
                        else:
                            last_instruction = "Path clear. Continue forward."
                    else:
                        last_instruction = "Path clear. Continue forward."
                    _speak_rate_limited(last_said, "clear", last_instruction, 3.0)
            else:
                # No obstacles - provide route-based guidance
                if current_step_idx < len(route_steps):
                    step = route_steps[current_step_idx]
                    distance_m = step.get("distance", 0)
                    instruction = step.get("maneuver", {}).get("instruction", "Continue")
                    
                    if distance_m < 50:  # Close to turn
                        if "left" in instruction.lower():
                            last_instruction = "Turn left now."
                        elif "right" in instruction.lower():
                            last_instruction = "Turn right now."
                        else:
                            last_instruction = instruction
                        _speak_rate_limited(last_said, "turn", last_instruction, 2.0)
                        current_step_idx += 1
                    else:
                        last_instruction = f"Path clear. Continue forward. {instruction} in {int(distance_m)} meters."
                        _speak_rate_limited(last_said, "route", last_instruction, 4.0)
                else:
                    # Near destination
                    remaining_km = distance_remaining / 1000
                    if remaining_km < 0.1:
                        speak(f"You have arrived at {dest_name}. Navigation complete!")
                        break
                    else:
                        last_instruction = f"Path clear. Continue forward. Approximately {round(remaining_km, 2)} kilometers remaining."
                        _speak_rate_limited(last_said, "progress", last_instruction, 5.0)
            
            # Check for user commands continuously
            cmd = listen_for_speech(timeout=1, phrase_time_limit=2)
            if cmd:
                cmd_l = cmd.lower().strip()
                if is_exit_command(cmd):
                    speak("Stopping navigation.")
                    break
                if any(x in cmd_l for x in ["repeat", "again"]):
                    if last_instruction:
                        speak(last_instruction)
                if "ready" in cmd_l:
                    _speak_rate_limited(last_said, "ack", "Okay, proceeding.", 1.0)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        speak("Navigation stopped by user.")
    except Exception as e:
        print(f"[Navigation Error]: {e}")
        speak("An error occurred during navigation.")
    finally:
        try:
            cap.release()
        except:
            pass

def _speak_rate_limited(last_times, key, text, min_interval=2.0):
    """Rate limit speech to avoid spam."""
    now = time.time()
    last = last_times.get(key, 0)
    if now - last >= min_interval:
        speak(text)
        last_times[key] = now

# ===============================
# MAIN NAVIGATION MODE
# ===============================
def navigation_mode():
    """Blind navigation assistant with step-by-step guidance."""
    speak("Hello! I am your navigation assistant.")
    speak("Where would you like to go today?")
    destination_text = listen_for_speech()
    
    if not destination_text:
        speak("Sorry, I did not catch that.")
        return
    
    # Geocode destination
    dest_coords = get_coordinates(destination_text)
    if not dest_coords:
        speak("I couldn't find that place. Please try again.")
        return
    
    # Get location name from coordinates
    try:
        geo_url = "https://nominatim.openstreetmap.org/reverse"
        r = requests.get(
            geo_url,
            params={"lat": dest_coords[0], "lon": dest_coords[1], "format": "json"},
            headers={"User-Agent": "blind-nav"},
            timeout=4
        )
        dest_name = r.json().get("display_name", destination_text)
    except Exception as e:
        print("[Reverse Geocode Warning]", e)
        dest_name = destination_text
    
    speak(f"Destination {dest_name} found.")

    # Automatically detect current location
    start_coords = get_current_location()
    
    if not start_coords:
        # Final fallback: ask user only if auto-detection fails
        speak("I couldn't automatically detect your location. Please tell me your current address or say 'skip' to use approximate location.")
        start_text = listen_for_speech()
        
        if not start_text or "skip" in start_text.lower():
            speak("Using approximate location. Navigation may be less accurate.")
            # Use a default location (can be improved with GPS if available)
            start_coords = (40.748817, -73.985428)  # Fallback location
        else:
            # Try parsing as coordinates first
            coords = parse_location_coordinates(start_text)
            if coords:
                start_coords = coords
                speak(f"Using coordinates: {coords[0]}, {coords[1]}")
            else:
                # Try geocoding
                start_coords = get_coordinates(start_text)
                if not start_coords:
                    speak("I couldn't locate that address. Using approximate location for navigation.")
                    start_coords = (40.748817, -73.985428)  # Fallback location

    # Transport mode selection (including bus)
    speak("How would you like to travel? Say walking, driving, bike, or bus.")
    mode_input = listen_for_speech()
    
    if not mode_input:
        mode_input = "walking"
    
    mode_input = mode_input.lower()
    
    if "walk" in mode_input:
        mode = "foot-walking"
        mode_name = "walking"
    elif "drive" in mode_input or "car" in mode_input:
        mode = "driving-car"
        mode_name = "driving"
    elif "bike" in mode_input or "cycle" in mode_input:
        mode = "cycling-regular"
        mode_name = "cycling"
    elif "bus" in mode_input:
        mode = "driving-car"  # Use driving profile for bus routing
        mode_name = "bus"
        
        # Find nearest bus stop
        speak("Finding nearest bus stop...")
        bus_stop = find_nearby_bus_stops(start_coords[0], start_coords[1])
        if bus_stop:
            speak(f"Nearest bus stop is {bus_stop['name']}, {bus_stop['distance']} meters away.")
            # Optionally use bus stop as starting point
            # start_coords = (bus_stop['lat'], bus_stop['lon'])
        else:
            speak("No nearby bus stops found. Switching to walking mode.")
            mode = "foot-walking"
            mode_name = "walking"
    else:
        mode = "foot-walking"
        mode_name = "walking"

    speak(f"Calculating the best route to {dest_name} by {mode_name}...")

    route = calculate_route(start_coords, dest_coords, mode)
    if not route:
        speak("Sorry, I couldn't find a valid route.")
        return

    speak(f"Route ready! Distance is {route['distance']} kilometers.")
    speak(f"Estimated time is {route['duration']} minutes.")
    speak("Starting real-time camera navigation. I will guide you using the camera.")
    
    # Start real-time camera-based navigation
    _start_realtime_navigation(route, dest_name, dest_coords, start_coords)  

# ===================== MODEL LOADING =====================
def load_object_detection_model():
    """Load MobileNet-SSD if available; skip download if not."""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    PROTOTXT_PATH = os.path.join(models_dir, "MobileNetSSD_deploy.prototxt")
    MODEL_PATH = os.path.join(models_dir, "MobileNetSSD_deploy.caffemodel")

    # Skip download if both files are already there
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


# ===================== SPEECH INPUT =====================
def listen_for_speech(timeout=5, phrase_time_limit=8):
    """Listen from microphone and recognize using Google."""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("(Listening...)")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            text = recognizer.recognize_google(audio)
            print("[You said]:", text)
            return text
    except sr.UnknownValueError:
        print("[Recognition] Could not understand audio")
    except sr.RequestError as e:
        print(f"[Recognition] API Error: {e}")
    except Exception as e:
        print(f"[Mic Error] {e}")
    return None


# ===================== SYSTEM STARTUP =====================

# ===============================
# SYSTEM STATUS CHECK FUNCTIONS
# ===============================

def check_battery_status():
    """
    Check battery status and return formatted message.
    Returns: (status_message, percent, is_charging)
    """
    try:
        battery = psutil.sensors_battery()
        if battery is None:
            return "Battery status not available on this device.", None, None
        
        percent = int(battery.percent)
        charging = battery.power_plugged
        
        if charging:
            if percent >= 95:
                return f"Battery is {percent} percent and charging. Almost full.", percent, True
            else:
                return f"Battery is {percent} percent and charging.", percent, True
        else:
            if percent <= 20:
                return f"Battery is {percent} percent. Low battery warning. Please plug in your charger.", percent, False
            elif percent <= 50:
                return f"Battery is {percent} percent. Consider charging soon.", percent, False
            elif percent >= 90:
                return f"Battery is {percent} percent. Good charge level.", percent, False
            else:
                return f"Battery is {percent} percent.", percent, False
    except Exception as e:
        print(f"[Battery Check Error]: {e}")
        return "Could not read battery status.", None, None


def check_camera_status():
    """
    Check if camera is available and working.
    Returns: (status_message, is_available)
    """
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Camera is not available. Please check the connection.", False
        
        # Try to read a frame to verify camera works
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            h, w = frame.shape[:2]
            return f"Camera is connected and working. Resolution: {w} by {h} pixels.", True
        else:
            return "Camera is connected but cannot capture frames. Please check the camera.", False
    except Exception as e:
        print(f"[Camera Check Error]: {e}")
        return "Could not check camera status.", False


def check_internet_status():
    """
    Check internet connectivity.
    Returns: (status_message, is_connected, speed_indicator)
    """
    try:
        # Quick check with a small request
        import socket
        socket.setdefaulttimeout(3)
        
        # Try to connect to a reliable server
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            socket.create_connection(("1.1.1.1", 53), timeout=3)
        except OSError:
            return "No internet connection detected. Some features may be limited.", False, "none"
        
        # Try HTTP request to check actual connectivity
        try:
            response = requests.get("https://www.google.com", timeout=3)
            if response.status_code == 200:
                return "Internet connection is active and working.", True, "good"
            else:
                return "Internet connection is active but may be slow.", True, "slow"
        except requests.exceptions.Timeout:
            return "Internet connection is slow or unstable.", True, "slow"
        except requests.exceptions.RequestException:
            return "Internet connection is active but some services may be unavailable.", True, "limited"
    except Exception as e:
        print(f"[Internet Check Error]: {e}")
        return "Could not check internet status.", False, "unknown"


def check_system_status_parallel():
    """
    Check all system statuses in parallel and announce them.
    Runs during startup while models are loading.
    """
    print("[System Status] Starting parallel status checks...")
    
    # Run checks in parallel using threading
    results = {}
    threads = []
    
    def check_battery_thread():
        results['battery'] = check_battery_status()
    
    def check_camera_thread():
        results['camera'] = check_camera_status()
    
    def check_internet_thread():
        results['internet'] = check_internet_status()
    
    # Start all checks
    t1 = threading.Thread(target=check_battery_thread, daemon=True)
    t2 = threading.Thread(target=check_camera_thread, daemon=True)
    t3 = threading.Thread(target=check_internet_thread, daemon=True)
    
    threads = [t1, t2, t3]
    for t in threads:
        t.start()
    
    # Wait for all to complete (with timeout)
    for t in threads:
        t.join(timeout=5)
    
    # Announce results
    speak("Checking system status...")
    time.sleep(0.3)
    
    # Battery status
    if 'battery' in results:
        battery_msg, percent, charging = results['battery']
        speak(battery_msg)
        time.sleep(0.2)
    
    # Camera status
    if 'camera' in results:
        camera_msg, is_available = results['camera']
        speak(camera_msg)
        time.sleep(0.2)
    
    # Internet status
    if 'internet' in results:
        internet_msg, is_connected, speed = results['internet']
        speak(internet_msg)
        time.sleep(0.2)
    
    return results


def monitor_battery_continuous(stop_event):
    """
    Continuously monitor battery status in background.
    Announces updates when battery level changes significantly.
    """
    last_percent = None
    last_charging = None
    
    while not stop_event.is_set():
        try:
            battery = psutil.sensors_battery()
            if battery:
                current_percent = int(battery.percent)
                current_charging = battery.power_plugged
                
                # Announce if significant change
                if last_percent is not None:
                    # Battery dropped significantly
                    if current_percent < last_percent - 5:
                        speak(f"Battery level dropped to {current_percent} percent.")
                    # Battery increased significantly (charging)
                    elif current_percent > last_percent + 5 and current_charging:
                        speak(f"Battery is now {current_percent} percent.")
                    # Low battery warning
                    elif current_percent <= 20 and last_percent > 20:
                        speak(f"Warning: Battery is low at {current_percent} percent. Please charge soon.")
                    # Critical battery
                    elif current_percent <= 10 and last_percent > 10:
                        speak(f"Critical: Battery is at {current_percent} percent. Please charge immediately.")
                
                last_percent = current_percent
                last_charging = current_charging
        except Exception as e:
            print(f"[Battery Monitor Error]: {e}")
        
        # Check every 30 seconds
        stop_event.wait(30)


def system_boot():
    """Play startup tone, check system status, and load models in parallel."""
    print("SYSTEM: Starting system boot sequence...")
    winsound.Beep(800, 300)
    time.sleep(0.2)
    
    speak("Initializing system. Please wait while I check everything.")
    
    # Start model loading in background threads (parallel with status checks)
    global blip_loading_thread, easyocr_loading_thread, battery_monitor_stop_event
    
    blip_loading_thread = threading.Thread(target=load_blip_model_async, daemon=True)
    easyocr_loading_thread = threading.Thread(target=load_easyocr_async, daemon=True)
    
    blip_loading_thread.start()
    easyocr_loading_thread.start()
    
    # Check system status in parallel (this will speak the status)
    # This runs while models are loading in the background
    print("[System Boot] Starting system status checks...")
    status_results = check_system_status_parallel()
    
    # Start continuous battery monitoring in background
    battery_monitor_stop_event = threading.Event()
    battery_monitor_thread = threading.Thread(
        target=monitor_battery_continuous, 
        args=(battery_monitor_stop_event,), 
        daemon=True
    )
    battery_monitor_thread.start()
    
    # Wait a moment for initial status announcements to complete
    time.sleep(0.5)
    
    # Wait for models to finish loading (with timeout)
    # Don't block indefinitely - allow system to continue even if models take longer
    print("[System Boot] Waiting for models to load...")
    blip_loading_thread.join(timeout=30)
    easyocr_loading_thread.join(timeout=30)
    
    # Check if models loaded successfully
    if processor is not None and model_blip is not None:
        print("[System Boot] BLIP model loaded successfully.")
    if easyocr_reader is not None:
        print("[System Boot] EasyOCR model loaded successfully.")
    
    speak("System ready. Welcome to BlindNav Plus, your personal navigation assistant.")
    speak("Please tell me your preferred language for our conversation. You can say English, Kannada, or Hindi. Just speak your choice now.")

    user_lang = "en"
    attempts = 0
    while attempts < 3:
        text = listen_for_speech()
        if text:
            try:
                user_lang = langdetect.detect(text)
                speak(f"Language detected as {user_lang}.")
                break
            except:
                speak("Sorry, I could not detect your language.")
        else:
            speak("I didn’t catch that. Please say your language again.")
        attempts += 1

    # Personalized, friendly, and confidence-boosting voice guide

    # Ask for user's name
    global user_name
    speak("Hello there! Before we begin, may I know your name?")
    name_input = listen_for_speech(timeout=5, phrase_time_limit=5)
    if not name_input:
        user_name = "friend"
        speak("Nice to meet you! I'll call you friend.")
    else:
        user_name = name_input.strip()

    speak(f"Nice to meet you, {user_name}! I'm BlindNav Plus, your personal navigation assistant. I'm here to help you navigate safely, read text, and assist you in your daily activities.")
    time.sleep(0.5)
    
    # Main features overview
    speak(f"{user_name}, let me explain how to use me. I have several modes that can help you:")
    time.sleep(0.3)
    
    speak("First, Navigation Mode. Say 'Start navigation mode' or 'Navigation mode' to get step-by-step directions to any destination. I'll guide you using your camera and GPS, helping you avoid obstacles and find your way safely.")
    time.sleep(0.5)
    
    speak("Second, Assistant Mode. Say 'Assistant mode' to ask me questions, check the time, weather, set reminders, or find nearby places like bus stops. I can help with everyday tasks.")
    time.sleep(0.5)
    
    speak("Third, Reading Mode. Say 'Reading mode' to have me read any printed or handwritten text through your camera. Just point the camera at the text, and I'll read it aloud to you.")
    time.sleep(0.5)
    
    speak("Fourth, Object Detection Mode. Say 'Object detection mode' to have me describe what's around you. I'll tell you about objects, people, doors, walls, and obstacles in your path.")
    time.sleep(0.5)
    
    speak("Fifth, Walking Mode. Say 'Walking mode' for real-time guidance while walking. I'll continuously monitor your path and warn you about obstacles, doors, and walls ahead.")
    time.sleep(0.5)
    
    # Emergency features
    speak(f"Important safety features, {user_name}:")
    time.sleep(0.3)
    
    speak("If you need emergency help, say 'Emergency mode' or 'Help me' and I'll activate emergency assistance immediately.")
    time.sleep(0.3)
    
    speak("For medical emergencies, say 'Medical mode' or 'Call medical service'.")
    time.sleep(0.3)
    
    speak("For police assistance, say 'Police mode' or 'Call police service'.")
    time.sleep(0.3)
    
    speak("For fire emergencies, say 'Fire mode' or 'Call fire service'.")
    time.sleep(0.5)
    
    # How to use tips
    speak(f"Here are some helpful tips for using me, {user_name}:")
    time.sleep(0.3)
    
    speak("Always keep your camera facing forward when using navigation or walking modes. This helps me see obstacles and guide you better.")
    time.sleep(0.5)
    
    speak("Speak clearly and wait for my response. If I don't understand, just repeat your request.")
    time.sleep(0.5)
    
    speak("To exit any mode, say 'Exit mode' or 'Stop mode'. To return to the main menu, say 'Exit' or 'Go back'.")
    time.sleep(0.5)
    
    speak("You can ask me to find nearby places. For example, say 'Find nearest bus stop' or 'Where is the nearest restaurant'.")
    time.sleep(0.5)
    
    speak("I'll continuously monitor your battery level and let you know if it gets low. Make sure your device is charged for the best experience.")
    time.sleep(0.5)
    
    # Final encouragement
    speak(f"You're all set, {user_name}! I'm here to help you navigate confidently and safely. Just tell me what you need, and I'll guide you. What would you like to do first?")
    time.sleep(0.3)

    select_mode()



# ===================== MODES =====================
def assistant_mode():
    speak("How can I assist you today?")
    while True:
        cmd = listen_for_speech()
        if not cmd:
            speak("I didn't catch that. Say exit to leave assistant mode.")
            continue
        cmd = cmd.lower()
        if "exit" in cmd:
            speak("Exiting assistant mode.")   
            break
        else:
            speak(f"You said: {cmd}. This is a placeholder response.")


# Removed duplicate navigation_mode - using the enhanced version above

import speech_recognition as sr

def wait_for_next(last_instruction, timeout=5, phrase_time_limit=8):
    """
    Waits for the user to say 'next' to proceed or 'repeat' to replay the last instruction.
    Uses Google Speech Recognition for voice input.
    Keeps listening until a valid response is received.
    """
    recognizer = sr.Recognizer()

    while True:
        try:
            with sr.Microphone() as source:
                print("(Listening...)")
                speak("Please say 'next' to continue or 'repeat' to hear it again.")
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            # Try recognizing speech
            user_input = recognizer.recognize_google(audio)
            user_input = user_input.lower().strip()
            print("[You said]:", user_input)

            # Handle user responses
            if any(word in user_input for word in ["next", "continue", "proceed", "go ahead", "okay", "done"]):
                speak("Okay, moving on.")
                break
            elif "repeat" in user_input or "say again" in user_input:
                speak("Repeating the last instruction.")
                speak(last_instruction)
            else:
                speak("Please say 'next' to continue or 'repeat' to hear it again.")
        
        except sr.WaitTimeoutError:
            speak("I didn't hear anything. Please say 'next' or 'repeat'.")
            continue
        except sr.UnknownValueError:
            speak("Sorry, I didn't catch that. Please say 'next' or 'repeat'.")
            continue
        except sr.RequestError:
            speak("There was a problem connecting to the speech service. Please try again.")
            break



def object_detection_mode(net):
    speak("Object detection mode activated. Scanning your surroundings.")      
    if net is None:
        speak("Model not available. Running in placeholder mode.")
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not accessible. Cannot start object detection.")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        speak("Could not capture frame from camera.")
        return

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    class_names = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    found = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = class_names[idx] if idx < len(class_names) else "object"
            found.append(label)

    if found:
        speak(f"I detected: {', '.join(found[:3])}.")
    else:
        speak("No prominent objects detected.")


# ===============================
# SMART CAMERA NAVIGATION SYSTEM
# ===============================

def start_smart_camera_navigation():
    """
    Main entry point for smart camera navigation.
    Analyzes environment and starts appropriate navigation mode.
    """
    speak("Turning on camera and analyzing your environment...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not available. Please check the connection.")
        return
    
    # Capture and analyze initial frame
    ret, frame = cap.read()
    if not ret:
        speak("Could not capture camera feed.")
        cap.release()
        return
    
    # Detect objects using YOLO
    try:
        results = local_model(frame, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.5:
                    obj_name = local_model.names[cls]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({
                        "label": obj_name,
                        "confidence": conf,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })
    except Exception as e:
        print(f"[Detection Error]: {e}")
        detections = []
    
    # Detect environment (indoor/outdoor)
    environment = detect_environment_enhanced(frame, detections)
    
    # Check for doors and walls before releasing camera
    door_result = detect_door(frame)
    wall_result = detect_wall(frame)
    
    # Announce detected doors and walls
    if door_result['detected']:
        speak(f"I can see a door {door_result['distance']} on your {door_result['position']}.")
    if wall_result['detected']:
        speak(f"I can see a wall {wall_result['distance']} {wall_result['direction']}.")
    
    cap.release()
    
    if environment == "indoor":
        speak("I can see you're indoors. Starting indoor navigation mode.")
        indoor_navigation_mode()
    elif environment == "outdoor":
        speak("I can see you're outdoors. Starting outdoor navigation mode.")
        outdoor_navigation_mode()
    else:
        speak("I'm not sure if you're indoors or outdoors. Let me ask you a quick question.")
        user_response = listen_for_speech(timeout=5, phrase_time_limit=5)
        if user_response and ("indoor" in user_response.lower() or "inside" in user_response.lower() or "room" in user_response.lower()):
            speak("Starting indoor navigation mode.")
            indoor_navigation_mode()
        else:
            speak("Starting outdoor navigation mode.")
            outdoor_navigation_mode()


def detect_environment_enhanced(frame, detections):
    """
    Enhanced indoor/outdoor detection using multiple indicators.
    """
    h, w = frame.shape[:2]
    
    # Check for outdoor indicators
    outdoor_indicators = ["car", "bus", "motorbike", "bicycle", "train", "traffic light", "road", "sidewalk"]
    vehicle_count = sum(1 for det in detections if det.get("label") in outdoor_indicators)
    
    # Analyze brightness and color distribution
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    std_dev = np.std(gray)
    
    # Indoor indicators
    indoor_indicators = ["chair", "sofa", "diningtable", "tvmonitor", "bed", "table", "laptop", "keyboard", "mouse"]
    indoor_count = sum(1 for det in detections if det.get("label") in indoor_indicators)
    
    # Check for ceiling/walls (indoor) vs sky (outdoor)
    top_region = frame[0:int(h*0.3), :]
    top_brightness = np.mean(cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY))
    
    # Decision logic
    if vehicle_count >= 2 or (brightness > 120 and std_dev > 50 and vehicle_count >= 1):
        return "outdoor"
    elif indoor_count >= 2 or (brightness < 90 and std_dev < 35):
        return "indoor"
    elif top_brightness > 150:  # Bright top region suggests sky (outdoor)
        return "outdoor"
    elif indoor_count >= 1:
        return "indoor"
    else:
        return "unknown"


def indoor_navigation_mode():
    """
    Indoor navigation mode with interactive questions and real-time guidance.
    Analyzes user position and asks what they want to do.
    """
    speak("Welcome to indoor navigation mode. I'll help you navigate inside.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not available.")
        return
    
    # Analyze current position
    speak("Let me analyze your current position...")
    ret, frame = cap.read()
    if ret:
        position_info = analyze_indoor_position(frame)
        speak(position_info)
        
        # Check for doors and walls
        door_result = detect_door(frame)
        wall_result = detect_wall(frame)
        
        if door_result['detected']:
            speak(f"I can see a door {door_result['distance']} on your {door_result['position']}.")
        if wall_result['detected']:
            speak(f"I can see a wall {wall_result['distance']} {wall_result['direction']}.")
    
    # Ask user what they want to do
    speak("What would you like to do? You can say: get out, move downstairs, move upstairs, move out of the room, go to seating space, or any other destination.")
    
    user_intent = listen_for_speech(timeout=10, phrase_time_limit=8)
    if not user_intent:
        speak("I didn't catch that. Let me start general indoor guidance.")
        user_intent = "general"
    
    user_intent = user_intent.lower()
    
    # Start real-time guidance based on intent
    if "out" in user_intent or "exit" in user_intent or "leave" in user_intent:
        guide_to_exit(cap)
    elif "down" in user_intent or "downstairs" in user_intent:
        guide_to_stairs(cap, direction="down")
    elif "up" in user_intent or "upstairs" in user_intent:
        guide_to_stairs(cap, direction="up")
    elif "room" in user_intent or "door" in user_intent:
        guide_to_door(cap)
    elif "seat" in user_intent or "sit" in user_intent or "chair" in user_intent:
        guide_to_seating(cap)
    else:
        general_indoor_guidance(cap)
    
    cap.release()


def analyze_indoor_position(frame):
    """
    Analyze user's current position in indoor space.
    """
    try:
        results = local_model(frame, verbose=False)
        detected_objects = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.5:
                    obj_name = local_model.names[cls]
                    detected_objects.append(obj_name)
        
        detected_objects = list(set(detected_objects))
        
        # Analyze position context
        if "door" in detected_objects:
            return "I can see you're near a door."
        elif "chair" in detected_objects or "sofa" in detected_objects:
            return "I can see seating furniture nearby."
        elif "table" in detected_objects or "diningtable" in detected_objects:
            return "I can see a table nearby."
        elif "bed" in detected_objects:
            return "I can see a bed nearby."
        else:
            return f"I can see {', '.join(detected_objects[:3]) if detected_objects else 'various objects'} around you."
    except Exception as e:
        print(f"[Position Analysis Error]: {e}")
        return "I'm analyzing your surroundings."


def guide_to_exit(cap):
    """Guide user to exit/leave the building."""
    speak("I'll help you find the exit. Let's start moving.")
    
    last_guidance_time = 0
    door_found = False
    last_door_announcement = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect doors using enhanced detection
        door_result = detect_door(frame)
        door_detected = door_result['detected'] and door_result['confidence'] > 0.4
        door_position = door_result['position']
        
        # Also check YOLO model as backup
        if not door_detected:
            results = local_model(frame, verbose=False)
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf > 0.5:
                        obj_name = local_model.names[cls]
                        if "door" in obj_name.lower():
                            door_detected = True
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            h, w = frame.shape[:2]
                            door_center_x = (x1 + x2) / 2
                            door_position = "left" if door_center_x < w/3 else "right" if door_center_x > 2*w/3 else "center"
                            break
        
        current_time = time.time()
        if door_detected and not door_found:
            if current_time - last_door_announcement > 3:
                speak(f"I found a door {door_result.get('distance', '')} on your {door_position}. Let me guide you to it.")
                door_found = True
                last_door_announcement = current_time
        
        if door_found:
            # Guide to door handle using pose detection
            guide_to_door_handle(cap, frame)
            break
        
        # Also check for walls and warn user
        wall_result = detect_wall(frame)
        if wall_result['detected'] and wall_result['distance'] in ['very close', 'close']:
            if current_time - last_door_announcement > 4:
                speak(f"Caution: Wall detected {wall_result['distance']} {wall_result['direction']}. Adjust your path.")
                last_door_announcement = current_time
        
        # General guidance
        if time.time() - last_guidance_time > 3:
            speak("Keep moving forward. I'm looking for an exit door.")
            last_guidance_time = time.time()
        
        # Check for exit command
        cmd = listen_for_speech(timeout=1, phrase_time_limit=2)
        if cmd and ("stop" in cmd.lower() or "exit" in cmd.lower() or "done" in cmd.lower()):
            speak("Stopping exit guidance.")
            break
        
        time.sleep(0.1)


def guide_to_stairs(cap, direction="up"):
    """Guide user to stairs and help them climb/descend."""
    direction_text = "upstairs" if direction == "up" else "downstairs"
    speak(f"I'll help you go {direction_text}. Let's find the stairs first.")
    
    stairs_found = False
    last_guidance_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect stairs (look for step-like patterns or handrails)
        # This is a simplified detection - in practice, you'd use more sophisticated methods
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 85 < angle < 95:  # Horizontal lines (steps)
                    horizontal_lines += 1
        
        if horizontal_lines > 5 and not stairs_found:
            speak("I can see stairs ahead. Let me guide you to them.")
            stairs_found = True
        
        if stairs_found:
            # Guide climbing/descending with pose detection
            guide_stair_climbing(cap, frame, direction)
            break
        
        # General guidance
        if time.time() - last_guidance_time > 3:
            speak(f"Keep moving forward. I'm looking for stairs to go {direction_text}.")
            last_guidance_time = time.time()
        
        # Check for exit command
        cmd = listen_for_speech(timeout=1, phrase_time_limit=2)
        if cmd and ("stop" in cmd.lower() or "done" in cmd.lower()):
            speak("Stopping stair guidance.")
            break
        
        time.sleep(0.1)


def guide_to_door(cap):
    """Guide user to a door."""
    speak("I'll help you find and open a door.")
    guide_to_exit(cap)  # Reuse exit guidance logic


def guide_to_seating(cap):
    """Guide user to seating area."""
    speak("I'll help you find a place to sit.")
    
    last_guidance_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect chairs/sofas
        results = local_model(frame, verbose=False)
        seating_found = False
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.5:
                    obj_name = local_model.names[cls]
                    if obj_name in ["chair", "sofa", "couch"]:
                        speak(f"I found a {obj_name} ahead. Let me guide you to it.")
                        seating_found = True
                        break
        
        if seating_found:
            speak("You're near the seating. Move forward slowly and I'll tell you when to sit.")
            time.sleep(2)
            speak("You can sit down now.")
            break
        
        # General guidance
        if time.time() - last_guidance_time > 3:
            speak("Keep moving forward. I'm looking for seating.")
            last_guidance_time = time.time()
        
        # Check for exit command
        cmd = listen_for_speech(timeout=1, phrase_time_limit=2)
        if cmd and ("stop" in cmd.lower() or "done" in cmd.lower()):
            speak("Stopping seating guidance.")
            break
        
        time.sleep(0.1)


def guide_to_door_handle(cap, frame):
    """
    Guide user to door handle using pose detection and natural language.
    """
    if not MEDIAPIPE_AVAILABLE:
        speak("Move your hand forward and I'll guide you to the door handle.")
        return
    
    speak("I found the door. Now let me help you find the handle. Please extend your hand forward.")
    
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    
    handle_detected = False
    last_guidance_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect door handle (simplified - look for door knob area)
        # In practice, you'd use object detection for handles
        frame_h, frame_w = frame.shape[:2]
        
        # Detect hand position
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        pose_results = pose.process(rgb_frame)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Get wrist position (landmark 0)
                wrist = hand_landmarks.landmark[0]
                wrist_x = int(wrist.x * frame_w)
                wrist_y = int(wrist.y * frame_h)
                
                # Estimate handle position (typically at door center, around 1-1.5m height)
                # This is simplified - in practice, detect actual handle
                handle_estimated_x = frame_w // 2
                handle_estimated_y = int(frame_h * 0.6)  # Middle height of door
                
                # Calculate relative position
                dx = handle_estimated_x - wrist_x
                dy = handle_estimated_y - wrist_y
                
                # Provide natural language guidance
                if abs(dx) < 50 and abs(dy) < 50:
                    speak("Perfect! Your hand is at the handle. You can grab it now.")
                    handle_detected = True
                    break
                else:
                    if time.time() - last_guidance_time > 2:
                        if abs(dx) > abs(dy):
                            direction = "left" if dx < 0 else "right"
                            speak(f"Move your hand {direction} a bit.")
                        else:
                            direction = "up" if dy < 0 else "down"
                            speak(f"Move your hand {direction} slightly.")
                        last_guidance_time = time.time()
        
        if handle_detected:
            speak("Great! Now turn the handle and push or pull the door.")
            time.sleep(3)
            break
        
        # Check for exit command
        cmd = listen_for_speech(timeout=1, phrase_time_limit=2)
        if cmd and ("stop" in cmd.lower() or "done" in cmd.lower()):
            speak("Stopping door handle guidance.")
            break
        
        time.sleep(0.1)
    
    hands.close()
    pose.close()


def guide_stair_climbing(cap, frame, direction):
    """
    Guide user to climb or descend stairs with real-time posture detection.
    """
    speak(f"I'll guide you to go {direction}. Please face the stairs.")
    
    if not MEDIAPIPE_AVAILABLE:
        speak(f"Start {direction} the stairs. I'll count the steps for you.")
        return
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    
    step_count = 0
    last_guidance_time = 0
    last_knee_y = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)
        
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            h, w = frame.shape[:2]
            
            # Get key points
            left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
            left_ankle = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
            
            # Calculate average knee and ankle positions
            knee_y = (left_knee.y + right_knee.y) / 2 * h
            ankle_y = (left_ankle.y + right_ankle.y) / 2 * h
            
            # Detect step (knee movement up/down)
            if last_knee_y is not None:
                if direction == "up":
                    if knee_y < last_knee_y - 20:  # Knee moved up
                        step_count += 1
                        speak(f"Step {step_count}. Keep going up.")
                        last_knee_y = knee_y
                else:  # down
                    if knee_y > last_knee_y + 20:  # Knee moved down
                        step_count += 1
                        speak(f"Step {step_count}. Keep going down.")
                        last_knee_y = knee_y
            else:
                last_knee_y = knee_y
            
            # Provide guidance
            if time.time() - last_guidance_time > 3:
                if direction == "up":
                    speak("Lift your foot and place it on the next step.")
                else:
                    speak("Lower your foot to the next step down.")
                last_guidance_time = time.time()
        
        # Check for completion or exit
        cmd = listen_for_speech(timeout=1, phrase_time_limit=2)
        if cmd:
            if "done" in cmd.lower() or "finished" in cmd.lower() or "stop" in cmd.lower():
                speak(f"Great job! You've completed {step_count} steps.")
                break
            elif "help" in cmd.lower():
                speak("Take your time. Move one step at a time. I'm here to guide you.")
        
        time.sleep(0.1)
    
    pose.close()


def general_indoor_guidance(cap):
    """General indoor guidance mode."""
    speak("I'll provide general indoor guidance. Tell me what you need help with.")
    
    last_guidance_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect obstacles
        results = local_model(frame, verbose=False)
        obstacles = []
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.5:
                    obj_name = local_model.names[cls]
                    if obj_name in ["person", "chair", "table", "sofa"]:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        h, w = frame.shape[:2]
                        obj_center_x = (x1 + x2) / 2
                        position = "left" if obj_center_x < w/3 else "right" if obj_center_x > 2*w/3 else "ahead"
                        obstacles.append(f"{obj_name} {position}")
        
        if obstacles and time.time() - last_guidance_time > 3:
            speak(f"I see {', '.join(obstacles[:2])}. Be careful.")
            last_guidance_time = time.time()
        
        # Check for user commands
        cmd = listen_for_speech(timeout=2, phrase_time_limit=3)
        if cmd:
            cmd_l = cmd.lower()
            if "stop" in cmd_l or "exit" in cmd_l or "done" in cmd_l:
                speak("Stopping indoor guidance.")
                break
            elif "help" in cmd_l:
                speak("I'm here to help. Tell me what you need, or I can guide you to exits, stairs, or seating.")
        
        time.sleep(0.1)


def outdoor_navigation_mode():
    """
    Outdoor navigation mode for navigating outside.
    """
    speak("Welcome to outdoor navigation mode. I'll help you navigate safely outside.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not available.")
        return
    
    # Use existing outdoor navigation from _start_realtime_navigation
    speak("Starting outdoor navigation. I'll guide you using real-time camera analysis.")
    
    # This can reuse the existing outdoor navigation logic
    # For now, provide basic outdoor guidance
    last_guidance_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect obstacles and provide guidance
        results = local_model(frame, verbose=False)
        obstacles = []
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.5:
                    obj_name = local_model.names[cls]
                    if obj_name in ["person", "car", "bus", "bicycle", "motorbike"]:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        h, w = frame.shape[:2]
                        obj_center_x = (x1 + x2) / 2
                        position = "left" if obj_center_x < w/3 else "right" if obj_center_x > 2*w/3 else "ahead"
                        obstacles.append(f"{obj_name} {position}")
        
        if obstacles and time.time() - last_guidance_time > 3:
            speak(f"Be careful, there's {', '.join(obstacles[:2])} nearby.")
            last_guidance_time = time.time()
        elif time.time() - last_guidance_time > 5:
            speak("Path looks clear. Continue forward.")
            last_guidance_time = time.time()
        
        # Check for user commands
        cmd = listen_for_speech(timeout=2, phrase_time_limit=3)
        if cmd:
            cmd_l = cmd.lower()
            if "stop" in cmd_l or "exit" in cmd_l or "done" in cmd_l:
                speak("Stopping outdoor navigation.")
                break
        
        time.sleep(0.1)
    
    cap.release()


def select_mode():
    speak("Now, please select the mode you want to continue in.")
    speak("Assistant mode lets you talk to me naturally, ask questions, and get everyday help.")
    speak("Navigation mode guides you step by step to your destination using real-time camera vision.")
    speak("Object detection mode describes your surroundings and identifies objects through the camera.")
    speak("Walking mode helps you move safely by detecting obstacles and giving directional cues.")
    speak("Emergency mode listens for distress or danger and connects you to emergency assistance if needed.")
    speak("Reading mode reads aloud any printed or handwritten text visible to the camera.")
    speak("you can choose from: assistant mode, navigation mode, object detection mode, walking mode, emergency mode, reading mode")

    # Load object detection model only once
    # net = load_object_detection_model()

    # Mode selection
    while True:
        mode_input = listen_for_speech()
        if not mode_input:
            speak("I didn’t catch that. Please say the mode name again.")
            continue

        mode_input = mode_input.lower()
        mode_input = mode_input.lower().strip()
        mode_input = mode_input.replace("model", "mode").replace("emmergency", "emergency")

        if any(
            phrase in mode_input
            for phrase in [
                "repeat available",
                "repeat the available",
                "repeat modes",
                "repeat the modes",
                "list modes",
                "what modes",
                "available modes",
                "state the available",
                "tell me the modes",
                "say the modes",
            ]
        ):
            speak_available_modes_overview()
            continue

        if any(
            phrase in mode_input
            for phrase in [
                "indoor navigation",
                "indoor mode",
                "smart navigation",
                "guide me out",
                "take me outside",
                "need to go out",
                "go outside",
                "go out",
            ]
        ):
            speak("I'll turn on the camera and guide you.")
            start_smart_camera_navigation()
            continue

        # ---------------- TURN ON CAMERA / SMART NAVIGATION ----------------
        if (
            "turn on camera" in mode_input or
            "turn on the camera" in mode_input or
            "start camera" in mode_input or
            "activate camera" in mode_input or
            "open camera" in mode_input or
            "enable camera" in mode_input or
            "camera on" in mode_input
        ):
            speak("Turning on camera and analyzing your surroundings...")
            start_smart_camera_navigation()
            continue

        # ---------------- ASSISTANT MODE ----------------
        if (
            "assistant" in mode_input or
            "assistant mode" in mode_input or
            "switch to assistant" in mode_input or
            "activate assistant" in mode_input or
            "start assistant" in mode_input or
            "run assistant mode" in mode_input or
            "go to assistant mode" in mode_input or
            "enable assistant" in mode_input or
            "enter assistant mode" in mode_input or
            "open assistant" in mode_input or
            "return to assistant mode" in mode_input or
            "be my assistant" in mode_input
        ):
            speak("Assistant mode activated. How can I assist you today?")
            run_assistant_mode()

        # ---------------- NAVIGATION MODE ----------------
        elif (
            "navigation" in mode_input or
            "navigation mode" in mode_input or
            "switch to navigation" in mode_input or
            "start navigation" in mode_input or
            "activate navigation mode" in mode_input or
            "enable navigation" in mode_input or
            "open navigation mode" in mode_input or
            "turn on navigation" in mode_input or
            "help me navigate" in mode_input or
            "i want directions" in mode_input or
            "guide me to my destination" in mode_input or
            "show me the way" in mode_input
        ):
            speak("Navigation mode activated. I can help you find directions or locations.")
            navigation_mode()

        # ---------------- OBJECT DETECTION MODE ----------------
        elif (
            "object detection" in mode_input or
            "detection mode" in mode_input or
            "start object detection" in mode_input or
            "detect objects" in mode_input or
            "identify objects" in mode_input or
            "run object mode" in mode_input or
            "switch to detection mode" in mode_input or
            "analyze my surroundings" in mode_input or
            "recognize objects" in mode_input or
            "what's around me" in mode_input or
            "object mode" in mode_input
        ):
            net = load_object_detection_model()
            speak("Object detection mode activated. Scanning your surroundings.")
            run_object_detection_mode()

        # ---------------- WALKING MODE ----------------
        elif (
            "walking" in mode_input or
            "walking mode" in mode_input or
            "switch to walking" in mode_input or
            "start walking mode" in mode_input or
            "enable walking mode" in mode_input or
            "help me while walking" in mode_input or
            "guide me as i walk" in mode_input or
            "turn on walking" in mode_input or
            "assist me on the road" in mode_input or
            "i'm going for a walk" in mode_input
        ):
            speak("Walking mode activated. I can help you while walking.")
            run_walking_mode()

        # ---------------- EMERGENCY MODE ----------------
        elif (
            "emergency" in mode_input or
            "emergency mode" in mode_input or
            "help" in mode_input or
            "help me" in mode_input or
            "please help" in mode_input or
            "i need help" in mode_input or
            "save me" in mode_input or
            "sos" in mode_input or
            "someone help" in mode_input or
            "anyone help" in mode_input or
            "can you hear me" in mode_input or
            "this is an emergency" in mode_input or
            "it's an emergency" in mode_input or
            "activate emergency mode" in mode_input or
            "trigger emergency" in mode_input or
            "call emergency" in mode_input or
            "call for help" in mode_input or
            "send help" in mode_input or
            "guide this is an emergency" in mode_input or
            "start emergency" in mode_input or
            "emergency please" in mode_input or

            # Physical danger or injury
            "i'm hurt" in mode_input or
            "i am hurt" in mode_input or
            "i'm injured" in mode_input or
            "i am injured" in mode_input or
            "i'm bleeding" in mode_input or
            "accident" in mode_input or
            "there's been an accident" in mode_input or
            "someone is hurt" in mode_input or
            "someone's hurt" in mode_input or
            "attack" in mode_input or
            "attacked" in mode_input or
            "i’m being attacked" in mode_input or
            "there’s an attack" in mode_input or
            "help me please" in mode_input or

            # Fear or panic phrases
            "oh my god" in mode_input or
            "please god" in mode_input or
            "no no no" in mode_input or
            "stop it" in mode_input or
            "stop please" in mode_input or
            "leave me alone" in mode_input or
            "let me go" in mode_input or
            "don’t hurt me" in mode_input or
            "don't hurt me" in mode_input or
            "i’m scared" in mode_input or
            "i’m afraid" in mode_input or
            "i’m in danger" in mode_input or
            "they’re hurting me" in mode_input or
            "they are hurting me" in mode_input or
            "get away" in mode_input or
            "go away" in mode_input or
            "run" in mode_input or
            "fire" in mode_input or
            "there’s fire" in mode_input or
            "flood" in mode_input or
            "earthquake" in mode_input or
            "explosion" in mode_input or
            "collapse" in mode_input or
            "bomb" in mode_input or
            "shooting" in mode_input or

            # Medical emergencies
            "i can’t breathe" in mode_input or
            "cant breathe" in mode_input or
            "heart attack" in mode_input or
            "chest pain" in mode_input or
            "stroke" in mode_input or
            "seizure" in mode_input or
            "i need a doctor" in mode_input or
            "call a doctor" in mode_input or
            "call ambulance" in mode_input or
            "ambulance" in mode_input or
            "medical emergency" in mode_input or
            "unconscious" in mode_input or
            "someone fainted" in mode_input or

            # Urgency / police calls
            "help us" in mode_input or
            "help now" in mode_input or
            "help me now" in mode_input or
            "police" in mode_input or
            "call police" in mode_input or
            "call the cops" in mode_input or
            "call 911" in mode_input or
            "contact emergency" in mode_input or
            "alert emergency" in mode_input or
            "emergency help" in mode_input or
            "please hurry" in mode_input or
            "hurry up" in mode_input or
            "danger" in mode_input or
            "mayday" in mode_input or
            "code red" in mode_input
        ):
            speak("Emergency mode activated.")
            run_emergency_mode()


        # ---------------- SECURITY MODE ----------------
        elif (
            "security" in mode_input or
            "security mode" in mode_input or
            "activate security" in mode_input or
            "enable security mode" in mode_input or
            "turn on security" in mode_input or
            "guard mode" in mode_input or
            "monitor surroundings" in mode_input or
            "run security mode" in mode_input or
            "start surveillance" in mode_input or
            "activate security system" in mode_input
        ):
            speak("Security mode activated.")  
            security_mode()

        # ---------------- MEDICAL MODE ----------------
        elif (
            "medical" in mode_input or
            "medical mode" in mode_input or
            "switch to medical mode" in mode_input or
            "activate medical help" in mode_input or
            "call medical assistance" in mode_input or
            "i need medical help" in mode_input or
            "health mode" in mode_input or
            "start medical mode" in mode_input or
            "run medical" in mode_input or
            "enable medical mode" in mode_input
        ):
            speak("Medical mode activated. I can help you with medical assistance.")
            run_medical_mode()

        # ---------------- FIRE MODE ----------------
        elif (
            "fire" in mode_input or
            "fire mode" in mode_input or
            "fire detection" in mode_input or
            "fire emergency" in mode_input or
            "activate fire alert" in mode_input or
            "start fire mode" in mode_input or
            "there is a fire" in mode_input or
            "i smell smoke" in mode_input or
            "enable fire mode" in mode_input or
            "turn on fire mode" in mode_input
        ):
            speak("Fire mode activated.")
            fire_mode()

        # ---------------- POLICE MODE ----------------
        elif (
            "police" in mode_input or
            "police mode" in mode_input or
            "call police" in mode_input or
            "alert police" in mode_input or
            "i need police help" in mode_input or
            "start police mode" in mode_input or
            "enable police mode" in mode_input or
            "switch to police" in mode_input or
            "activate police mode" in mode_input or
            "call the cops" in mode_input
        ):
            speak("yes calling the police/cops")
            police_mode()

        # ---------------- READING MODE ----------------
        elif (
            "reading" in mode_input or
            "reading mode" in mode_input or
            "read" in mode_input or
            "read this" in mode_input or
            "start reading mode" in mode_input or
            "read the text" in mode_input or
            "read document" in mode_input or
            "read the sign" in mode_input or
            "enable reading mode" in mode_input or
            "ocr mode" in mode_input or
            "activate reading" in mode_input or
            "i want to read" in mode_input or
            "guide read what's in front" in mode_input
        ):
            speak("Reading mode activated. I can help you read text or documents.")
            run_reading_mode()

        # ---------------- EXIT / QUIT ----------------
        elif (
            "exit" in mode_input or
            "quit" in mode_input or
            "close" in mode_input or
            "shut down" in mode_input or
            # "stop" in mode_input or
            "terminate" in mode_input or
            "goodbye" in mode_input or
            "end session" in mode_input or
            "turn off" in mode_input or
            "i'm done" in mode_input or
            "stop everything" in mode_input or
            "shutdown" in mode_input or 
            "shut down" in mode_input or
            "stop system" in mode_input
        ):
            speak("thank you for using the system, Always here to guide you again anytime")
            speak("Shutting down. Goodbye!")
            winsound.Beep(800, 300)
            time.sleep(0.2)
            exit()

        # ---------------- UNKNOWN INPUT ----------------
        else:
            speak("Sorry, I didn’t understand that. Please repeat or choose a mode again.")


# ===================== MAIN FLOW =====================
def main():
    system_boot()
    winsound.Beep(800, 300)
    time.sleep(0.2)
    select_mode()
    winsound.Beep(800, 300)
    time.sleep(0.2)
    




# ===================== ENTRY POINT =====================
if __name__ == "__main__":
    main()
