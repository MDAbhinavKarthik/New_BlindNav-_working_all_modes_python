import cv2
import torch
import time
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import pyttsx3
from PIL import Image
import random

# -------------------------
# Initialize global models
# -------------------------
engine = pyttsx3.init()
local_model = YOLO("yolov8n.pt")  # local detection model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# -------------------------
# Basic system functions
# -------------------------
def speak(text):
    print(f"SYSTEM: {text}")
    engine.say(text)
    engine.runAndWait()

def listen_for_speech():
    """Mock function — replace with your actual microphone recognizer."""
    return input("[You said]: ")

# -------------------------
# Core object detection logic
# -------------------------
def run_object_detection_mode():
    speak("Object Detection Mode Activated. You can ask me to find any object around you.")
    speak("Say, for example, 'Find a bottle', or 'Describe the scene'.")

    last_detection = None
    cap = None

    while True:
        user_input = listen_for_speech().lower().strip()

        if not user_input:
            continue

        # Exit condition
        if "exit" in user_input or "stop" in user_input:
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
    speak("Capturing the environment for analysis...")
    caption = run_online_detection()
    speak(f"It looks like {caption}.")
