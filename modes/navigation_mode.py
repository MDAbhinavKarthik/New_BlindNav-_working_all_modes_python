import time
import cv2
import numpy as np
import pytesseract
from utils.speech_utils import speak, listen_for_speech


# ===============================
# CAMERA-BASED REAL-TIME GUIDANCE
# ===============================
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]


def _load_mobilenet_ssd():
    prototxt_path = "models/MobileNetSSD_deploy.prototxt.txt"
    caffemodel_path = "models/MobileNetSSD_deploy.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    return net


def _estimate_steps_from_offset(x_center, frame_width):
    center = frame_width // 2
    dx = x_center - center
    # steps: 1 step per 60 pixels offset, minimum 1 when direction exists
    steps = int(abs(dx) / 60)
    if steps == 0 and abs(dx) > frame_width * 0.1:
        steps = 1
    direction = "left" if dx < 0 else "right"
    return steps, direction


def _extract_bus_number_candidates(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    text = pytesseract.image_to_string(gray)
    # Extract digit sequences of length >= 2 as possible bus numbers
    import re
    candidates = re.findall(r"\b\d{2,5}\b", text)
    # Deduplicate, keep order
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq[:3]


def _speak_rate_limited(last_times, key, text, min_interval=2.0):
    now = time.time()
    last = last_times.get(key, 0)
    if now - last >= min_interval:
        speak(text)
        last_times[key] = now


def _relative_position(x_center, frame_width):
    center = frame_width // 2
    dx = x_center - center
    if abs(dx) < frame_width * 0.07:
        return "center"
    return "left" if dx < 0 else "right"


def _clearer_side(detections, frame_width):
    """Heuristic: choose side with fewer/lower-ratio obstacles."""
    left_score = 0.0
    right_score = 0.0
    for det in detections:
        x_center = (det["startX"] + det["endX"]) / 2.0
        ratio = det["ratio"]
        if x_center < frame_width / 2:
            left_score += ratio
        else:
            right_score += ratio
    if abs(left_score - right_score) < 0.05:
        return "either"
    return "left" if left_score < right_score else "right"


def _guidance_loop(destination_hint: str = ""):
    speak("Starting camera guidance. I will guide you in real time.")
    if destination_hint:
        speak(f"Destination set to {destination_hint}.")

    net = _load_mobilenet_ssd()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not available. Please check the connection.")
        return

    last_said = {}
    last_instruction = ""
    confirm_moves = True  # ask user to say 'ready' after step guidance
    info_mode = "normal"  # or "brief"
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                _speak_rate_limited(last_said, "cam", "Camera feed lost. Trying again...")
                time.sleep(0.5)
                continue;

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            closest_obstacle = None
            bus_seen = False
            det_list = []

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < 0.4:
                    continue
                idx = int(detections[0, 0, i, 1])
                if idx < 0 or idx >= len(CLASSES):
                    continue
                label = CLASSES[idx]
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Determine proximity using bbox height ratio
                height_ratio = (endY - startY) / float(h)

                if label in ["person", "car", "bus", "chair", "bicycle", "motorbike", "train", "dog"]:
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

                if label == "bus":
                    bus_seen = True

            # Give guidance for closest obstacle
            if closest_obstacle is not None:
                ox_center = int((closest_obstacle["startX"] + closest_obstacle["endX"]) / 2)
                steps, dirn = _estimate_steps_from_offset(ox_center, w)
                ratio = closest_obstacle["ratio"]
                relpos = _relative_position(ox_center, w)

                if ratio > 0.55:
                    last_instruction = f"Stop. {closest_obstacle['label']} very close ahead, {relpos}."
                    _speak_rate_limited(last_said, "stop", last_instruction)
                elif ratio > 0.35:
                    if steps > 0:
                        last_instruction = f"Move {steps} steps {dirn} to avoid the {closest_obstacle['label']} {relpos}."
                        _speak_rate_limited(last_said, "avoid", last_instruction)
                    else:
                        last_instruction = f"Careful, {closest_obstacle['label']} ahead {relpos}."
                        _speak_rate_limited(last_said, "careful", last_instruction)
                else:
                    last_instruction = "Path looks clear. Continue straight."
                    _speak_rate_limited(last_said, "clear", last_instruction)
            else:
                last_instruction = "Path looks clear. Continue straight."
                _speak_rate_limited(last_said, "clear", last_instruction)

            # Bus handling: OCR numbers when a bus is present occasionally
            if bus_seen:
                _speak_rate_limited(last_said, "bus", "Bus detected. Checking bus number.", 6.0)
                candidates = _extract_bus_number_candidates(frame)
                if candidates:
                    speak(f"I see bus number candidates: {', '.join(candidates)}. Are you looking for one of these? Say yes or no.")
                else:
                    _speak_rate_limited(last_said, "busnum", "I couldn't read the bus number.", 6.0)

            # Offer optional confirmation after movement instruction
            if confirm_moves and last_instruction and any(x in last_instruction.lower() for x in ["move", "stop", "careful"]):
                _speak_rate_limited(last_said, "askready", "Say 'ready' when you have adjusted, or say 'repeat'.", 4.0)

            # Check for user commands interactively
            cmd = listen_for_speech(timeout=2, phrase_time_limit=3)
            if cmd:
                cmd_l = cmd.lower().strip()
                # Session controls
                if any(x in cmd_l for x in ["stop", "exit", "cancel", "end"]):
                    speak("Stopping navigation guidance.")
                    break
                if any(x in cmd_l for x in ["repeat", "again"]):
                    if last_instruction:
                        speak(last_instruction)
                    continue
                if "ready" in cmd_l:
                    _speak_rate_limited(last_said, "ack", "Okay, proceeding.", 1.0)
                    continue
                if any(x in cmd_l for x in ["describe", "what's around", "whats around", "around"]) and "scene" in cmd_l:
                    # Use enhanced scene description
                    from main import describe_scene
                    describe_scene()
                    continue
                elif any(x in cmd_l for x in ["describe", "what's around", "whats around", "around"]):
                    # Quick description of current view
                    if det_list:
                        # List up to 3 closest by ratio
                        det_list_sorted = sorted(det_list, key=lambda d: d["ratio"], reverse=True)[:3]
                        descs = []
                        for d in det_list_sorted:
                            pos = _relative_position(int((d["startX"] + d["endX"]) / 2), w)
                            descs.append(f"{d['label']} {pos}")
                        speak("I see: " + ", ".join(descs) + ".")
                    else:
                        speak("I don't see any obstacles nearby.")
                    continue
                if any(x in cmd_l for x in ["scan left", "look left", "scan right", "look right", "which side"]):
                    side = _clearer_side(det_list, w)
                    if side == "either":
                        speak("Both sides look similar. Continue straight.")
                    else:
                        speak(f"The {side} side looks clearer.")
                    continue
                if any(x in cmd_l for x in ["less talk", "be brief", "brief"]):
                    info_mode = "brief"
                    speak("Okay, I'll keep guidance brief.")
                    continue
                if any(x in cmd_l for x in ["more detail", "explain", "talk more", "verbose"]):
                    info_mode = "normal"
                    speak("Okay, I'll provide more detail.")
                    continue
                if bus_seen and any(x in cmd_l for x in ["yes", "no"]):
                    if "yes" in cmd_l:
                        speak("Okay, I'll help you move towards the bus doors safely. Keep your hand ready to hold the railing.")
                    else:
                        speak("Okay, I will keep checking bus numbers.")
                    continue

            time.sleep(0.1)
    finally:
        try:
            cap.release()
        except:
            pass


def navigation_mode():
    speak("Navigation mode activated. Say your destination or say 'skip'.")
    dest = listen_for_speech()
    dest = (dest or "").strip()
    if not dest or dest.lower() == "skip":
        _guidance_loop("")
    else:
        _guidance_loop(dest)

# ===============================
# RUN THE PROGRAM
# ===============================
if __name__ == "__main__":
    navigation_mode()
