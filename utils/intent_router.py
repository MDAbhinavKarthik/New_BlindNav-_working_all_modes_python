from typing import Optional, Dict, List
from enum import Enum

class IntentType(Enum):
    MODE_CHANGE = "mode_change"
    CONTROL = "control"
    QUERY = "query"
    EMOTIONAL = "emotional"
    EMERGENCY = "emergency"
    GREETING = "greeting"
    FAREWELL = "farewell"
    UNKNOWN = "unknown"

class Intent:
    def __init__(self, type: IntentType, confidence: float, data: Dict = None):
        self.type = type
        self.confidence = confidence
        self.data = data or {}

class IntentRouter:
    def __init__(self):
        self.CONTROL_KEYWORDS = {
            "pause": ["pause", "hold", "wait", "stop for now"],
            "resume": ["continue", "go on", "carry on", "restart"],
            "stop": ["stop", "end", "quit", "terminate", "cancel"],
            "help": ["help", "what can you do", "commands", "options"]
        }
        
        self.MODE_KEYWORDS = {
            "walking": ["walk", "walking", "pedestrian"],
            "navigation": ["navigate", "navigation", "guide", "directions"],
            "reading": ["read", "text", "ocr", "document"],
            "object_scene": ["object", "scene", "detect", "identify"],
            "assistant": ["assist", "help", "utility", "info"]
        }
        
        self.EMOTIONAL_KEYWORDS = {
            "tired": ["tired", "exhausted", "sleepy", "rest"],
            "confused": ["confused", "lost", "don't understand", "unclear"],
            "frustrated": ["frustrated", "angry", "annoyed", "upset"],
            "happy": ["happy", "great", "wonderful", "good"],
            "worried": ["worried", "scared", "nervous", "afraid"]
        }
        
        self.GREETING_KEYWORDS = [
            "hello", "hi", "hey", "good morning",
            "good afternoon", "good evening"
        ]
        
        self.FAREWELL_KEYWORDS = [
            "bye", "goodbye", "see you", "farewell",
            "good night", "exit", "shut down"
        ]
        
        self.EMERGENCY_KEYWORDS = [
            "emergency", "help me", "sos", "danger",
            "urgent", "critical", "medical"
        ]
        
    def parse_intent(self, text: str) -> Intent:
        if not text:
            return Intent(IntentType.UNKNOWN, 0.0)
        
        text = text.lower()
        
        # Check for emergency first (highest priority)
        if any(word in text for word in self.EMERGENCY_KEYWORDS):
            return Intent(IntentType.EMERGENCY, 1.0)
        
        # Check for mode changes
        for mode, keywords in self.MODE_KEYWORDS.items():
            if any(k in text for k in keywords):
                return Intent(IntentType.MODE_CHANGE, 0.8, {"mode": mode})
        
        # Check for control commands
        for action, keywords in self.CONTROL_KEYWORDS.items():
            if any(k in text for k in keywords):
                return Intent(IntentType.CONTROL, 0.9, {"action": action})
        
        # Check for emotional content
        for emotion, keywords in self.EMOTIONAL_KEYWORDS.items():
            if any(k in text for k in keywords):
                return Intent(IntentType.EMOTIONAL, 0.7, {"emotion": emotion})
        
        # Check for greetings
        if any(word in text for word in self.GREETING_KEYWORDS):
            return Intent(IntentType.GREETING, 0.9)
        
        # Check for farewells
        if any(word in text for word in self.FAREWELL_KEYWORDS):
            return Intent(IntentType.FAREWELL, 0.9)
        
        # If nothing else matches, treat as a query
        return Intent(IntentType.QUERY, 0.5, {"text": text})
    
    def get_follow_up_questions(self, intent: Intent) -> List[str]:
        """Generate appropriate follow-up questions based on the intent."""
        if intent.type == IntentType.MODE_CHANGE:
            return [
                f"Would you like me to pause the current task before switching to {intent.data.get('mode', '')} mode?",
                "Should I remember your current progress for later?"
            ]
        elif intent.type == IntentType.CONTROL and intent.data.get('action') == 'pause':
            return [
                "For how long should I pause?",
                "Would you like me to remind you to resume?"
            ]
        elif intent.type == IntentType.EMOTIONAL:
            emotion = intent.data.get('emotion')
            if emotion == 'tired':
                return ["Would you like to take a break?", "Should I pause all active tasks?"]
            elif emotion == 'confused':
                return ["Would you like me to explain what's happening?", "Should I slow down?"]
        return []
    
    def should_interrupt(self, text: str) -> bool:
        """Determine if the current command should interrupt ongoing tasks."""
        intent = self.parse_intent(text)
        return (intent.type in [IntentType.EMERGENCY, IntentType.MODE_CHANGE] or
                (intent.type == IntentType.CONTROL and 
                 intent.data.get('action') in ['stop', 'pause']))