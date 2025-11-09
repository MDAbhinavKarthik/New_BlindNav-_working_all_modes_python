import random
from typing import Optional, List, Dict, Callable
from utils.voice_io import ResponseType

# Constants
OBSTACLES = [
    "wall", "pole", "barrier", "fence", "construction",
    "stairs", "hole", "bump", "rock", "tree"
]

class NavigationHelpers:
    def __init__(self, voice, state_manager, task_manager):
        self.voice = voice
        self.state_manager = state_manager
        self.task_manager = task_manager
        self.lock = None  # Will be set by the mode that uses this class
        self.detected_objects = []  # Will be updated by object detection
        self.current_mode = None  # Will be set by the mode manager

    def _handle_emotional_response(self, emotion: str) -> None:
        """Handle user's emotional state appropriately."""
        if not emotion:
            return
            
        responses = {
            "tired": {
                "message": "I understand you're tired. Would you like to pause and take a break?",
                "action": lambda: self._handle_pause_request()
            },
            "confused": {
                "message": "Let me help you. What part would you like me to explain?",
                "action": lambda: self.speak_help()
            },
            "frustrated": {
                "message": "I'm sorry you're frustrated. Let's try a different approach.",
                "action": lambda: self._suggest_alternatives()
            },
            "worried": {
                "message": "I'm here to help. Would you like me to double-check everything?",
                "action": lambda: self._perform_safety_check()
            },
            "happy": {
                "message": "I'm glad you're feeling good! Shall we continue?",
                "action": None
            }
        }
        
        response = responses.get(emotion, {
            "message": "I understand. How can I help you?",
            "action": None
        })
        
        self.voice.speak(response["message"], ResponseType.ACKNOWLEDGMENT)
        if response["action"]:
            response["action"]()
            
    def _handle_pause_request(self) -> None:
        """Handle request to pause current activities."""
        active_tasks = self.state_manager.get_active_tasks()
        if not active_tasks:
            self.voice.speak(
                "There are no active tasks to pause.",
                ResponseType.CLARIFICATION
            )
            return
            
        for task in active_tasks:
            self.task_manager.pause_task(task.name)
        
        self.voice.speak(
            "Tasks paused. Say 'resume' when you want to continue.",
            ResponseType.CONFIRMATION
        )
        
    def _handle_resume_request(self) -> None:
        """Handle request to resume paused activities."""
        task_name = self.state_manager.resume_last_interrupted()
        if task_name:
            self.voice.speak(
                f"Resuming {task_name}.",
                ResponseType.CONFIRMATION
            )
        else:
            self.voice.speak(
                "No paused tasks to resume.",
                ResponseType.CLARIFICATION
            )
            
    def _handle_stop_request(self) -> None:
        """Handle request to stop current activities."""
        self.voice.speak(
            "Are you sure you want to stop all current tasks?",
            ResponseType.CLARIFICATION
        )
        
        response = self.voice.listen(timeout=5)
        if response and any(word in response.lower() 
                          for word in ["yes", "yeah", "sure"]):
            self.task_manager.stop_all()
            self.voice.speak(
                "All tasks stopped.",
                ResponseType.CONFIRMATION
            )
        else:
            self.voice.speak(
                "Continuing current tasks.",
                ResponseType.CONFIRMATION
            )
            
    def _suggest_alternatives(self) -> None:
        """Suggest alternative approaches based on current mode."""
        if self.current_mode == "walking":
            suggestions = [
                "Would you like to try a different route?",
                "I can scan the area more thoroughly.",
                "We can take smaller steps if needed."
            ]
        elif self.current_mode == "reading":
            suggestions = [
                "I can try to enhance the text visibility.",
                "Would you like me to read more slowly?",
                "We can adjust the scanning angle."
            ]
        else:
            suggestions = [
                "Would you like to try a different mode?",
                "I can explain the current options.",
                "We can take a different approach."
            ]
            
        self.voice.speak(
            random.choice(suggestions),
            ResponseType.CLARIFICATION
        )
        
    def _perform_safety_check(self) -> None:
        """Perform a comprehensive safety check of the environment."""
        self.voice.speak(
            "Running safety check...",
            ResponseType.CONFIRMATION
        )
        
        # Check for obstacles
        with self.lock:
            obstacles = [obj for obj in self.detected_objects 
                        if obj["label"] in OBSTACLES]
        
        if obstacles:
            self.voice.speak(
                f"Found {len(obstacles)} potential obstacles. "
                "Would you like details?",
                ResponseType.CLARIFICATION
            )
        else:
            self.voice.speak(
                "No immediate obstacles detected. The path appears clear.",
                ResponseType.CONFIRMATION
            )