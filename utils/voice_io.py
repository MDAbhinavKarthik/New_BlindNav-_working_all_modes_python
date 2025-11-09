import speech_recognition as sr
from typing import Optional, Dict, List
import threading
import queue
import time
import random
from enum import Enum

class ResponseType(Enum):
    CONFIRMATION = "confirmation"
    ERROR = "error"
    PROMPT = "prompt"
    ACKNOWLEDGMENT = "acknowledgment"
    CLARIFICATION = "clarification"

class EnhancedVoiceIO:
    def __init__(self, lang="en", audio_manager=None, tts=None):
        self.lang = lang
        self.audio_manager = audio_manager
        self.tts = tts
        self.recognizer = sr.Recognizer()
        self.vosk_recognizer = None
        self._vosk_models_available = False
        if lang != "en":
            try:
                from voice_recognition import VoskRecognizer
                self.vosk_recognizer = VoskRecognizer()
                # Check if models are available
                self._vosk_models_available = True
            except Exception as e:
                print(f"Vosk models not available: {e}")
                self._vosk_models_available = False
        self.response_templates = self._load_response_templates()
        self.last_error_time = 0
        self.error_count = 0
        self.max_consecutive_errors = 3
        self.listening_lock = threading.Lock()
        self.is_listening = False
        
    def _load_response_templates(self) -> Dict[ResponseType, Dict[str, List[str]]]:
        return {
            ResponseType.CONFIRMATION: {
                "en": [
                    "Got it!", "I understand.", "Sure thing!",
                    "Will do.", "Okay!", "Understood."
                ],
                "hi": [
                    "समझ गया!", "ठीक है!", "जरूर!",
                    "कर रहा हूं।", "समझ में आया।"
                ],
                "kn": [
                    "ಅರ್ಥವಾಯಿತು!", "ಸರಿ!", "ಖಂಡಿತ!",
                    "ಮಾಡುತ್ತೇನೆ.", "ಗೊತ್ತಾಯಿತು."
                ]
            },
            ResponseType.ERROR: {
                "en": [
                    "I didn't catch that.", "Could you repeat that?",
                    "I'm having trouble understanding.",
                    "Could you speak more clearly?",
                    "I missed what you said."
                ],
                "hi": [
                    "मैं समझ नहीं पाया।", "क्या आप दोहरा सकते हैं?",
                    "समझने में दिक्कत हो रही है।",
                    "कृपया स्पष्ट बोलें।"
                ],
                "kn": [
                    "ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ.", "ದಯವಿಟ್ಟು ಮತ್ತೆ ಹೇಳಿ.",
                    "ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ಕಷ್ಟವಾಗುತ್ತಿದೆ.",
                    "ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟವಾಗಿ ಮಾತನಾಡಿ."
                ]
            },
            ResponseType.PROMPT: {
                "en": [
                    "I'm listening...", "Go ahead.", "Yes?",
                    "What would you like?", "How can I help?"
                ],
                "hi": [
                    "मैं सुन रहा हूं...", "बताइए।", "हां?",
                    "क्या चाहिए?", "मैं कैसे मदद कर सकता हूं?"
                ],
                "kn": [
                    "ನಾನು ಕೇಳುತ್ತಿದ್ದೇನೆ...", "ಹೇಳಿ.", "ಹೌದು?",
                    "ಏನು ಬೇಕು?", "ನಾನು ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?"
                ]
            }
        }
    
    def get_response(self, response_type: ResponseType) -> str:
        """Get a random response from the templates based on type and current language."""
        templates = self.response_templates.get(response_type, {})
        lang_templates = templates.get(self.lang, templates.get("en", []))
        return random.choice(lang_templates) if lang_templates else ""
    
    def set_lang(self, lang: str):
        """Change the language for voice recognition and responses."""
        self.lang = lang
        # Update recognizer language if needed
        
    def _should_retry(self) -> bool:
        """Determine if we should retry after an error."""
        current_time = time.time()
        if current_time - self.last_error_time > 30:  # Reset after 30 seconds
            self.error_count = 0
        return self.error_count < self.max_consecutive_errors
    
    def start_listening(self):
        """Prepare to listen for voice input."""
        with self.listening_lock:
            if self.is_listening:
                return False
            self.is_listening = True
            if self.audio_manager:
                self.audio_manager.dim_background_music()
            return True
    
    def stop_listening(self):
        """Stop listening and restore audio state."""
        with self.listening_lock:
            if not self.is_listening:
                return
            self.is_listening = False
            if self.audio_manager:
                self.audio_manager.restore_background_music()
    
    def set_language(self, lang_code):
        """Switch to a new language."""
        if lang_code != self.lang:
            self.lang = lang_code
            if lang_code in ["hi", "kn"]:
                if not self.vosk_recognizer:
                    from voice_recognition import VoskRecognizer
                    self.vosk_recognizer = VoskRecognizer()
            else:
                # For English, we'll use the default speech recognition
                if self.vosk_recognizer:
                    self.vosk_recognizer.stop_listening()
                    self.vosk_recognizer = None
    
    def listen(self, timeout=7, phrase_time_limit=None, prompt=None) -> Optional[str]:
        """Enhanced listen method with better error handling and user feedback."""
        if not self.start_listening():
            return None
            
        try:
            if prompt:
                self.speak(prompt, ResponseType.PROMPT)
                
            # Use Google Speech Recognition for all languages during startup
            # (Vosk models may not be available during initial setup)
            if True:  # Always use Google Speech Recognition for voice-driven startup
                with sr.Microphone() as source:
                    try:
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio = self.recognizer.listen(source, timeout=timeout, 
                                                    phrase_time_limit=phrase_time_limit)
                        
                        text = self.recognizer.recognize_google(audio, language=self.lang)
                        self.error_count = 0  # Reset error count on success
                        return text.lower()
                        
                    except sr.WaitTimeoutError:
                        self.error_count += 1
                        self.last_error_time = time.time()
                        if self._should_retry():
                            self.speak("I'm still listening...", ResponseType.PROMPT)
                            return self.listen(timeout, phrase_time_limit)
                        return None
                    
                    except sr.UnknownValueError:
                        self.error_count += 1
                        self.last_error_time = time.time()
                        if self._should_retry():
                            self.speak(self.get_response(ResponseType.ERROR))
                            return self.listen(timeout, phrase_time_limit)
                        return None
                    
                    except Exception as e:
                        # Voice-only interface: no typed fallback; speak brief error and return None
                        self.speak("Voice recognition error.", ResponseType.ERROR)
                        return None
                    
        finally:
            self.stop_listening()
    
    def speak(self, text: str, response_type: ResponseType = None):
        """Speak with appropriate tone and style based on response type."""
        if not text:
            return
            
        if response_type:
            template = self.get_response(response_type)
            if template and not text.startswith(template):
                text = f"{template} {text}"
        
        # Prefer explicit TTS if provided
        if self.tts:
            try:
                # Use blocking to ensure prompts precede listening
                self.tts.speak_blocking(text)
                return
            except Exception:
                pass
        # Fallback to audio_manager if it has speak
        if self.audio_manager and hasattr(self.audio_manager, 'speak'):
            try:
                self.audio_manager.speak(text)
                return
            except Exception:
                pass
        # Last resort: avoid printing for blind users; do nothing
        return
    
    @staticmethod
    def _typed_input_fallback(prompt=None) -> Optional[str]:
        """Disabled typed input fallback for voice-only interface."""
        return None


class BasicVoiceIO:
    """Basic voice I/O for fallback when EnhancedVoiceIO fails."""
    
    def __init__(self, lang="en"):
        self.lang = lang
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.mic_available = False
        
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.mic_available = True
            print("[BasicVoiceIO] Microphone initialized successfully")
        except Exception as e:
            print(f"[BasicVoiceIO] Microphone setup failed: {e}")
            self.mic_available = False
    
    def listen(self, timeout=7, phrase_time_limit=None, prompt=None):
        """Basic speech recognition using Google Speech Recognition."""
        if not self.mic_available:
            return self._typed_input_fallback(prompt)
        
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                text = self.recognizer.recognize_google(audio, language=self.lang)
                return text.lower() if text else None
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"[BasicVoiceIO] Recognition error: {e}")
            return None
        except Exception as e:
            print(f"[BasicVoiceIO] Error: {e}")
            return None
    
    def set_lang(self, lang):
        """Set the recognition language."""
        self.lang = lang
    
    @staticmethod
    def _typed_input_fallback(prompt=None):
        """Fallback to typed input."""
        if prompt:
            print(f"\n💬 {prompt}")
        try:
            return input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            return None