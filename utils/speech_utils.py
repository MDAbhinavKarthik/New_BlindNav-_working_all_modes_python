import threading
from typing import Optional

import pyttsx3
import speech_recognition as sr

_engine_lock = threading.Lock()
_engine: Optional[pyttsx3.Engine] = None


def _get_engine() -> pyttsx3.Engine:
    global _engine
    with _engine_lock:
        if _engine is None:
            try:
                # Try SAPI5 on Windows first (more reliable)
                import platform
                if platform.system() == 'Windows':
                    try:
                        _engine = pyttsx3.init('sapi5')
                    except:
                        _engine = pyttsx3.init()
                else:
                    _engine = pyttsx3.init()
            except Exception as e:
                # Fallback to default
                print(f"[Engine Init Warning]: {e}")
                _engine = pyttsx3.init()
            
            # Set default properties immediately
            try:
                _engine.setProperty('rate', 150)
                _engine.setProperty('volume', 1.0)
                # Test the engine is working by doing a quick say/stop
                _engine.say("")
                _engine.stop()
            except Exception as e:
                print(f"[Engine Property Warning]: {e}")
                pass
        return _engine


def speak(text: str) -> None:
    """Speak text synchronously using a shared pyttsx3 engine.

    This blocks until the utterance is finished to ensure ordering.
    Handles multi-line text by splitting into lines and speaking each separately.
    """
    if not isinstance(text, str):
        text = str(text)
    
    if not text or not text.strip():
        return
    
    import time
    
    # Split text into lines to handle multi-line text properly
    # This prevents the engine from getting stuck after the first line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # If no newlines found but text is very long, split by sentences to prevent engine from getting stuck
    if len(lines) == 1 and len(lines[0]) > 500:
        import re
        # Split by sentence endings (period, exclamation, question mark followed by space)
        sentences = re.split(r'([.!?]+\s+)', lines[0])
        # Recombine sentences with their punctuation
        lines = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                combined = (sentences[i] + sentences[i + 1]).strip()
                if combined:
                    lines.append(combined)
            else:
                if sentences[i].strip():
                    lines.append(sentences[i].strip())
        # If still no good splits, split by commas for very long text
        if len(lines) == 1 and len(lines[0]) > 300:
            lines = [line.strip() for line in lines[0].split(',') if line.strip()]
    
    if not lines:
        return
    
    try:
        # Process each line separately with a fresh engine to avoid SAPI lockups
        for line_idx, line in enumerate(lines):
            try:
                # Initialize a fresh engine per line (more reliable on Windows)
                import platform
                if platform.system() == 'Windows':
                    try:
                        engine = pyttsx3.init('sapi5')
                    except:
                        engine = pyttsx3.init()
                else:
                    engine = pyttsx3.init()
                
                # Small delay to ensure engine is ready
                time.sleep(0.1)
                
                # Set properties for better speech (ensure they're set each time)
                try:
                    engine.setProperty('rate', 150)  # Speed
                    engine.setProperty('volume', 1.0)  # Volume
                except:
                    pass
                
                # Speak the current line
                engine.say(line)
                
                # Wait for speech to complete - this blocks until done
                engine.runAndWait()
                
                # Ensure engine is ready for next line
                try:
                    engine.stop()
                except:
                    pass
                finally:
                    # Explicitly delete engine to free SAPI resources
                    try:
                        del engine
                    except:
                        pass
                
                # Small delay after each line completes (except for the last line)
                if line_idx < len(lines) - 1:
                    time.sleep(0.2)  # Slightly longer pause between lines
                else:
                    time.sleep(0.1)  # Shorter pause after last line
                
                print(f"🗣️ [SPOKEN - Line {line_idx + 1}/{len(lines)}]: {line}")  # Debug output
                
            except Exception as line_error:
                print(f"❌ [TTS Error on line {line_idx + 1}]: {line_error}")
                # Try to recover and continue with next line
                try:
                    engine.stop()
                except:
                    pass
                time.sleep(0.15)
                continue
        
    except Exception as e:
        print(f"❌ [TTS Error]: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: try to reinitialize engine and speak all text
        try:
            global _engine
            with _engine_lock:
                if _engine is not None:
                    try:
                        _engine.stop()
                    except:
                        pass
                _engine = None  # Force reinit
            time.sleep(0.2)
            engine = _get_engine()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            
            # Try speaking each line again
            for line in lines:
                try:
                    engine.stop()
                    time.sleep(0.15)
                    engine.say(line)
                    engine.runAndWait()
                    time.sleep(0.1)
                    print(f"🗣️ [SPOKEN - Retry]: {line}")
                except:
                    continue
                    
        except Exception as e2:
            print(f"❌ [TTS Error - Retry Failed]: {e2}")
            import traceback
            traceback.print_exc()


def listen_for_speech(timeout: int = 5, phrase_time_limit: int = 8) -> Optional[str]:
    """Capture speech from default microphone and return recognized text via Google API."""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        # Return empty string so callers can safely do .lower().strip()
        return ""
    except Exception:
        return ""
