import datetime
import random
import time
from utils.speech_utils import speak, listen_for_speech
from utils.knowledge_base import get_term, add_term, learn_from_web
import requests
import pyttsx3
import speech_recognition as sr
from datetime import datetime, timedelta
import re
import threading
import urllib.parse
import webbrowser
import psutil

# You can add APIs later
# import requests
# import pyowm   # For weather
# import playsound  # For music

reminders = []


def get_time():
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


def get_date():
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y")


def get_period():
    hour = datetime.datetime.now().hour
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

    def speak(text):
        """Speak text aloud."""
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    def listen():
        """Listen for voice input and return recognized text."""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            speak("Sorry, I didn't understand that.")
        except sr.RequestError:
            speak("Sorry, my speech service is unavailable right now.")
        return None

    def get_location():
        """Fetch approximate city via IP geolocation."""
        try:
            res = requests.get("https://ipinfo.io/json", timeout=5)
            return res.json().get("city", None)
        except Exception:
            return None

    def check_rain(city):
        """Check if it's likely to rain soon in the given city."""
        try:
            res = requests.get(f"https://wttr.in/{city}?format=j1", timeout=5)
            data = res.json()
            hourly = data["weather"][0]["hourly"][:3]  # next few hours
            will_rain = any(int(h["chanceofrain"]) > 50 for h in hourly)
            return will_rain
        except Exception:
            return None

    # 1️⃣ Get current location and check rain
    city = get_location()
    if not city:
        speak("Sorry, I couldn’t determine your location.")
        return

    speak(f"Let me check if it might rain soon in {city}. Please wait.")
    rain_now = check_rain(city)

    if rain_now is None:
        speak("Sorry, I couldn’t fetch the weather right now.")
    elif rain_now:
        speak(f"Yes, it looks like it might rain in {city} soon. You should take an umbrella.")
    else:
        speak(f"No, it doesn’t look like rain in {city} soon.")

    # 2️⃣ Ask the user for another city
    speak("For which location do you want me to check the rainfall? Please say the city name after the beep.")
    other_city = listen()

    if not other_city:
        speak("I didn’t catch any city name.")
        return

    # 3️⃣ Get weather for the requested city
    speak(f"Checking rain conditions for {other_city}. Please wait.")
    rain_other = check_rain(other_city)

    if rain_other is None:
        speak(f"Sorry, I couldn’t fetch the weather for {other_city}.")
    elif rain_other:
        speak(f"Yes, it looks like it might rain soon in {other_city}. You should take an umbrella.")
    else:
        speak(f"No, it doesn’t look like rain in {other_city} soon.")


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
    speak("Assistant mode activated. How can I help you today?")

    while True:
        command = listen_for_speech()
        if not command:
            continue

        command = command.lower()

        # Exit condition
        if "exit" in command or "stop assistant" in command:
            speak("Exiting assistant mode. Returning to main system.")
            break

        # Time and date
        elif "time" in command:
            speak(f"The time is {get_time()}.")

        elif "date" in command or "today" in command:
            speak(f"Today is {get_date()}.")

        elif "morning" in command or "afternoon" in command or "evening" in command:
            speak(f"It’s {get_period()} right now.")

        elif "wake me up" in command or "set alarm" in command:
            set_alarm("7 a.m. tomorrow")

        # Weather and climate
        elif "weather" in command:
            get_weather()

        elif "rain" in command or "umbrella" in command:
            need_umbrella()

        elif "temperature" in command:
            get_temperature()

        # elif "umbrella" in command:
        #     need_umbrella()

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
            from main import describe_scene
            describe_scene()

        # Mode switching (future integration)
        elif "switch" in command or "mode" in command:
            speak("Mode switching is managed by the main system. Please say 'activate navigation mode' or another mode name in main system.")
            break

        else:
            # Try to learn unknown terms in real time
            term = command.strip()
            if not term:
                speak("I didn't catch that. Please say it again.")
                continue
            known = get_term(term)
            if known:
                speak(known.get("summary", f"I know about {term}."))
                continue
            speak("Let me check that for you.")
            learned = learn_from_web(term)
            if learned:
                add_term(term, learned.get("summary", ""), learned.get("source", ""))
                # Keep summary concise when speaking
                summary = learned.get("summary", "").strip()
                if len(summary) > 280:
                    summary = summary[:277] + "..."
                if summary:
                    speak(summary)
                else:
                    speak("I saved a link to learn more. You can ask me again later.")
            else:
                speak("Sorry, I couldn't find information right now. Please try again later.")


