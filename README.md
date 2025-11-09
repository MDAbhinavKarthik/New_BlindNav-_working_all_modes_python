# BlindNav+ - Your Personal Navigator

**BlindNav+** is an intelligent, voice-activated navigation and assistance system designed specifically for visually impaired users. The system uses real-time computer vision, GPS navigation, and advanced AI models to help users navigate safely, read text, detect obstacles, and perform daily tasks through natural voice commands.

## 🌟 Key Features

### 🗺️ **Navigation Modes**
- **Real-Time Camera Navigation**: Step-by-step GPS-guided navigation with live obstacle detection
- **Indoor Navigation**: Smart camera-based guidance to help navigate indoor spaces and find exits
- **Outdoor Navigation**: GPS-based routing with real-time path analysis and obstacle avoidance
- **Walking Mode**: Continuous real-time guidance while walking, detecting obstacles, doors, and walls

### 👁️ **Vision & Detection**
- **Object Detection**: Real-time detection of objects, people, vehicles, and obstacles using YOLO and MobileNet-SSD
- **Door & Wall Detection**: Specialized detection for doors and walls with position and distance information
- **Text Reading**: Advanced OCR for both printed text and handwriting using Tesseract and EasyOCR
- **Scene Description**: Comprehensive scene analysis using BLIP image captioning model
- **Camera Orientation Monitoring**: Continuous feedback to ensure proper camera alignment

### 🎙️ **Voice Assistant**
- **Natural Language Processing**: Understands voice commands in multiple languages (English, Kannada, Hindi)
- **Text-to-Speech**: Clear audio feedback using pyttsx3 and edge_tts
- **Context-Aware Commands**: Intelligent command parsing that understands context (e.g., "bus stop" vs "stop system")
- **Personalized Experience**: Remembers user's name and provides personalized interactions

### 📍 **Location Services**
- **Precise Location Detection**: Multi-source IP geolocation with coordinate averaging for accuracy
- **Nearby Places**: Find nearest bus stops, restaurants, and other points of interest
- **Route Calculation**: Step-by-step navigation using OSRM and OpenRouteService APIs
- **Weather Integration**: Location-based weather forecasts and rain predictions

### 🚨 **Safety Features**
- **Emergency Modes**: Quick access to medical, police, and fire emergency services
- **Obstacle Avoidance**: Real-time detection and guidance to avoid obstacles
- **Road Crossing Detection**: Specialized detection for road crossings with vehicle awareness
- **Battery Monitoring**: Continuous battery level monitoring with low battery warnings

### 📖 **Reading & Text Recognition**
- **Live Text Reading**: Continuous scanning and reading of text from camera feed
- **Handwriting Recognition**: Support for handwritten text using EasyOCR
- **Image Quality Assessment**: Automatic feedback on image quality with improvement suggestions
- **Multiple OCR Methods**: Combines Tesseract and EasyOCR for best accuracy

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam/Camera
- Microphone
- Internet connection (for model downloads and location services)
- Windows 10/11 (tested), Linux, or macOS

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd "BlindNav+ your personal navigator"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Install Tesseract OCR
- **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH
- **Linux**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

### Step 4: Download Models (Automatic)
The system will automatically download required models on first run:
- YOLOv8n (object detection)
- DETR (object detection)
- BLIP (image captioning)
- EasyOCR (handwriting recognition)

### Step 5: Run the Application
```bash
python main.py
```

## 📱 Usage Guide

### Starting the System
1. Run `python main.py`
2. Wait for system boot (models will load automatically)
3. Provide your name when prompted
4. Select your preferred language
5. Choose a mode to begin

### Available Modes

#### 🧭 **Navigation Mode**
- **Activate**: Say "Start navigation mode" or "Navigation mode"
- **Usage**: Provide destination address when prompted
- **Features**: 
  - Real-time obstacle detection
  - Step-by-step directions
  - Camera orientation feedback
  - Route recalculation

#### 🤖 **Assistant Mode**
- **Activate**: Say "Assistant mode"
- **Capabilities**:
  - Ask questions: "What time is it?", "What's the weather?"
  - Find places: "Find nearest bus stop"
  - Set reminders: "Remind me to take medicine at 9 PM"
  - Check battery: "What's my battery level?"
  - Weather queries: "Will it rain today?"

#### 📖 **Reading Mode**
- **Activate**: Say "Reading mode"
- **Usage**: Point camera at text (printed or handwritten)
- **Features**:
  - Continuous text scanning
  - Handwriting recognition
  - Image quality feedback
  - Voice commands: "Read again", "Stop reading"

#### 🚶 **Walking Mode**
- **Activate**: Say "Walking mode"
- **Features**:
  - Real-time obstacle detection
  - Door and wall detection
  - Continuous path analysis
  - Safety warnings

#### 👁️ **Object Detection Mode**
- **Activate**: Say "Object detection mode"
- **Features**:
  - Real-time object identification
  - Scene description
  - 360-degree analysis (front, left, right, back)

#### 🚨 **Emergency Modes**
- **Medical**: Say "Medical mode" or "Call medical service"
- **Police**: Say "Police mode" or "Call police service"
- **Fire**: Say "Fire mode" or "Call fire service"

### Voice Commands

#### Navigation Commands
- "Start navigation to [destination]"
- "Find nearest bus stop"
- "Guide me out"
- "Take me outside"

#### General Commands
- "What time is it?"
- "What's the date?"
- "What's the weather?"
- "Will it rain today?"
- "Find nearest [place]"

#### Mode Control
- "Exit mode" - Exit current mode
- "Stop mode" - Stop current operation
- "Go back" - Return to main menu
- "Repeat" - Repeat last instruction

#### Camera Control
- "Turn on camera"
- "Describe scene"
- "What do you see?"

## 🛠️ Technical Details

### Technologies Used
- **Computer Vision**: OpenCV, YOLOv8, MobileNet-SSD, DETR, BLIP
- **OCR**: Tesseract, EasyOCR
- **Speech**: Google Speech Recognition, pyttsx3, edge_tts
- **Navigation**: OSRM, OpenRouteService, Nominatim, Overpass API
- **AI/ML**: PyTorch, Transformers, TensorFlow
- **Languages**: Python 3.8+

### System Architecture
- **Main Module**: `main.py` - Core application logic
- **Modes**: `modes/` - Individual mode implementations
- **Utils**: `utils/` - Helper functions and utilities
- **Models**: `models/` - Pre-trained model files

### Key Functions
- `check_camera_orientation()`: Monitors camera alignment
- `get_current_location()`: Multi-source location detection
- `find_nearby_bus_stops()`: Nearby POI search
- `extract_text_combined()`: Advanced OCR with multiple methods
- `detect_door()` / `detect_wall()`: Specialized detection
- `_start_realtime_navigation()`: Real-time navigation engine

## ⚙️ System Requirements

### Minimum Requirements
- **CPU**: Dual-core 2.0 GHz or higher
- **RAM**: 4 GB
- **Storage**: 5 GB free space (for models)
- **Camera**: USB webcam or built-in camera
- **Microphone**: Built-in or USB microphone
- **Internet**: Required for initial setup and location services

### Recommended Requirements
- **CPU**: Quad-core 2.5 GHz or higher
- **RAM**: 8 GB or more
- **GPU**: Optional (for faster model inference)
- **Internet**: Stable connection for best experience

## 🔧 Configuration

### Language Selection
The system supports multiple languages:
- English (default)
- Kannada
- Hindi

Language is detected automatically or can be set during startup.

### Voice Settings
- Adjustable speech rate
- Multiple voice options
- Customizable feedback frequency

### Camera Settings
- Automatic orientation monitoring
- Quality assessment and feedback
- Adaptive detection based on alignment

## 🐛 Troubleshooting

### Camera Issues
- **Problem**: "Camera not found"
  - **Solution**: Check camera connection, ensure no other app is using it
- **Problem**: Camera orientation warnings
  - **Solution**: Adjust camera angle based on audio feedback

### Model Loading Issues
- **Problem**: Models not loading
  - **Solution**: Check internet connection, ensure sufficient disk space
- **Problem**: Slow performance
  - **Solution**: Close other applications, check system resources

### Location Detection Issues
- **Problem**: Location not detected
  - **Solution**: Check internet connection, try again
- **Problem**: Inaccurate location
  - **Solution**: System uses IP-based location (1-5km accuracy). For better accuracy, enable GPS on device

### Speech Recognition Issues
- **Problem**: Commands not recognized
  - **Solution**: Speak clearly, reduce background noise, check microphone
- **Problem**: System shutting down unexpectedly
  - **Solution**: Use specific exit commands like "Exit mode" instead of just "stop"

## 📝 Notes

### Location Accuracy
- IP-based location provides city-level accuracy (1-5km)
- For precise navigation, GPS coordinates are recommended
- The system averages multiple sources for better accuracy

### Model Downloads
- Models are downloaded automatically on first run
- Initial setup may take 5-10 minutes depending on internet speed
- Models are cached locally for faster subsequent starts

### Privacy & Security
- All processing is done locally when possible
- Location data is used only for navigation services
- Voice commands are processed using Google Speech Recognition (requires internet)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

### Areas for Contribution
- Additional language support
- Improved OCR accuracy
- New detection features
- Performance optimizations
- Documentation improvements

## 📄 License

This project is open source. Please refer to the license file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- Ultralytics for YOLO models
- Hugging Face for transformer models
- OpenStreetMap for mapping data
- All open-source contributors whose libraries make this project possible

## 📞 Support

For issues, questions, or feature requests, please open an issue on the repository.

---

**BlindNav+** - Empowering independence through intelligent navigation and assistance.

*Version: 1.0*  
*Last Updated: 2024*
