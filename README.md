# Face Recognition Robot with Persian LLM Responses

## Overview
This project implements a prototype for an intelligent robot that uses a webcam to detect and recognize human faces. It leverages computer vision for face detection, stores known faces persistently, and generates natural Persian-language responses using a Hugging Face LLM. New faces prompt users to enter a name via a simple web interface, while returning faces receive personalized greetings. Responses can be displayed on-screen and spoken via text-to-speech (TTS).

Key features:
- Real-time face detection and recognition using OpenCV and `face_recognition`.
- Persistence of known faces via pickle files.
- Persian-specific LLM integration (Hugging Face Inference API) for dynamic greetings.
- Optional TTS playback for new detections.
- Simple Flask web UI for video streaming and name entry.

The code is structured in an object-oriented manner for modularity and extensibility.

## Installation
1. Clone or download the project files.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Obtain a free Hugging Face API token:
   - Go to [Hugging Face Settings](https://huggingface.co/settings/tokens).
   - Create a new token with read access.
   - Replace `"hf_your_token_here"` in `app.py` with your token.

4. Ensure your webcam is connected and accessible.

## Project Structure
```
project/
--- app.py                 # Main Flask application with OOP classes
--- requirements.txt       # Python dependencies
--- known_faces.pkl        # Generated: Stores known face encodings (created on first run)
--- templates/
    --- index.html         # Web UI template (Persian RTL layout)
```

## Usage
1. Run the application:
   ```
   python app.py
   ```
   - The server starts on `http://0.0.0.0:5000` (accessible via `http://localhost:5000` in your browser).
   - Debug mode is enabled for development.

2. Interact via the web UI:
   - Point your webcam at a face.
   - **New face**: A blue rectangle appears with "جدید" (New). Enter the name in the form and submit to register. A spoken greeting plays.
   - **Known face**: A green rectangle with the name and a "welcome back" message (e.g., "خوش آمدی دوباره، سارا!").
   - Responses are overlaid on the video feed in Persian.
   - Speech (TTS) triggers for new registrations (server-side, plays on host machine).

3. Quit: Ctrl+C in the terminal, or close the browser.

**Example Responses** (from LLM):
- New: "سلام! خوشحالم که می‌بینمت. چطوری؟"
- Returning: "خوش آمدی دوباره، علی! امروز چیکار می‌کنی؟"

## Configuration
- **HF Token**: Edit `app.py` for the API token. Uses the Persian model `fibonacciai/Persian-llm-fibonacci-1-7b-chat.P1_0` (free tier; rate-limited).
- **Secret Key**: Change `app.secret_key` in `app.py` for production.
- **Port/Host**: Modify `app.run()` if needed.
- **TTS**: Requires speakers/headphones on the host. gTTS uses Google servers (internet required).

## Classes in `app.py`
- `LLMResponseGenerator`: Handles Persian LLM API calls.
- `SpeechHandler`: Manages TTS playback with gTTS and Pygame.
- `FaceManager`: Loads/saves faces, processes frames for detection/recognition.
- `RobotApp`: Orchestrates the system, streams video frames.

## Troubleshooting
- **No video feed**: Check webcam permissions (browser may prompt). Test with `cv2.VideoCapture(0)`.
- **API Errors**: Verify HF token; fallback to "سلام! چطوری؟".
- **face_recognition fails**: Ensure dlib is installed; encodings are 128D vectors.
- **Speech issues**: Pygame may need SDL libs; test audio separately.
- **Multiple faces**: Prototype handles one pending new face; extend `FaceManager` for multi-user.

## Ideas to Enhance the Face Recognition Robot Project
### 1. Emotion-Aware Greetings
- **Application**: In elderly care or mental health counseling, the robot detects the user's emotion (joy/stress) and provides empathetic responses, such as "You seem tired—shall I make you some tea?"
- **Technical**: Integrate the DeepFace library for emotion detection in `FaceManager`; update the LLM with dynamic prompts (e.g., "Give an empathetic response to a sad face").
- **Optimization**: Use lightweight models (like MobileNet) to reduce memory usage (under 100MB) and increase speed (30fps); scalability with TensorFlow Lite for edge execution (on Raspberry Pi).

### 2. Augmented Reality Overlays (AR Overlays)
- **Application**: In children's education, the robot displays interactive animations (like Persian cartoons) on the user's face, e.g., "Sara, welcome with this Shahnameh mask!"
- **Technical**: Add an AR module with OpenCV AR markers to `gen_frames`; have the LLM generate custom AR descriptions.
- **Optimization**: Cache AR images in Redis to reduce latency (under 50ms); scalability with Docker for multi-node deployment, and memory optimization with automatic Python garbage collection.

### 3. Two-Way Voice-Activated Dialogue
- **Application**: In shopping centers, the robot chats with Persian-speaking users and offers personalized suggestions (like products) based on interaction history.
- **Technical**: Integrate Whisper (from HuggingFace) for speech-to-text in a new `VoiceHandler` class; expand the LLM for Q&A cycles.
- **Optimization**: Parallel processing with multiprocessing for speed (response <1s); store history in SQLite for low memory (under 1GB for 1000 users) and cloud scalability with AWS Lambda.

### 4. Smart Home Integration (Smart Home Sync)
- **Application**: In smart homes, user detection activates devices, such as "Ali's back; playing your favorite music."
- **Technical**: Connect to an MQTT broker in `RobotApp` for IoT control (e.g., Philips Hue); LLM for generating custom commands.
- **Optimization**: Use asyncio for asynchronous I/O to boost speed; scalability with Kubernetes for multi-device setups, and face data compression with PCA for 50% memory savings.

### 5. Privacy-Preserving Federated Learning
- **Application**: In corporate environments, the face recognition model trains without sharing personal data, ensuring employee security.
- **Technical**: Implement Flower (federated framework) in `FaceManager`; integrate differential privacy with Opacus.
- **Optimization**: Distributed training for scalability (up to 100 nodes); memory optimization with model quantization (from 16-bit to 8-bit) and 10x speed via CUDA GPU acceleration.

### 6. Gamified Social Interactions
- **Application**: In team events, score interactions (e.g., "Sara, 10 points for the hello!") to boost employee motivation.
- **Technical**: Add a simple DB (MongoDB) to `RobotApp` for tracking scores; LLM for generating gamified challenges.
- **Optimization**: Database indexing for fast searches (under 10ms); scalability with sharding and Redis caching for efficient memory (supporting 10k concurrent users).

### 7. Cultural and Contextual Adaptation
- **Application**: In tourist tours, the robot provides culturally tailored responses based on user attire/background (detected via CV), like suggesting local dishes.
- **Technical**: Use YOLO for background object detection in `process_frame`; enrich LLM prompts with cultural data (from HuggingFace datasets).
- **Optimization**: Lightweight ONNX models for CPU inference speed; scalability with serverless (Google Cloud Run) and memory optimization via lazy data loading.

### 8. Edge Swarm Robotics
- **Application**: In museums, a group of robots shares face data, e.g., "Guest handed off from robot 1 to 2."
- **Technical**: Synchronize the face database with WebSockets (Flask-SocketIO) in a `SwarmManager` class.
- **Optimization**: Load balancing with Apache Kafka for scalability (up to 50 robots); encoding compression with autoencoders for 30% less memory and 5G for fast transfers.

### 9. Health and Wellness Prompts
- **Application**: For remote workers, detect fatigue (via pose estimation) and suggest breaks, like "Ali, take a 5-minute walk!"
- **Technical**: Integrate MediaPipe for pose in `FaceManager`; LLM for generating Persian health tips.
- **Optimization**: Real-time processing with TensorRT for speed (60fps); anonymous log storage in InfluxDB for scalability and low memory (with auto-cleanup scheduling).

### 10. Sustainable Feedback Loop (Eco-Feedback Loop)
- **Application**: In green offices, track user habits (like visit frequency) and suggest eco-friendly actions, e.g., "Sara, use a bike to reduce carbon."
- **Technical**: Connect to carbon APIs (like Carbon Interface) in `LLMResponseGenerator`; use graph NNs for habit modeling.
- **Optimization**: Training optimization with PyTorch Lightning for speed; scalability with microservices (FastAPI) and vector databases (Pinecone) for efficient habit search memory.

## Implementation Notes
These ideas transform the project from a simple prototype into an advanced system, emphasizing efficiency (speed >30fps, memory <500MB) and scalability (thousands of users). For implementation, we can use tools like Docker for deployment and Prometheus for monitoring.
