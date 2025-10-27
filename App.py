
import os
import pickle
import threading
import base64
import io
from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
import cv2
import face_recognition
from requests import post
from gtts import gTTS
import pygame
import tempfile

app = Flask(__name__)
app.secret_key = 'c00b3e82-6931-4e9c-826f-9d1010b82d76'

HF_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/fibonacciai/Persian-llm-fibonacci-1-7b-chat.P1_0"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

KNOWN_FILE = "known_faces.pkl"

class LLMResponseGenerator:
    """Class to generate Persian responses using Hugging Face API."""
    
    def __init__(self, api_url=API_URL, headers=headers):
        self.api_url = api_url
        self.headers = headers
    
    def generate(self, name, is_new):
        if is_new:
            prompt = "یک سلام گرم و دوستانه به یک فرد جدید بگو."
        else:
            prompt = f"یک پیام خوشامدگویی گرم به {name} که قبلاً آمده، بگو. طبیعی و فارسی باشد."
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 50,
                "temperature": 0.7,
                "do_sample": True
            }
        }
        
        try:
            response = post(self.api_url, headers=self.headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated = result[0].get('generated_text', prompt)
                    if generated.startswith(prompt):
                        generated = generated[len(prompt):].strip()
                    return generated
            return "سلام! چطوری؟"  # Fallback
        except Exception as e:
            print(f"Error generating response: {e}")
            return "سلام! چطوری؟"

class SpeechHandler:
    """Class to handle text-to-speech in Persian."""
    
    def __init__(self):
        pygame.mixer.init()
    
    def speak(self, text):
        try:
            tts = gTTS(text=text, lang='fa', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                tts.save(tmp.name)
                pygame.mixer.music.load(tmp.name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
            os.unlink(tmp.name)
        except Exception as e:
            print(f"Error in speech: {e}")

class FaceManager:
    """Class to manage known faces, detection, and recognition."""
    
    def __init__(self, known_file=KNOWN_FILE):
        self.known_file = known_file
        self.known_names = []
        self.known_encodings = []
        self.load_known_faces()
    
    def load_known_faces(self):
        if os.path.exists(self.known_file):
            with open(self.known_file, 'rb') as f:
                data = pickle.load(f)
                self.known_names = data['names']
                self.known_encodings = data['encodings']
    
    def save_known_faces(self):
        with open(self.known_file, 'wb') as f:
            pickle.dump({'names': self.known_names, 'encodings': self.known_encodings}, f)
    
    def add_new_face(self, name, encoding):
        if name:
            self.known_names.append(name)
            self.known_encodings.append(encoding)
            self.save_known_faces()
    
    def recognize_face(self, face_encoding):
        matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
        if True in matches:
            match_index = matches.index(True)
            return self.known_names[match_index], False  # name, is_new
        return "ناشناس", True  # name, is_new
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        annotations = []  # list of (location, name, is_new, color, response)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name, is_new = self.recognize_face(face_encoding)
            color = (0, 255, 0) if not is_new else (255, 0, 0)
            annotations.append(((top, right, bottom, left), name, is_new, color))
        
        return face_locations, face_encodings, annotations

class RobotApp:
    """Main class orchestrating the robot's face recognition and response system."""
    
    def __init__(self):
        self.face_manager = FaceManager()
        self.llm = LLMResponseGenerator()
        self.speech = SpeechHandler()
        self.last_new_encoding = None
        self.pending_new = False
    
    def gen_frames(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            face_locations, face_encodings, annotations = self.face_manager.process_frame(frame)
            
            for i, ((top, right, bottom, left), name, is_new, color) in enumerate(annotations):
                if is_new and i == 0:  # Assume first new face as pending
                    self.last_new_encoding = face_encodings[i]
                    self.pending_new = True
                    name = "جدید"
                    color = (0, 0, 255)  # Blue for pending
                    response = self.llm.generate(name, True)
                    self.speech.speak(response)  # Speak for new
                else:
                    response = self.llm.generate(name, is_new)
                
                # Draw rectangle
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                # Draw name
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                # Draw response
                cv2.putText(frame, response, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if self.pending_new:
                cv2.putText(frame, "لطفاً نام را در فرم وارد کنید", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()
    
    def add_new_person(self, name):
        if name and self.last_new_encoding is not None:
            self.face_manager.add_new_face(name, self.last_new_encoding)
            response = self.llm.generate(name, True)
            threading.Thread(target=self.speech.speak, args=(response,), daemon=True).start()
            self.last_new_encoding = None
            self.pending_new = False

robot = RobotApp()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(robot.gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_name', methods=['POST'])
def add_name():
    name = request.form['name'].strip()
    robot.add_new_person(name)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)