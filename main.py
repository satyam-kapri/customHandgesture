import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')
import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle
import pyttsx3
import threading
import queue
import time

# Initialize TTS Engine
engine = pyttsx3.init()
tts_queue = queue.Queue()

prev_gesture_label=None
# Function to handle text-to-speech in a separate thread
def tts_worker():
    global prev_gesture_label
    while True:
        text = tts_queue.get()
        if text is None:  # Sentinel value to exit the thread
            break
        if prev_gesture_label!=text:
            engine.say(text)
            engine.runAndWait()
            prev_gesture_label=text
        tts_queue.task_done()

# Start TTS worker thread
threading.Thread(target=tts_worker, daemon=True).start()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize data storage for training
X = []
y = []
label_encoder = LabelEncoder()

# Function to extract features from hand landmarks
def extract_features(landmarks):
    features = []
    for landmark in landmarks:
        features.append(landmark.x)
        features.append(landmark.y)
    return features

# Function to capture training data
def capture_training_data():
    global X, y
    while True:
        gesture_label = input("Enter the gesture label (or 'q' to quit capturing): ")
        if gesture_label.lower() == 'q':
            train_model()
            break
        sample_count=0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if(sample_count==200):
               save_sample(frame, gesture_label)
            sample_count+=1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    features = extract_features(hand_landmarks.landmark)
                    X.append(features)
                    y.append(gesture_label)

            cv2.imshow('Capture Training Data', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to stop capturing for this gesture
                break
            
def save_sample(frame, label):
    sample_dir = 'samples'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    label_dir = os.path.join(sample_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Save image with label as filename
    filename = f'{label}_{len(os.listdir(label_dir)) + 1}.jpg'  # Unique filename
    filepath = os.path.join(label_dir, filename)
    cv2.imwrite(filepath, frame)

# Function to train the model
def train_model():
    global X, y, label_encoder
    
    if os.path.exists('gesture_model.pkl') and os.path.exists('label_encoder.pkl'):
        with open('gesture_model.pkl', 'rb') as model_file:
            clf = pickle.load(model_file)
        with open('label_encoder.pkl', 'rb') as le_file:
            label_encoder = pickle.load(le_file)
            prev_X = clf.support_vectors_
            prev_y = label_encoder.inverse_transform(clf.predict(prev_X))
            X.extend(prev_X)
            y.extend(prev_y)

    X = np.array(X)
    y = label_encoder.fit_transform(y)

   

    # Train SVM classifier
    clf = SVC(kernel='linear')
    clf.fit(X,y)
    
    # Save the trained model and label encoder
    with open('gesture_model.pkl', 'wb') as model_file:
        pickle.dump(clf, model_file)
    with open('label_encoder.pkl', 'wb') as le_file:
        pickle.dump(label_encoder, le_file)
    print("Model trained and saved successfully!")

# Function to recognize gestures in real-time
def recognize_gestures():
    with open('gesture_model.pkl', 'rb') as model_file:
        clf = pickle.load(model_file)
    with open('label_encoder.pkl', 'rb') as le_file:
        label_encoder = pickle.load(le_file)
    
    prev_gesture_label=None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                features = np.array([extract_features(hand_landmarks.landmark)])
                prediction = clf.predict(features)
                gesture_label = label_encoder.inverse_transform(prediction)[0]
                cv2.putText(frame, gesture_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                if(prev_gesture_label!=gesture_label):
                    tts_queue.put(gesture_label)
                    prev_gesture_label=gesture_label
                
        cv2.imshow('Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break


# Start video capture
cap = cv2.VideoCapture(0)

# Main loop
while True:
    print("Press 't' to capture training data, 'r' to recognize gestures, or 'q' to quit.")
    choice = input().lower()
    if choice == 't':
        capture_training_data()
    elif choice == 'r':
        recognize_gestures()
    elif choice == 'q':
        break
    else:
        print("Invalid choice. Please try again.")

# Train the model after capturing training data
# if X and y:
#     train_model()

cap.release()
cv2.destroyAllWindows()

