import cv2
from deepface import DeepFace
import face_recognition
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dlib
# from sklearn.metrics import accuracy_score
import shutil
import time
import warnings
warnings.filterwarnings('ignore')

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while cap.isOpened():
    try:
        ret, frame = cap.read() 
        
        if not ret:
            print("Error in video")
            break
        
        frame = cv2.flip(frame, 1)
        
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))[0]

        x, y, w, h = faces

        # for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        detected_face = frame[y:y+h, x:x+w]
        result = DeepFace.analyze(detected_face, actions=['emotion'],enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        print(dominant_emotion)
        if dominant_emotion == "sad":
            emotion = "You look sad today !!"
        elif dominant_emotion == "happy":
            emotion = "You look happy today !!"
        elif dominant_emotion == "fear":
            emotion = "You look feared today !!"
        elif dominant_emotion == "surprise":
            emotion = "You look surprised today !!"
        elif dominant_emotion == "angry":
            emotion = "You look angry today !!"
        else: pass

        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imshow("", frame)
            
        if cv2.waitKey(1) == ord('q'):
            break
    except: pass

cap.release()
cv2.destroyAllWindows()
