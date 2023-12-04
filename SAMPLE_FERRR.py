import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
import face_recognition
# from sklearn.metrics import accuracy_score
import shutil
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator,save_img
import optuna
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import load_model
import cv2
from time import sleep

# Load the face emotion recognition model
model = load_model(r"C:\Users\MIT\Desktop\Facial_emotion_recognition\Face_emotion_recognition_trail6_4091_IMAGES_LANCOSZ_cnn2.h5")
x, y, w, h = 0, 0, 0, 0
# Function to predict emotions for a face image
def predict_emotion(face_image):
    global x,y,w,h
    image=face_image
    # image_path=r"C:\Users\mitvo\Documents\Facial_Image_Dataset\captured_images\captured_image.jpg"

    output_image_path=r"C:\Users\MIT\Desktop\Facial_emotion_recognition\grayscaled_image.jpg" # Predefined Output_path of the grayscaled image

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height,width = gray_image.shape
    blank_image = np.zeros((height,width,3),dtype=np.uint8)
    blank_image[:,:,0] = gray_image
    blank_image[:,:,1] = gray_image
    blank_image[:,:,2] = gray_image
    gimg=blank_image  
    cv2.imwrite(output_image_path,gimg)

    img = cv2.GaussianBlur(gimg,ksize=(3,3),sigmaX=2)

    face_location=face_recognition.face_locations(img)
    cv2.rectangle(frame, (x, y), (x+h, y-w), (0, 255, 255), 1)
    # detected_face = frame[y:y+h, x:x+w]
    if face_location==[]:
        face=img
    else:
        top,right,bottom,left= face_location[0]
        x, y, w, h =top,right,bottom,left # Extracting facial coordinates
        face=img[top:bottom,left:right]
        print(face.shape)
        cv2.imwrite(r"C:\Users\MIT\Desktop\Facial_emotion_recognition\recognized_part.jpg",face)

    target_size = (62,63,3)

    # Image preprocessing
    processed_face=[]
    if face.shape[0] < target_size[0] or face.shape[1] < target_size[1]:
        # Resizing smaller images while maintaining aspect ratio
        scaling_factor = min(target_size[0] / face.shape[0],target_size[1] / face.shape[1])

        new_width=int(face.shape[1]*scaling_factor)

        new_height=int(face.shape[0]*scaling_factor)

        resized_image=cv2.resize(face,(new_width,new_height),interpolation=cv2.INTER_LANCZOS4)

        # Creating a blank image of the target size with the same number of channels as the original image
        blank_image = np.zeros((target_size[0],target_size[1],face.shape[2]), dtype=np.uint8)
        # Calculating the position to paste the resized image in the blank image to center it
        x_offset = (target_size[1] - resized_image.shape[1])//2

        y_offset = (target_size[0] - resized_image.shape[0])//2

        # Pasting the resized image into the blank image
        blank_image[y_offset:y_offset + resized_image.shape[0],x_offset:x_offset + resized_image.shape[1], :] = resized_image

        processed_face.append(blank_image)
    else:
        cropped_image=cv2.resize(face,(target_size[1],target_size[0]))

        processed_face.append(cropped_image)

    pixel_data = np.array(processed_face)

    datagen = ImageDataGenerator(horizontal_flip=True,zoom_range=0.2,shear_range=0.2,featurewise_center=True,
                                 featurewise_std_normalization=True)
    
    datagen.fit(pixel_data)

    generator = datagen.flow(x=pixel_data,batch_size=len(processed_face),shuffle=False)

    preprocessed_data = next(generator)
    # Perform emotion prediction using your model
    predictions = model.predict(preprocessed_data)

    max_index = np.argmax(predictions)
    max_value = predictions[0, max_index]
    emotions = {0: "angry", 1: "contempt", 2: "disgust", 3: "fear", 4: "happy", 5: "neutral", 6: "sad", 7: "surprise"}
    detected_expression = emotions[max_index]
    return detected_expression

# Open the video capture
video_capture = cv2.VideoCapture(0)  

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture the frame.")
        break

    # Display the frame
    
    pixel_values = frame
    print(pixel_values.shape)
    Detected_expression=predict_emotion(pixel_values)
    cv2.putText(frame, Detected_expression, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imshow('Live Video', frame)
    # Check for the 'q' key to exit the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
