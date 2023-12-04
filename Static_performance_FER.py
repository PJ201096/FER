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

''' TO CHECK WHETHER THE IMAGE CHANNEL IS IN BGR (OR) RGB FORMAT
 CONVERTING THE INPUT IMAGE TO THE GRAY SCALE 
'''

image_path=r"C:\Users\MIT\Desktop\Facial_emotion_recognition\Testing images\sample_image_9.jpg" #path of input image
output_image_path=r"C:\Users\MIT\Desktop\Facial_emotion_recognition\Testing images\grayscaled_image9.jpg" # Predefined Output_path of the grayscaled image
image = cv2.imread(image_path)
height,width,channels = image.shape
region = image[:10,:10] 
blue_channel_mean = region[:,:,0].mean()
red_channel_mean = region[:,:,2].mean()
print(f"{blue_channel_mean} : blue channel intensity, {red_channel_mean} : red_channel_intensity")
if red_channel_mean > blue_channel_mean:
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
else:
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image=cv2.cvtColor(rgb_img,cv2.COLOR_RGB2GRAY)

cv2.imwrite(output_image_path,gray_image)

image = cv2.imread(output_image_path)
img = cv2.GaussianBlur(image,ksize=(3,3),sigmaX=2)
face_location=face_recognition.face_locations(img)
if face_location==[]:
    plt.imshow(img)
    face=img
else:
    top,right,bottom,left= face_location[0] # Extracting facial coordinates
    face=img[top:bottom,left:right]
    print(face.shape)
    plt.imshow(face)
    cv2.imwrite(r"C:\Users\MIT\Desktop\Facial_emotion_recognition\Testing images\recognized_part9.jpg",face)

target_size = (62,63,3)

# Image preprocessing
processed_face=[]
if face.shape[0] < target_size[0] or face.shape[1] < target_size[1]:
    # Resizing smaller images while maintaining aspect ratio
    scaling_factor = min(target_size[0] / image.shape[0],target_size[1] / image.shape[1])
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
    plt.imshow(blank_image)
    processed_face.append(blank_image)
else:
    cropped_image=cv2.resize(face,(target_size[1],target_size[0]))
    plt.imshow(cropped_image)
    processed_face.append(cropped_image)

pixel_data = np.array(processed_face)

datagen = ImageDataGenerator(horizontal_flip=True,zoom_range=0.2,shear_range=0.2,featurewise_center=True,featurewise_std_normalization=True)
datagen.fit(pixel_data)
generator = datagen.flow(x=pixel_data,batch_size=len(processed_face),shuffle=False)
preprocessed_data = next(generator)

model = load_model(r"C:\Users\MIT\Desktop\Facial_emotion_recognition\Face_emotion_recognition_trail6_4091_IMAGES_LANCOSZ_cnn2_normalised_layers.h5")
predictions = model.predict(preprocessed_data)
max_index = np.argmax(predictions)
# Get the maximum value using the index
max_value = predictions[0, max_index]
print("Index with maximum value:",max_index)
print("Maximum value:", max_value)
emotions={0:"angry",1:"contempt",2:"disgust",3:"fear",4:"happy",5:"neutral",6:"sad",7:"surprise"}
detected_expression=emotions[max_index]
print(detected_expression)

