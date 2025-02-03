import cv2
import numpy as np
import mediapipe as mp
import os

#using functions and delete.py
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 

# Function to process the frame and apply MediaPipe model
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image.flags.writeable = False  # Make image non-writeable to improve performance
    result = model.process(image)  # Process the frame with the model
    image.flags.writeable = True  # Make image writeable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    return image, result

def draw_landmark(image, result):
    mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# def extract_keypoints(result):
#     if result.left_hand_landmarks:
#                 # left_hand = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten()
#                 lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21*3)
               
#     if result.right_hand_landmarks:
#                 # right_hand = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten()
#                 rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21*3)

def extract_keypoints(result):
    # Initialize landmarks to zero arrays by default
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
# The result of lh and rh result in 126 which is thhe number of features aavailable for extraction
    if result.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten()

    if result.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten()

    return np.concatenate([lh, rh])  # Concatenate left and right hand keypoints

               
    




