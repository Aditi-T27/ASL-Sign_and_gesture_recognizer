import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Holistic
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

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use the first camera (index 0)

# Check if camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Use MediaPipe Holistic model with specified confidence values
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()  # Capture frame-by-frame
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Apply MediaPipe processing
        image, result = mediapipe_detection(frame, holistic)
        data=result
        # Draw landmarks only if they exist
        if result.face_landmarks:
            mp_drawing.draw_landmarks(image, result.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if result.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if result.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Display the processed frame
        cv2.imshow('MediaPipe Feed', image)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
