from functions import *
import os
import cv2
import numpy as np
import mediapipe as mp

# Directory to save data
DATA_PATH = os.path.join('MP_Data')

# Actions to detect
actions = np.array(['hello', 'thanks'])

# Number of videos per action
no_sequences = 30

# Frames per video
sequence_length = 30

# Create folders for saving data
for action in actions:
    for sequence in range(no_sequences):
        folder_path = os.path.join(DATA_PATH, action, str(sequence))
        os.makedirs(folder_path, exist_ok=True)  # Ensure directory exists

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera could not be accessed.")
    exit()

# Use MediaPipe Holistic model
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    try:
        for action in actions:  # Loop through actions
            print(f"Collecting data for action: {action}")
            for sequence in range(no_sequences):  # Loop through sequences
                print(f"  Sequence {sequence}")
                for frame_num in range(sequence_length):  # Loop through frames
                    # Read video feed
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to capture frame.")
                        break

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_landmark(image, results)

                    # Display feedback
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting frames for {action}, Video {sequence}', (15, 12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)  # Wait 2 seconds at the start
                    else:
                        cv2.putText(image, f'Collecting frames for {action}, Video {sequence}', (15, 12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                    # Extract keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                    
                    try:
                        np.save(npy_path, keypoints)
                    except Exception as e:
                        print(f"Error saving keypoints at {npy_path}: {e}")
                        break

                    # Stop gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        print("Exit requested.")
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
