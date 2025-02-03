# import cv2
# import numpy as np
# import mediapipe as mp

# # Initialize MediaPipe Holistic
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# # Function to process the frame and apply MediaPipe model
# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#     image.flags.writeable = False  # Make image non-writeable to improve performance
#     result = model.process(image)  # Process the frame with the model
#     image.flags.writeable = True  # Make image writeable again
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
#     return image, result

# # Function to draw landmarks
# def draw_landmark(image, result):
#     mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#     mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# # Global variable to store keypoints
# data = []

# # Main function to collect keypoints
# def collect_keypoints():
#     global data
#     cap = cv2.VideoCapture(0)  # Use the first camera (index 0)

#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 print("Failed to grab frame")
#                 break

#             # Apply MediaPipe processing
#             image, result = mediapipe_detection(frame, holistic)

#             # Draw landmarks
#             draw_landmark(image, result)

#             # Extract keypoints if detected
#             if result.left_hand_landmarks:
#                 left_hand = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten()
#                 data.append(left_hand)
#             if result.right_hand_landmarks:
#                 right_hand = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten()
#                 data.append(right_hand)

#             # Display the processed frame
#             cv2.imshow('MediaPipe Feed', image)

#             # Exit the loop if 'q' is pressed
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()

# # Execute only when the script is run directly
# if __name__ == "__main__":
#     collect_keypoints()
#     print("Keypoints collected:", data)

import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Path to save the data
DATA_FILE = "keypoints_data.npy"

# Function to process the frame and apply MediaPipe model
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image.flags.writeable = False  # Make image non-writeable to improve performance
    result = model.process(image)  # Process the frame with the model
    image.flags.writeable = True  # Make image writeable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    return image, result

# Function to draw landmarks
def draw_landmark(image, result):
    mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Global variable to store keypoints
data = []

def extract_keypoints(result):
    if result.left_hand_landmarks:
                # left_hand = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten()
                lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21*3)
                data.append(lh)
    if result.right_hand_landmarks:
                # right_hand = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten()
                rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21*3)
                data.append(rh)
    
    return np.concatenate(lh,rh)
# Main function to collect keypoints
def collect_keypoints():
    global data
    cap = cv2.VideoCapture(0)  # Use the first camera (index 0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Apply MediaPipe processing
            image, result = mediapipe_detection(frame, holistic)

            # Draw landmarks
            draw_landmark(image, result)

            # Extract keypoints if detected
            # if result.left_hand_landmarks:
            #     # left_hand = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten()
            #     lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21*3)
            #     data.append(lh)
            # if result.right_hand_landmarks:
            #     # right_hand = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten()
            #     rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21*3)
            #     data.append(rh)
            extract_keypoints(result)


            # Display the processed frame
            cv2.imshow('MediaPipe Feed', image)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Save the collected data to a file
    np.save(DATA_FILE, data)
    print(f"Keypoints saved to {DATA_FILE}")

# Load data if the file exists
# if os.path.exists(DATA_FILE):
#     data = np.load(DATA_FILE)
#     print(f"Loaded keypoints from {DATA_FILE}")
# else:
#     print("No keypoints data found. Run the script to collect keypoints.")

# Execute only when the script is run directly
if __name__ == "__main__":
    collect_keypoints()
    # print("Keypoints collected:", data)
    if os.path.exists(DATA_FILE):
     data = np.load(DATA_FILE)
     print(f"Loaded keypoints from {DATA_FILE}")
    else:
     print("No keypoints data found. Run the script to collect keypoints.")

