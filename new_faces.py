import cv2
import numpy as np
from backend_communicator import send_snapshot_to_server
from logger_setup import logger

curr_face = None
no_face_counter = 0  # Counter for consecutive frames with no face detected

def set_curr_face(mediapipe_result, frame, callback):
    global curr_face, no_face_counter
    if mediapipe_result and mediapipe_result.detections:
        no_face_counter = 0  # Reset counter if a face is detected
        update_face_detection(frame, callback)
    else:
        no_face_counter += 1
        if no_face_counter >= 10:
            curr_face = None
            no_face_counter = 0  # Reset the counter
            print('No face detected for 10 consecutive frames, resetting curr_face.')
            logger.info("No face detected for 10 consecutive frames, resetting curr_face.")

def update_face_detection(frame, callback):
    global curr_face
    is_new_face = curr_face is None

    # Validate the frame
    if frame is None:
        logger.error("update_face_detection: frame is None")
        return

    if not isinstance(frame, np.ndarray):
        logger.error(f"update_face_detection: frame is not a valid numpy array. Type: {type(frame)}")
        return

    if is_new_face:
        print('New face detected')
        logger.info("New face detected")
        curr_face = frame  # Store the original frame
        send_snapshot_to_server(curr_face, callback)
    else:
        curr_face = frame  # Update the current frame
