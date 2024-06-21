import cv2
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage
from mediapipe_face_detection import MediaPipeFaceDetection
import numpy as np
from logger_setup import logger

class VideoProcessor(QThread):
    frame_ready = pyqtSignal(QImage)

    def __init__(self, camera_index=0, square_size=300, callback=None):
        super().__init__()
        self.camera_index = camera_index
        self.square_size = square_size
        self.face_detector = MediaPipeFaceDetection()
        self.cap = cv2.VideoCapture(self.camera_index)
        self.callback = callback

        if not self.cap.isOpened():
            return

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(0)  # Process frames as quickly as possible

        # Initialize Kalman Filter
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

        # Increase process noise covariance
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-5  # Larger value for more smoothing

        # Increase measurement noise covariance
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10  # Larger value for more smoothing

        self.last_frame_with_face = None
        self.current_position = np.array([0, 0], dtype=np.float64)

    def run(self):
        self.exec_()

    def process_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                if self.last_frame_with_face is not None:
                    frame = self.last_frame_with_face
                else:
                    logger.error("No valid frame available.")
                    return  # Exit if no valid frame is available

            original_frame = frame.copy()  # Copy the full frame before processing

            frame, bbox = self.face_detector.detect_faces(frame, self.callback)
            if bbox:
                x, y, w, h = bbox
                cx, cy = x + w // 2, y + h // 2
                self.current_position = np.array([cx, cy], dtype=np.float64)
            else:
                if self.last_frame_with_face is not None:
                    frame = self.last_frame_with_face
                logger.info("No face detected in the current frame.")
                return  # Keep displaying the last position if no face detected

            # Kalman filter prediction
            prediction = self.kalman.predict()

            # Kalman filter correction
            measurement = np.array([[np.float32(self.current_position[0])], [np.float32(self.current_position[1])]])
            estimated = self.kalman.correct(measurement)

            # Calculate frame bounds using Kalman filter estimation
            cx, cy = estimated[0], estimated[1]
            half_w = self.square_size // 2
            half_h = self.square_size // 2

            # Adjust bounds to prevent exceeding the frame size
            x1, y1 = int(cx - half_w), int(cy - half_h)
            x2, y2 = int(cx + half_w), int(cy + half_h)

            if x1 < 0:
                x1 = 0
                x2 = self.square_size
                cx = half_w  # Adjust center accordingly
            if y1 < 0:
                y1 = 0
                y2 = self.square_size
                cy = half_h  # Adjust center accordingly
            if x2 > frame.shape[1]:
                x2 = frame.shape[1]
                x1 = frame.shape[1] - self.square_size
                cx = frame.shape[1] - half_w  # Adjust center accordingly
            if y2 > frame.shape[0]:
                y2 = frame.shape[0]
                y1 = frame.shape[0] - self.square_size
                cy = frame.shape[0] - half_h  # Adjust center accordingly

            # Extract frame
            frame = frame[y1:y2, x1:x2]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.square_size, self.square_size))
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.frame_ready.emit(q_img)

            # Save this frame as the last frame with a detected face
            self.last_frame_with_face = original_frame.copy()  # Save the full frame

        except Exception as e:
            logger.exception(f"Error processing frame: {e}")

    def stop(self):
        logger.info("Stopping VideoProcessor...")
        self.timer.stop()
        self.cap.release()
