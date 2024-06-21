import cv2
import numpy as np
import requests
import base64
import config
from logger_setup import logger
from PyQt5.QtCore import QThread, QTimer
from image_loader import ImageLoader

SERVER_URL = "http://localhost:3000/get-matches"

def convert_image_to_data_url(image):
    if image is None:
        logger.error("convert_image_to_data_url: image is None")
        return None

    _, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{jpg_as_text}"
    return data_url

def send_snapshot_to_server(frame, callback):
    if frame is None:
        logger.error("send_snapshot_to_server: frame is None")
        return None, None, False

    image_data_url = convert_image_to_data_url(frame)
    if image_data_url is None:
        logger.error("send_snapshot_to_server: Failed to convert frame to data URL")
        return None, None, False

    payload = {'image': image_data_url, 'numVids': config.num_vids}

    try:
        response = requests.post(SERVER_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            most_similar = result.get('mostSimilar')
            least_similar = result.get('leastSimilar')
            logger.info(f"Most Similar: {most_similar}")
            logger.info(f"Least Similar: {least_similar}")

            if most_similar is None or least_similar is None:
                logger.error("Received None for most_similar or least_similar")
                return None, None, False

            # Call the callback function with the results
            callback(most_similar, least_similar)
            return most_similar, least_similar, True
        else:
            logger.error(f"Failed to get matches from server: {response.status_code}")
            logger.error(f"Server response: {response.text}")
            if response.status_code == 404 and "No face detected" in response.text:
                return None, None, False
    except Exception as e:
        logger.exception("Error sending snapshot to server: %s", e)

    return None, None, False
