from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
import config
from logger_setup import logger

class ImageLoader(QThread):
    sprite_loaded = pyqtSignal(int, list)
    loading_completed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.num_cols = config.num_cols
        self.num_rows = config.num_rows
        self.most_similar = []
        self.least_similar = []

    def set_data(self, most_similar, least_similar):
        if most_similar is None or least_similar is None:
            raise ValueError("most_similar or least_similar data cannot be None")
        self.most_similar = most_similar
        self.least_similar = least_similar

    def run(self):
        print('starting load ')
        combined_images = self.least_similar + self.most_similar

        sprites = [[] for _ in range(self.num_cols * self.num_rows)]
        loaded_labels = 0

        for image_info in combined_images:
            if loaded_labels >= self.num_cols * self.num_rows:
                break

            image_path = image_info['path']
            num_images = image_info['numImages']
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Image at path {image_path} could not be loaded")
                continue

            for i in range(num_images):
                x = (i % 19) * 100
                y = (i // 19) * 100
                cropped_image = image[y:y + 100, x:x + 100]
                if cropped_image.shape[0] == 100 and cropped_image.shape[1] == 100:
                    sprites[loaded_labels].append(cropped_image)
            self.sprite_loaded.emit(loaded_labels, sprites[loaded_labels])
            loaded_labels += 1

        logger.info("All images have been loaded.")
        print('done loading')
        self.loading_completed.emit()
