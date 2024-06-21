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
        print('Starting load')
        sprites = [[] for _ in range(self.num_cols * self.num_rows)]

        # Central grid coordinates
        center_row = config.num_rows // 2
        center_col = config.num_cols // 2

        # Define ranges for the central exclusion area
        middle_col_index = center_col
        mid_block_start_col = center_col - 4
        mid_block_end_col = center_col + 4
        mid_block_start_row = center_row - 1
        mid_block_end_row = center_row + 1

        # Generate grid positions, excluding middle column and central block
        positions = [
            (r, c) for r in range(config.num_rows) for c in range(config.num_cols)
            if c != middle_col_index and not (mid_block_start_col <= c <= mid_block_end_col and mid_block_start_row <= r <= mid_block_end_row)
        ]

        # Sort positions by their distance from the center of the grid
        positions.sort(key=lambda pos: (abs(pos[1] - center_col) ** 2 + abs(pos[0] - center_row) ** 2))

        # Load images starting from the closest to the center
        least_similar_index = 0
        most_similar_index = 0
        for pos in positions:
            row, col = pos
            grid_index = row * config.num_cols + col
            if col < center_col:
                # Load least similar images on the left side
                if least_similar_index < len(self.least_similar):
                    if load_and_append_image(self.least_similar[least_similar_index], grid_index, sprites):
                        self.sprite_loaded.emit(grid_index, sprites[grid_index])
                    least_similar_index += 1
            else:
                # Load most similar images on the right side
                if most_similar_index < len(self.most_similar):
                    if load_and_append_image(self.most_similar[most_similar_index], grid_index, sprites):
                        self.sprite_loaded.emit(grid_index, sprites[grid_index])
                    most_similar_index += 1

        self.loading_completed.emit()

def load_and_append_image(image_info, grid_index, sprites):
    image = cv2.imread(image_info['path'])
    if image is None:
        logger.error(f"Image at path {image_info['path']} could not be loaded")
        return False

    for i in range(image_info['numImages']):
        x = (i % 19) * 100
        y = (i // 19) * 100
        cropped_image = image[y:y + 100, x:x + 100]
        if cropped_image.shape[0] == 100 and cropped_image.shape[1] == 100:
            sprites[grid_index].append(cropped_image)
    return True