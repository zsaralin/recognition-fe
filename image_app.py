import sys
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QGridLayout, QWidget, QVBoxLayout, QSpacerItem, QSizePolicy, QShortcut
from PyQt5.QtGui import QKeySequence, QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import config
from video_processor import VideoProcessor
from image_loader import ImageLoader
from backend_communicator import send_snapshot_to_server
from new_faces import set_curr_face, update_face_detection
from mediapipe_face_detection import MediaPipeFaceDetection
from logger_setup import logger

class ImageApp(QWidget):
    def __init__(self):
        super().__init__()
        print("Initializing ImageApp.")
        self.sprites = []
        self.sprite_indices = []
        self.animating_labels = set()
        self.image_loader_thread = None
        self.initUI()

        # Initialize the VideoProcessor
        self.video_processor = VideoProcessor(square_size=self.square_size * 3, callback=self.load_images)
        self.video_processor.frame_ready.connect(self.update_video_label)
        print("Starting VideoProcessor in ImageApp.")
        self.video_processor.start()

        # Set up a timer to update sprites
        self.sprite_timer = QTimer(self)
        self.sprite_timer.timeout.connect(self.update_sprites)
        self.sprite_timer.start(33)  # Update sprite animation at ~30 fps

    def initUI(self):
        print("Setting up UI.")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.setWindowTitle('Image Display App')

        screen_sizes = [(screen.size().width(), screen.size().height()) for screen in QApplication.screens()]
        largest_screen_width, largest_screen_height = max(screen_sizes, key=lambda s: s[0] * s[1])
        print(f"Largest screen size: width={largest_screen_width}, height={largest_screen_height}")

        window_width = largest_screen_width // 2
        window_height = largest_screen_height
        self.setFixedSize(window_width, window_height)
        print(f"Window dimensions set: width={window_width}, height={window_height}")

        self.num_cols = 21
        self.square_size = window_width // self.num_cols
        print(f"Number of columns: {self.num_cols}, square size: {self.square_size}")

        self.num_rows = window_height // self.square_size
        print(f"Number of rows: {self.num_rows}")

        config.num_cols = self.num_cols
        config.num_rows = self.num_rows

        config.num_vids = self.num_rows * self.num_cols
        print(f"Number of videos: {config.num_vids}")

        grid_widget = QWidget()
        grid_layout = QGridLayout()
        grid_layout.setSpacing(0)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_widget.setLayout(grid_layout)
        print("Grid layout created")

        spacer_top = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        spacer_bottom = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout.addItem(spacer_top)
        self.layout.addWidget(grid_widget)
        self.layout.addItem(spacer_bottom)
        print("Grid layout added to main layout with spacers")

        self.image_labels = []
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                label = QLabel(self)
                label.setFixedSize(self.square_size, self.square_size)
                label.setStyleSheet("border: 1px solid black")
                label.setAlignment(Qt.AlignCenter)
                grid_layout.addWidget(label, row, col)
                self.image_labels.append(label)
                self.sprites.append([])
                self.sprite_indices.append(0)
        print("Image labels created and added to grid layout")

        center_row = self.num_rows // 2
        center_col = self.num_cols // 2

        video_label_width = self.square_size * 3
        video_label_height = self.square_size * 3
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(video_label_width, video_label_height)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid black; margin: 1px;")
        grid_layout.addWidget(self.video_label, center_row - 1, center_col - 1, 3, 3)
        print("Video label created and added to grid layout")

        self.show()
        print("Main window displayed")

        self.shortcut = QShortcut(QKeySequence("Escape"), self)
        self.shortcut.activated.connect(self.close)
        print("Escape shortcut set up")

    def closeEvent(self, event):
        print("Close event triggered")
        if hasattr(self, 'video_processor'):
            print("Stopping VideoProcessor.")
            self.video_processor.stop()
        if self.image_loader_thread is not None:
            self.image_loader_thread.quit()
            self.image_loader_thread.wait()
        event.accept()
        print("Close event accepted")

    def handle_sprite_loaded(self, label_index, sprites):
        self.sprites[label_index] = sprites

    def handle_loading_completed(self):
        print("All images have been loaded.")
        logger.info("All images have been loaded.")

    def update_sprites(self):
        for i in range(len(self.image_labels)):
            if self.sprites[i]:
                self.image_labels[i].setPixmap(self.cv2_to_qpixmap(self.sprites[i][self.sprite_indices[i]], self.square_size))
                self.sprite_indices[i] = (self.sprite_indices[i] + 1) % len(self.sprites[i])

    def update_video_label(self, q_img):
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def cv2_to_qpixmap(self, cv_img, size):
        height, width, channel = cv_img.shape
        bytes_per_line = channel * width
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        q_img = QImage(cv_img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img).scaled(size, size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    def load_images(self, most_similar, least_similar):
        # Initialize and start the ImageLoader thread
        self.image_loader_thread = QThread()
        self.image_loader = ImageLoader()
        self.image_loader.moveToThread(self.image_loader_thread)
        self.image_loader.set_data(most_similar, least_similar)
        self.image_loader.sprite_loaded.connect(self.handle_sprite_loaded)
        self.image_loader.loading_completed.connect(self.handle_loading_completed)
        self.image_loader_thread.started.connect(self.image_loader.run)
        self.image_loader_thread.start()
