# main.py (or wherever your main application code is)
import sys
from PyQt5.QtWidgets import QApplication
from image_app import ImageApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageApp()
    window.show()
    app.exec_()

    # Ensure all threads are stopped properly
    window.video_processor.stop()
    if window.image_loader and window.image_loader.isRunning():
        window.image_loader.quit()
        window.image_loader.wait()
