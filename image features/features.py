import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QFileDialog, QWidget, QStatusBar
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image

class FeatureVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feature Scale Visualizer")
        
        # Variables
        self.image = None
        self.keypoints = None
        
        # Main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)
        
        # Control layout
        control_layout = QHBoxLayout()
        self.layout.addLayout(control_layout)
        
        # Load image button
        load_btn = QPushButton("Bild laden")
        load_btn.clicked.connect(self.load_image)
        control_layout.addWidget(load_btn)
        
        # Method selection
        control_layout.addWidget(QLabel("Methode:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["SIFT", "ORB", "AKAZE"])
        control_layout.addWidget(self.method_combo)
        
        # Detect features button
        detect_btn = QPushButton("Features anzeigen")
        detect_btn.clicked.connect(self.detect_features)
        control_layout.addWidget(detect_btn)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Bild laden", "", "Image files (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is not None:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.show_image(self.image)
                self.status_bar.showMessage(f"Bild geladen: {file_path}")
    
    def show_image(self, img):
        img_pil = Image.fromarray(img)
        
        # Scaling for display
        max_size = 800
        if img_pil.width > max_size or img_pil.height > max_size:
            ratio = min(max_size / img_pil.width, max_size / img_pil.height)
            new_size = (int(img_pil.width * ratio), int(img_pil.height * ratio))
            img_pil = img_pil.resize(new_size, Image.LANCZOS)
        
        img_qt = QImage(img_pil.tobytes(), img_pil.width, img_pil.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img_qt)
        self.image_label.setPixmap(pixmap)
    
    def detect_features(self):
        if self.image is None:
            return
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        method = self.method_combo.currentText()
        
        # Initialize feature detector
        if method == "SIFT":
            detector = cv2.SIFT_create()
        elif method == "ORB":
            detector = cv2.ORB_create(nfeatures=200)
        elif method == "AKAZE":
            detector = cv2.AKAZE_create()
        else:
            detector = cv2.SIFT_create()
        
        # Detect keypoints
        self.keypoints = detector.detect(gray, None)
        img_with_kp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        for kp in self.keypoints:
            radius = int(getattr(kp, 'size', 10) / 2)
            radius = max(3, min(radius, 50))
            
            if hasattr(kp, 'angle'):
                try:
                    hsv_color = np.uint8([[[kp.angle % 180, 255, 255]]])
                    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
                    color = (int(bgr_color[0, 0, 2]), int(bgr_color[0, 0, 1]), int(bgr_color[0, 0, 0]))
                except Exception:
                    color = (0, 255, 0)
            else:
                color = (0, 255, 0)
            
            if all(isinstance(c, (int, float)) for c in color):
                pt = (int(kp.pt[0]), int(kp.pt[1]))
                cv2.circle(img_with_kp, pt, radius, color, 1)
                
                if hasattr(kp, 'angle'):
                    end_x = int(kp.pt[0] + radius * np.cos(np.radians(kp.angle)))
                    end_y = int(kp.pt[1] + radius * np.sin(np.radians(kp.angle)))
                    cv2.line(img_with_kp, pt, (end_x, end_y), color, 1)
        
        self.show_image(img_with_kp)
        self.status_bar.showMessage(f"{method}: {len(self.keypoints)} Features - Kreisgröße = Skala")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FeatureVisualizer()
    window.show()
    sys.exit(app.exec_())