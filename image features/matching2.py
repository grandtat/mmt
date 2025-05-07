import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QComboBox, QStatusBar
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QEvent


class FeatureMatcherGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feature Matcher")
        self.showFullScreen()  # Open the UI in fullscreen mode
        self.installEventFilter(self)  # Install event filter for key press handling
        
        # Variables
        self.img1 = None
        self.img2 = None
        
        # Main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Image display
        self.image_layout = QVBoxLayout()  # Change to vertical layout
        self.top_image_layout = QHBoxLayout()  # Add a horizontal layout for the top images
        self.img1_label = QLabel("Bild 1")
        self.img1_label.setAlignment(Qt.AlignCenter)
        self.img2_label = QLabel("Bild 2")
        self.img2_label.setAlignment(Qt.AlignCenter)
        self.result_label = QLabel("Matches")
        self.result_label.setAlignment(Qt.AlignCenter)
        
        # Add image1 and image2 to the top layout
        self.top_image_layout.addWidget(self.img1_label)
        self.top_image_layout.addWidget(self.img2_label)
        
        # Add the top layout and result label to the main image layout
        self.image_layout.addLayout(self.top_image_layout)
        self.image_layout.addWidget(self.result_label)
        self.layout.addLayout(self.image_layout)
        
        # Controls
        self.control_layout = QHBoxLayout()  # Horizontal layout for side-by-side controls
        
        # Load image buttons
        self.load_buttons_layout = QVBoxLayout()  # Vertical layout for load buttons
        self.load_img1_btn = QPushButton("Bild 1 laden")
        self.load_img1_btn.clicked.connect(lambda: self.load_image(1))
        self.load_img2_btn = QPushButton("Bild 2 laden")
        self.load_img2_btn.clicked.connect(lambda: self.load_image(2))
        self.load_buttons_layout.addWidget(self.load_img1_btn)
        self.load_buttons_layout.addWidget(self.load_img2_btn)
        self.control_layout.addLayout(self.load_buttons_layout)  # Add load buttons layout to control layout
        
        # Feature method selection
        self.method_label = QLabel("Feature-Methode:")
        self.method_combobox = QComboBox()
        self.method_combobox.addItems(["SIFT", "ORB", "AKAZE"])
        self.control_layout.addWidget(self.method_label)
        self.control_layout.addWidget(self.method_combobox)
        
        # Matcher type selection
        self.matcher_label = QLabel("Matcher-Typ:")
        self.matcher_combobox = QComboBox()
        self.matcher_combobox.addItems(["BruteForce", "FLANN"])
        self.control_layout.addWidget(self.matcher_label)
        self.control_layout.addWidget(self.matcher_combobox)
        
        # Match button
        self.match_btn = QPushButton("Features matchen")
        self.match_btn.setEnabled(False)
        self.match_btn.clicked.connect(self.match_features)
        self.control_layout.addWidget(self.match_btn)
        
        self.layout.addLayout(self.control_layout)  # Add the control layout to the main layout
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
    
    def load_image(self, img_num):
        file_path, _ = QFileDialog.getOpenFileName(self, "Bild laden", "", "Image files (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if img_num == 1:
                    self.img1 = img
                    self.show_image(img, self.img1_label)
                else:
                    self.img2 = img
                    self.show_image(img, self.img2_label)
                
                if self.img1 is not None and self.img2 is not None:
                    self.match_btn.setEnabled(True)
                
                self.status_bar.showMessage(f"Bild {img_num} geladen: {file_path}")
    
    def show_image(self, img, label_widget):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label_widget.setPixmap(pixmap.scaled(label_widget.width(), label_widget.height(), Qt.KeepAspectRatio))
    
    def match_features(self):
        if self.img1 is None or self.img2 is None:
            return
            
        # Convert to grayscale
        gray1 = cv2.cvtColor(self.img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(self.img2, cv2.COLOR_RGB2GRAY)
        
        method = self.method_combobox.currentText()
        matcher_type = self.matcher_combobox.currentText()
        
        # Initialize feature detector
        if method == "SIFT":
            detector = cv2.SIFT_create()
        elif method == "ORB":
            detector = cv2.ORB_create(nfeatures=1000)
        elif method == "AKAZE":
            detector = cv2.AKAZE_create()
        else:
            detector = cv2.SIFT_create()
        
        # Detect keypoints and descriptors
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)
        
        if desc1 is None or desc2 is None:
            self.status_bar.showMessage("Fehler: Keine Features gefunden!")
            return
        
        # Convert AKAZE descriptors for FLANN if needed
        if method == "AKAZE" and matcher_type == "FLANN":
            desc1 = np.float32(desc1)
            desc2 = np.float32(desc2)
        
        # Initialize matcher
        if matcher_type == "BruteForce":
            if method in ["SIFT", "AKAZE"]:
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            else:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)[:50]
            
        else:  # FLANN
            # Set FLANN parameters
            if method in ["SIFT", "AKAZE"]:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
            else:
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                   table_number=6,
                                   key_size=12,
                                   multi_probe_level=1)
                search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)[:50]
        
        # Create result image
        result_img = cv2.drawMatches(
            self.img1, kp1,
            self.img2, kp2,
            matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0)
        )
        
        # Display result
        self.show_image(result_img, self.result_label)
        self.status_bar.showMessage(f"{method} + {matcher_type}: {len(matches)} Matches gefunden")
        cv2.imwrite("match_eval.png", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self.close()  # Close the application when Esc is pressed
        return super().eventFilter(source, event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FeatureMatcherGUI()
    window.show()
    sys.exit(app.exec_())