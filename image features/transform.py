import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class HomographyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Homography Transformation")
        
        # Variables for images
        self.img1 = None
        self.img2 = None
        self.img1_gray = None
        self.img2_gray = None
        self.result_img = None
        
        # GUI elements
        self.init_ui()
    
    def init_ui(self):
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Buttons
        self.btn_load1 = QPushButton("Load Image 1")
        self.btn_load1.clicked.connect(self.load_image1)
        self.layout.addWidget(self.btn_load1)
        
        self.btn_load2 = QPushButton("Load Image 2")
        self.btn_load2.clicked.connect(self.load_image2)
        self.layout.addWidget(self.btn_load2)
        
        self.btn_transform = QPushButton("Transform")
        self.btn_transform.setEnabled(False)
        self.btn_transform.clicked.connect(self.compute_homography)
        self.layout.addWidget(self.btn_transform)
        
        # Image display labels
        self.label_img1 = QLabel()
        self.label_img1.setFixedSize(400, 300)
        self.label_img1.setStyleSheet("background-color: gray;")
        self.layout.addWidget(self.label_img1)
        
        self.label_img2 = QLabel()
        self.label_img2.setFixedSize(400, 300)
        self.label_img2.setStyleSheet("background-color: gray;")
        self.layout.addWidget(self.label_img2)
        
        self.label_result = QLabel()
        self.label_result.setFixedSize(400, 300)
        self.label_result.setStyleSheet("background-color: gray;")
        self.layout.addWidget(self.label_result)
    
    def load_image1(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image 1", "", "Image Files (*.jpg *.png *.jpeg)")
        if file_path:
            self.img1 = cv2.imread(file_path)
            self.img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            self.display_image(self.img1, self.label_img1)
            self.check_images_loaded()
    
    def load_image2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image 2", "", "Image Files (*.jpg *.png *.jpeg)")
        if file_path:
            self.img2 = cv2.imread(file_path)
            self.img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
            self.display_image(self.img2, self.label_img2)
            self.check_images_loaded()
    
    def check_images_loaded(self):
        if self.img1 is not None and self.img2 is not None:
            self.btn_transform.setEnabled(True)
    
    def display_image(self, img, label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = channel * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
    
    def compute_homography(self):
        # Feature Matching with SIFT
        sift = cv2.SIFT_create()
        kp1, desc1 = sift.detectAndCompute(self.img1_gray, None)
        kp2, desc2 = sift.detectAndCompute(self.img2_gray, None)
        
        # BFMatcher with L2 norm
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        # Lowe's Ratio Test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.55 * n.distance:
                good_matches.append(m)
        
        # Compute Homography (at least 4 points)
        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # Apply transformation
            height, width = self.img2.shape[:2]
            self.result_img = cv2.warpPerspective(self.img1, H, (width, height))
            
            # Display result
            self.display_image(self.result_img, self.label_result)
        else:
            self.show_error("Not enough matches for homography (at least 4 required)!")
    
    def show_error(self, message):
        error_dialog = QFileDialog(self)
        error_dialog.setWindowTitle("Error")
        error_dialog.setLabelText(message)
        error_dialog.exec_()

# Main application
app = QApplication([])
window = HomographyApp()
window.show()
app.exec_()