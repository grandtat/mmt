import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QFileDialog, QWidget, QSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage


class HSVColorPicker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSV Color Range Picker")
        self.setGeometry(100, 100, 800, 600)

        # Initialize variables
        self.image_path = None
        self.original_image = None
        self.hsv_image = None
        self.display_image = None

        # HSV range variables
        self.hue_min = 0
        self.hue_max = 179
        self.sat_min = 0
        self.sat_max = 255
        self.val_min = 0
        self.val_max = 255

        # Create GUI elements
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()

        # Image display
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label)

        # Controls layout
        controls_layout = QVBoxLayout()

        # Load image button
        load_button = QPushButton("Load Image")
        load_button.clicked.connect(self.load_image)
        controls_layout.addWidget(load_button)

        # HSV sliders
        self.add_slider(controls_layout, "Hue Min", 0, 179, self.hue_min, self.update_mask_preview)
        self.add_slider(controls_layout, "Hue Max", 0, 179, self.hue_max, self.update_mask_preview)
        self.add_slider(controls_layout, "Sat Min", 0, 255, self.sat_min, self.update_mask_preview)
        self.add_slider(controls_layout, "Sat Max", 0, 255, self.sat_max, self.update_mask_preview)
        self.add_slider(controls_layout, "Val Min", 0, 255, self.val_min, self.update_mask_preview)
        self.add_slider(controls_layout, "Val Max", 0, 255, self.val_max, self.update_mask_preview)

        # Apply mask button
        apply_button = QPushButton("Apply Mask")
        apply_button.clicked.connect(self.apply_mask)
        controls_layout.addWidget(apply_button)

        # Add controls to main layout
        main_layout.addLayout(controls_layout)

        # Set central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def add_slider(self, layout, label_text, min_val, max_val, default_val, callback):
        layout_row = QHBoxLayout()

        # Label
        label = QLabel(label_text)
        layout_row.addWidget(label)

        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(callback)
        layout_row.addWidget(slider)

        # Spin box for precise input
        spin_box = QSpinBox()
        spin_box.setMinimum(min_val)
        spin_box.setMaximum(max_val)
        spin_box.setValue(default_val)
        spin_box.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(spin_box.setValue)
        layout_row.addWidget(spin_box)

        layout.addLayout(layout_row)

        # Store slider values in the instance
        if "Hue Min" in label_text:
            self.hue_min_slider = slider
        elif "Hue Max" in label_text:
            self.hue_max_slider = slider
        elif "Sat Min" in label_text:
            self.sat_min_slider = slider
        elif "Sat Max" in label_text:
            self.sat_max_slider = slider
        elif "Val Min" in label_text:
            self.val_min_slider = slider
        elif "Val Max" in label_text:
            self.val_max_slider = slider

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)

            if self.original_image is not None:
                # Convert to HSV
                self.hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)

                # Display original image
                self.display_image = self.original_image.copy()
                self.update_image_display()
            else:
                self.image_label.setText("Failed to load image")

    def update_image_display(self):
        if self.display_image is not None:
            # Convert from BGR to RGB
            img_rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)

            # Convert to QImage
            height, width, channel = img_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Convert to QPixmap and display
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    def update_mask_preview(self):
        if self.hsv_image is not None:
            # Get current HSV range values
            lower = np.array([self.hue_min_slider.value(), self.sat_min_slider.value(), self.val_min_slider.value()])
            upper = np.array([self.hue_max_slider.value(), self.sat_max_slider.value(), self.val_max_slider.value()])

            # Create mask
            mask = cv2.inRange(self.hsv_image, lower, upper)

            # Apply mask to original image (standard preview)
            self.display_image = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)

            # Update display
            self.update_image_display()

    def apply_mask(self):
        if self.hsv_image is not None:
            # Get current HSV range values
            lower = np.array([self.hue_min_slider.value(), self.sat_min_slider.value(), self.val_min_slider.value()])
            upper = np.array([self.hue_max_slider.value(), self.sat_max_slider.value(), self.val_max_slider.value()])

            # Create mask
            mask = cv2.inRange(self.hsv_image, lower, upper)

            # Convert original image to grayscale
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            # Apply color only where mask is active
            color_parts = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)
            inverse_mask = cv2.bitwise_not(mask)
            gray_parts = cv2.bitwise_and(gray_image, gray_image, mask=inverse_mask)

            # Combine grayscale and color parts
            self.display_image = cv2.add(gray_parts, color_parts)

            # Update display
            self.update_image_display()


if __name__ == "__main__":
    app = QApplication([])
    window = HSVColorPicker()
    window.show()
    app.exec_()