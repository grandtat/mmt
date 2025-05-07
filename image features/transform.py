import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class HomographyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Homography Transformation")
        
        # Variablen für Bilder
        self.img1 = None
        self.img2 = None
        self.img1_gray = None
        self.img2_gray = None
        self.result_img = None
        
        # GUI-Elemente erstellen
        self.create_widgets()
    
    def create_widgets(self):
        # Buttons
        self.btn_load1 = tk.Button(self.root, text="Bild 1 laden", command=self.load_image1)
        self.btn_load1.pack(pady=5)
        
        self.btn_load2 = tk.Button(self.root, text="Bild 2 laden", command=self.load_image2)
        self.btn_load2.pack(pady=5)
        
        self.btn_transform = tk.Button(self.root, text="Transform", command=self.compute_homography, state=tk.DISABLED)
        self.btn_transform.pack(pady=10)
        
        # Bild-Anzeige (Canvas)
        self.canvas1 = tk.Canvas(self.root, width=400, height=300, bg="gray")
        self.canvas1.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.canvas2 = tk.Canvas(self.root, width=400, height=300, bg="gray")
        self.canvas2.pack(side=tk.RIGHT, padx=10, pady=10)
        
        self.canvas_result = tk.Canvas(self.root, width=400, height=300, bg="gray")
        self.canvas_result.pack(pady=10)
    
    def load_image1(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.img1 = cv2.imread(file_path)
            self.img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            self.display_image(self.img1, self.canvas1)
            self.check_images_loaded()
    
    def load_image2(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.img2 = cv2.imread(file_path)
            self.img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
            self.display_image(self.img2, self.canvas2)
            self.check_images_loaded()
    
    def check_images_loaded(self):
        if self.img1 is not None and self.img2 is not None:
            self.btn_transform.config(state=tk.NORMAL)
    
    def display_image(self, img, canvas):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((400, 300), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        
        canvas.image = img_tk  # Referenz halten, um Garbage Collection zu vermeiden
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    
    def compute_homography(self):
        # Feature Matching mit SIFT
        sift = cv2.SIFT_create()
        kp1, desc1 = sift.detectAndCompute(self.img1_gray, None)
        kp2, desc2 = sift.detectAndCompute(self.img2_gray, None)
        
        # BFMatcher mit Hamming-Distanz (für binäre Features: ORB/BRIEF würden cv2.NORM_HAMMING nutzen)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        # Lowe's Ratio Test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.55 * n.distance:
                good_matches.append(m)
        
        # Homographie berechnen (mind. 4 Punkte)
        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # Transformation anwenden
            height, width = self.img2.shape[:2]
            self.result_img = cv2.warpPerspective(self.img1, H, (width, height))
            
            # Ergebnis anzeigen
            self.display_image(self.result_img, self.canvas_result)
        else:
            tk.messagebox.showerror("Fehler", "Nicht genügend Matches für Homographie (mind. 4 benötigt)!")

# Hauptfenster starten
root = tk.Tk()
app = HomographyApp(root)
root.mainloop()