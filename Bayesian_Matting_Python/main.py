import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import os
from matting_functions import compositing
from Bayesian_matte_OB import Bayesian_Matte
from Bayesian_matte_KNN import Bayesian_Matte_KNN
from quality_metrics import quality_metrics
from timeit import default_timer as timer
import datetime
import tkinter as tk
from tkinter import filedialog
import unittest
from matting_functions import matlab_style_gauss2d


import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import numpy as np
from PIL import Image, ImageOps

app = QApplication(sys.argv)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window Title
        self.setWindowTitle("Bayesian Matte")

        # Load Image Button
        self.load_image_button = QPushButton("Load Image", self)
        self.load_image_button.setGeometry(50, 50, 100, 30)
        self.load_image_button.clicked.connect(self.load_image)

        # Load Trimap Button
        self.load_trimap_button = QPushButton("Load Trimap", self)
        self.load_trimap_button.setGeometry(50, 100, 100, 30)
        self.load_trimap_button.clicked.connect(self.load_trimap)

        # Run Algorithm Button
        self.run_button = QPushButton("Run Algorithm", self)
        self.run_button.setGeometry(50, 150, 100, 30)
        self.run_button.clicked.connect(self.run_algorithm)

        # Image Display
        self.image_label = QLabel(self)
        self.image_label.setGeometry(200, 50, 400, 400)

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.xpm *.jpg *.bmp *.tif)", options=options)
        if file_name:
            self.image = np.array(Image.open(file_name))
            pixmap = QPixmap(file_name).scaled(400, 400, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

    def load_trimap(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Trimap", "", "Images (*.png *.xpm *.jpg *.bmp *.tif)", options=options)
        if file_name:
            self.trimap = np.array(ImageOps.grayscale(Image.open(file_name)))
            pixmap = QPixmap(file_name).scaled(400, 400, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

    def run_algorithm(self):
        N = 105
        alpha_OB = Bayesian_Matte(self.image, self.trimap, N)
        alpha_OB = np.uint8(alpha_OB * 255)
        alpha_OB_image = QImage(alpha_OB.data, alpha_OB.shape[1], alpha_OB.shape[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(alpha_OB_image).scaled(400, 400, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)


window = MainWindow()
window.show()
sys.exit(app.exec_())
