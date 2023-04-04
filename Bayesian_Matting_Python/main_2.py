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


# root = tk.Tk()
# root.withdraw() # Hide the main window

# image = filedialog.askopenfilename()
# image_trimap = filedialog.askopenfilename()

#image = np.array(Image.open('High_Resolution/input_training_highres/GT04.png'))
#image_trimap = np.array(ImageOps.grayscale(Image.open('High_Resolution/trimap_training_highres/Trimap1/GT04.png')))
image = np.array(Image.open('input_training_lowres\GT01.png'))
image_trimap = np.array(ImageOps.grayscale(Image.open('trimap_training_lowres\Trimap1\GT01.png')))

# Window Size
N = 105

start = timer()

alpha_OB= Bayesian_Matte(image, image_trimap, N)
end = timer()
# alpha_OB[alpha_OB < 0.5] = 0
# alpha_OB[alpha_OB >= 0.5] = 1
alpha_OB = alpha_OB*255

# write_path = 'C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/alpha.png'
# cv2.imwrite(write_path, alpha_OB)

# image_alpha = np.array(ImageOps.grayscale(Image.open('High_Resolution/gt_training_highres/GT04.png')))
ground_truth = np.array(ImageOps.grayscale(Image.open('gt_training_lowres/GT01.png')))
quality_metrics(alpha_OB, ground_truth)
print('Time taken: ', datetime.timedelta(seconds = end - start))
# plt.imshow(alpha_OB, cmap='gray')
# plt.show()
background = cv2.imread('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/background.jpg')
background2 = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

# comp_OB = compositing(image, alpha_OB, background2)
# plt.imshow(comp_OB)
# plt.show()


# alpha_knn = Bayesian_Matte_KNN(image, image_trimap)

# image_alpha = np.array(ImageOps.grayscale(Image.open('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/gt_training_lowres/GT10.png')))

# # Plot both images side by side using subplot
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# # Display Orchard-Bouman image on the left
# alpha_OB = alpha_OB*255
# ax[0].imshow(alpha_OB, cmap='gray')
# ax[0].set_title('Orchard-Bouman')

# # Display KNN image on the right
# alpha_KNN = alpha_knn*255
# ax[1].imshow(alpha_KNN, cmap='gray')
# ax[1].set_title('KNN')

# # Display the plot
# plt.show()


# background = cv2.imread('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/background.jpg')
# background2 = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

# # Plot both composites side by side using subplot
# # Create both composites
# comp_OB = compositing(image, alpha_OB, background2)
# comp_KNN = compositing(image, alpha_knn, background2)

# # Plot both composites side by side using subplot
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# # Display Orchard-Bouman composite on the left
# ax[0].imshow(comp_OB)
# ax[0].set_title('Orchard-Bouman')

# # Display KNN composite on the right
# ax[1].imshow(comp_KNN)
# ax[1].set_title('KNN')

# # Display the plot
# plt.show()
from numpy.testing import assert_allclose

class TestBayesianMatte(unittest.TestCase):
    def test_image_and_alpha_size(self):
        print('\nTesting if the image size and alpha size are the same\n')
        
        # Load the images
        try:
            input_image = image
            image_alpha = alpha_OB
        except IOError:
            self.fail("Failed to load images")

        # Check if the images have the same dimensions
        try:
            self.assertEqual(input_image.shape[0:2], image_alpha.shape)
        except AssertionError:
            self.fail("Image and alpha have different dimensions")
        
        # If everything is proper, assert that the test passes successfully
        self.assertTrue(True, "The images have the same dimensions")

    def test_window_size_is_odd(self):
        self.assertNotEqual(N % 2, 0, "N is not odd")

    def test_images(self):
        # Load example images
        input_image = image
        trimap_image = image_trimap
        ground_truth_image = ground_truth
        # Check if images are not None
        assert input_image is not None, "Input image is not loaded"
        assert trimap_image is not None, "Trimap image is not loaded"
        assert ground_truth_image is not None, "Ground truth image is not loaded"
        # Check if images are valid
        assert input_image.shape[:2] == trimap_image.shape[:2] == ground_truth_image.shape[:2], "Images have different sizes"
    
if __name__ == '__main__':
    unittest.main()