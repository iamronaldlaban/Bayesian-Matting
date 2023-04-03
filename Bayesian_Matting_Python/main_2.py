import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import os
from matting_functions import compositing
from Bayesian_matte_OB import Bayesian_Matte
from Bayesian_matte_KNN import Bayesian_Matte_KNN


image = np.array(Image.open('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/input_training_lowres/GT10.png'))
image_trimap = np.array(ImageOps.grayscale(Image.open('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/trimap_training_lowres/Trimap1/GT10.png')))
alpha_OB= Bayesian_Matte(image, image_trimap)
alpha_knn = Bayesian_Matte_KNN(image, image_trimap)

image_alpha = np.array(ImageOps.grayscale(Image.open('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/gt_training_lowres/GT10.png')))

# Plot both images side by side using subplot
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Display Orchard-Bouman image on the left
alpha_OB = alpha_OB*255
ax[0].imshow(alpha_OB, cmap='gray')
ax[0].set_title('Orchard-Bouman')

# Display KNN image on the right
alpha_KNN = alpha_knn*255
ax[1].imshow(alpha_KNN, cmap='gray')
ax[1].set_title('KNN')

# Display the plot
plt.show()


background = cv2.imread('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/background.jpg')
background2 = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

# Plot both composites side by side using subplot
# Create both composites
comp_OB = compositing(image, alpha_OB, background2)
comp_KNN = compositing(image, alpha_knn, background2)

# Plot both composites side by side using subplot
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Display Orchard-Bouman composite on the left
ax[0].imshow(comp_OB)
ax[0].set_title('Orchard-Bouman')

# Display KNN composite on the right
ax[1].imshow(comp_KNN)
ax[1].set_title('KNN')

# Display the plot
plt.show()