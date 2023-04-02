import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import os

from matting_functions import compositing
from Bayesian_matte import Bayesian_Matte






image = np.array(Image.open('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/input_training_lowres/GT10.png'))
image_trimap = np.array(ImageOps.grayscale(Image.open('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/trimap_training_lowres/Trimap1/GT10.png')))

alpha = Bayesian_Matte(image, image_trimap)


image_alpha = np.array(ImageOps.grayscale(Image.open('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/gt_training_lowres/GT10.png')))

#alpha2 = Image.fromarray((alpha * 255).astype(np.uint8))

alpha = alpha*255


# # save the alpha matte
# alpha2.save('alpha.png')

#plt.imsave("alpha.png", alpha, cmap='gray')
plt.imshow(alpha, cmap='gray')
plt.show()

#alpha_matte = np.array(Image.open('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/alpha.png'))

background = cv2.imread('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/background.jpg')
background2 = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)


comp = compositing(image,alpha,background2)
plt.imshow(comp)
plt.show()




