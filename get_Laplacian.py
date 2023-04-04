import numpy as np
from scipy.ndimage import laplace
from PIL import Image, ImageOps

image = np.array(Image.open('C:/Users/Chengyu/Desktop/Bayesian-Matting-main/Bayesian_Matting_Python/input_training_lowres/GT02.png'))
trimap = np.array(ImageOps.grayscale(Image.open('C:/Users/Chengyu/Desktop/Bayesian-Matting-main/Bayesian_Matting_Python/trimap_training_lowres/Trimap2/GT02.png')))

def get_Laplacian(input_img, trimap):
    # Convert the image and trimap to double precision and normalize
    img = input_img.astype(np.float64)
    trimap = trimap.astype(np.float64)

    # Get the size of the image
    m, n, c = img.shape

    # Calculate the foreground, background, and unknown pixels
    fg = trimap > 0.99
    bg = trimap < 0.01
    unk = ~(fg | bg)

    # Create the Laplacian matrix
    Laplacian = laplace(img)

    # Calculate the alpha matte
    alpha = np.zeros((m, n))
    for i in range(c):
        alpha[unk[:, :, 0]] += Laplacian[unk[:, :, 0], i] ** 2
    
    alpha = 1 - np.sqrt(alpha / c)
    alpha[bg[:, :, 0]] = 0
    alpha[fg[:, :, 0]] = 1

    return alpha
