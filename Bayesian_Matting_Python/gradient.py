import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image



def gradient_magnitude(img):

    # Load an image as a grayscale image
    #img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # Calculate the gradient in x and y direction using Sobel operator
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the magnitude and direction of the gradient
    magnitude = np.sqrt(dx**2 + dy**2)
    direction = np.arctan2(dy, dx)

    #print(np.mean(magnitude))

    return magnitude

def gradient_error(alpha_gt, alpha_est):
    grad_gt = gradient_magnitude(alpha_gt)
    #print(grad_gt)
    grad_est = gradient_magnitude(alpha_est)
    #print(grad_est)
    mean_grad_gt = np.mean(grad_gt)
    a = grad_gt[:,:,1]/np.max(grad_gt[:,:,1])
    b = grad_est/np.max(grad_est)
    ge = np.abs(a - b) 
    #print(ge)
    return np.mean(ge)



# Load image as numpy array
alpha = np.array(Image.open('alpha.png'))
gt_alpha = np.array(Image.open('gt_training_lowres\GT04.png'))

# Define the standard deviation and size of the filter
sigma = 0.5
size = 10

# Apply Gaussian derivative filter along the x-axis
alpha_filtered_img = gaussian_filter1d(alpha, sigma=sigma, order=1, axis=1, mode='reflect')
gt_alpha_filtered_img = gaussian_filter1d(gt_alpha, sigma=sigma, order=1, axis=1, mode='reflect')

# Display the filtered image
#Image.fromarray(dx_filtered_img.astype(np.uint8)).show()

print(gradient_error(gt_alpha_filtered_img,alpha_filtered_img))


