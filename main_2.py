import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import os
from numba import jit
from orchard_bouman_clust import clustFunc
#from Bayesian_Matte import Bayesian_Matte
import unittest

# GAUSSIAN FILTER


def matlab_style_gauss2d(shape=(3, 3), sigma=0.5):
    """
    Returns a 2D Gaussian filter with the specified shape and standard deviation.

    Args:
        shape: A tuple of two integers specifying the dimensions of the output filter. The default value is (3, 3).
        sigma: A float specifying the standard deviation of the Gaussian filter. The default value is 0.5.

    Returns:
        A two-dimensional NumPy array containing the Gaussian filter. The dimensions of the array are determined by
        the shape argument, and the values of the array are determined by the Gaussian function with the specified sigma.
        The sum of the values in the array will be approximately equal to 1.
    """
    m, n = [(ss-1)/2 for ss in shape]
    x, y = np.meshgrid(np.arange(-n, n+1), np.arange(-m, m+1))
    h = np.exp(-(x**2 + y**2)/(2*sigma**2))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    h /= np.sum(h)
    return h


def show_im(img):
    """
    img - input image should be a numpy array.
    """
    plt.imshow(img)
    plt.show()


@jit(nopython=True, cache=True)
def get_window(image, x_center, y_center, window_size):
    height, width, channels = image.shape
    half_window_size = window_size // 2
    window = np.zeros((window_size, window_size, channels))

    xmin = max(0, x_center - half_window_size)
    xmax = min(width, x_center + (half_window_size + 1))
    ymin = max(0, y_center - half_window_size)
    ymax = min(height, y_center + (half_window_size + 1))

    x_offset_min = half_window_size - (x_center - xmin)
    x_offset_max = half_window_size + (xmax - x_center)
    y_offset_min = half_window_size - (y_center - ymin)
    y_offset_max = half_window_size + (ymax - y_center)

    window[y_offset_min:y_offset_max,
           x_offset_min:x_offset_max] = image[ymin:ymax, xmin:xmax]

    return window


@jit(nopython=True, cache=True)
def solve(mu_F, Sigma_F, mu_B, Sigma_B, C, Sigma_C, alpha_init, maxIter = 50, minLike = 1e-6):
    """
    mu_F - Mean of foreground pixel
    Sigma_F - Covariance Mat of foreground pixel
    mu_B, Sigma_B - Mean and Covariance of background pixel
    C, Sigma_C - Current pixel, and its variance
    alpha_init - starting alpha value
    maxIter - Iterations to solve the value of the pixel
    minLike - min likelihood to reach to stop before maxIterations. 
    """

    # Initializing Matrices
    I = np.eye(3)
    fg_best = np.zeros(3)
    bg_best = np.zeros(3)
    a_best = 0.0
    maxLike = -np.inf
    
    invsgma2 = 1/Sigma_C**2
    
    for i in range(mu_F.shape[0]):
        # Mean of Foreground pixel can have multiple possible values, iterating for all.
        mu_Fi = mu_F[i]
        invSigma_Fi = np.linalg.inv(Sigma_F[i])

        for j in range(mu_B.shape[0]):
            # Similarly, multiple mean values be possible for background pixel.
            mu_Bj = mu_B[j]
            invSigma_Bj = np.linalg.inv(Sigma_B[j])

            alpha = alpha_init
            myiter = 1
            lastLike = -1.7977e+308

            # Solving Minimum likelihood through numerical methods
            while True:
                # Making Equations for AX = b, where we solve for X.abs
                # X here has 3 values of forground pixel (RGB) and 3 values for background
                A = np.zeros((6,6))
                A[:3,:3] = invSigma_Fi + I*alpha**2 * invsgma2
                A[:3,3:] = A[3:,:3] = I*alpha*(1-alpha) * invsgma2
                A[3:,3:] = invSigma_Bj+I*(1-alpha)**2 * invsgma2
                
                b = np.zeros((6,1))
                b[:3] = np.reshape(invSigma_Fi @ mu_Fi + C*(alpha) * invsgma2,(3,1))
                b[3:] = np.reshape(invSigma_Bj @ mu_Bj + C*(1-alpha) * invsgma2,(3,1))

                # Solving for X and storing values for Forground and Background Pixels 
                X = np.linalg.solve(A, b)
                F = np.maximum(0, np.minimum(1, X[0:3]))
                B = np.maximum(0, np.minimum(1, X[3:6]))
                
                # Solving for value of alpha once F and B are calculated
                alpha = np.maximum(0, np.minimum(1, ((np.atleast_2d(C).T-B).T @ (F-B))/np.sum((F-B)**2)))[0,0]
                
                # Calculating likelihood value for
                like_C = - np.sum((np.atleast_2d(C).T -alpha*F-(1-alpha)*B)**2) * invsgma2
                like_fg = (- ((F- np.atleast_2d(mu_Fi).T).T @ invSigma_Fi @ (F-np.atleast_2d(mu_Fi).T))/2)[0,0]
                like_bg = (- ((B- np.atleast_2d(mu_Bj).T).T @ invSigma_Bj @ (B-np.atleast_2d(mu_Bj).T))/2)[0,0]
                like = (like_C + like_fg + like_bg)

                if like > maxLike:
                    a_best = alpha
                    maxLike = like
                    fg_best = F.ravel()
                    bg_best = B.ravel()

                if myiter >= maxIter or abs(like-lastLike) <= minLike:
                    break

                lastLike = like
                myiter += 1
    return fg_best, bg_best, a_best 


#def Bayesian_Matte(img, trimap, N=105, sig=8, minNeighbours=10):
def Bayesian_Matte(img, trimap, N=None, sig=8, minNeighbours=10, return_mean_values=False, return_windows=False):
    if N is None:
        N = 105  # Set the default N value here.
    ...
    ...
    '''
    img - input image that the user will give to perform the foreground-background mapping
    trimap - the alpha mapping that is given with foreground and background determined.
    N - Window size, determines how many pixels will be sampled around the pixel to be solved, should be always odd.
    sig - wieghts of the neighbouring pixels. less means more centered.
    minNeighbours - Neigbour pixels available to solve, should be greater than 0, else inverse wont be calculated
    '''

    # We Convert the Images to float so that we are able to play with the pixel values
    img = np.array(img, dtype='float')
    trimap = np.array(image_trimap, dtype='float')

    # Here we normalise the Images to range from 0 and 1.
    img /= 255
    trimap /= 255

    # We get the dimensions
    h, w, c = img.shape

    # Preparing the gaussian weights for window
    gaussian_weights = matlab_style_gauss2d((N, N), sig)
    gaussian_weights /= np.max(gaussian_weights)

    fg_map = trimap == 1
    fg_map_resized = cv2.resize(fg_map.astype(np.float32), (w, h))
    fg_actual = np.zeros((h, w, c))
    fg_actual = img * fg_map_resized[..., np.newaxis]

    # We seperate the background specified in the trimap from the main image.
    bg_map = trimap == 0
    bg_map_resized = cv2.resize(bg_map.astype(np.float32), (w, h))
    bg_actual = np.zeros((h, w, c))
    bg_actual = img * bg_map_resized[..., np.newaxis]


    # Creating empty alpha channel to fill in by the program
    unknown_map = np.logical_or(fg_map, bg_map) == False
    a_channel = np.zeros(unknown_map.shape)
    a_channel[fg_map] = 1
    a_channel[unknown_map] = np.nan

    # Finding total number of unkown pixels to be calculated
    n_unknown = np.sum(unknown_map)

    # Making the datastructure for finding pixel values and saving id they have been solved yet or not.
    A, B = np.where(unknown_map == True)
    not_visited = np.vstack((A, B, np.zeros(A.shape))).T

    print("Solving Image with {} unsovled pixels... Please wait...".format(len))

    # running till all the pixels are solved.
    while(sum(not_visited[:, 2]) != n_unknown):
        last_n = sum(not_visited[:, 2])

        # iterating for all pixels
        for i in range(n_unknown):
            # checking if solved or not
            if not_visited[i, 2] == 1:
                continue

            # If not solved, we try to solve
            else:
                # We get the location of the unsolved pixel
                y, x = map(int, not_visited[i, :2])

                # Creating an window which states what pixels around it are solved(forground/background)
                a_window = get_window(
                    a_channel[:, :, np.newaxis], x, y, N)[:, :, 0]

                # Creating a window and weights of solved foreground window
                fg_window = get_window(fg_actual, x, y, N)
                fg_weights = np.reshape(a_window**2 * gaussian_weights, -1)
                values_to_keep = np.nan_to_num(fg_weights) > 0
                fg_pixels = np.reshape(fg_window, (-1, 3))[values_to_keep, :]
                fg_weights = fg_weights[values_to_keep]

                # Creating a window and weights of solved background window
                bg_window = get_window(bg_actual, x, y, N)
                bg_weights = np.reshape((1-a_window)**2 * gaussian_weights, -1)
                values_to_keep = np.nan_to_num(bg_weights) > 0
                bg_pixels = np.reshape(bg_window, (-1, 3))[values_to_keep, :]
                bg_weights = bg_weights[values_to_keep]

                # We come back to this pixel later if it doesnt has enough solved pixels around it.
                if len(bg_weights) < minNeighbours or len(fg_weights) < minNeighbours:
                    continue

                # If enough pixels, we cluster these pixels to get clustered colour centers and their covariance    matrices
                mean_fg, cov_fg = clustFunc(fg_pixels, fg_weights)
                mean_bg, cov_bg = clustFunc(bg_pixels, bg_weights)
                alpha_init = np.nanmean(a_window.ravel())

                # We try to solve our 3 equation 7 variable problem with minimum likelihood estimation
                fg_pred, bg_pred, alpha_pred = solve(
                    mean_fg, cov_fg, mean_bg, cov_bg, img[y, x], 0.7, alpha_init)

                # storing the predicted values in appropriate windows for use for later pixels.
                fg_actual[y, x] = fg_pred.ravel()
                bg_actual[y, x] = bg_pred.ravel()
                a_channel[y, x] = alpha_pred
                not_visited[i, 2] = 1
                if(np.sum(not_visited[:, 2]) % 1000 == 0):
                    print("Solved {} out of {}.".format(
                        np.sum(not_visited[:, 2]), len(not_visited)))

                if return_mean_values:
                   return mean_fg.ravel(), mean_bg.ravel()

        if sum(not_visited[:, 2]) == last_n:
            # ChangingWindow Size
            # Preparing the gaussian weights for window
            N += 2
            # sig += 1
            gaussian_weights = matlab_style_gauss2d((N, N), sig)
            gaussian_weights /= np.max(gaussian_weights)
            print(N)
   
    if return_windows:
     return bg_window, fg_window, a_channel, n_unknown,N
    else:
     return a_channel, n_unknown,N

def image_composite(foreground, background, alpha):
        
    alpha = np.array(alpha)

    # Normalize the alpha matte to have values between 0 and 1
    alpha = alpha.astype(np.float64) / 255.0

    # Create the composite image
    composite = alpha * foreground + (1 - alpha) * background

    # Display the composite image
    plt.imshow(composite, cmap='gray')
    plt.show()



image = np.array(Image.open('C:/Users/Chengyu/Desktop/Bayesian-Matting-main/Bayesian_Matting_Python/input_training_lowres/GT02.png'))

image_trimap = np.array(ImageOps.grayscale(Image.open('C:/Users/Chengyu/Desktop/Bayesian-Matting-main/Bayesian_Matting_Python/trimap_training_lowres/Trimap2/GT02.png')))

alpha, pixel_count ,N = Bayesian_Matte(image, image_trimap )


image_alpha = np.array(ImageOps.grayscale(Image.open('C:/Users/Chengyu/Desktop/Bayesian-Matting-main/Bayesian_Matting_Python/gt_training_lowres/GT02.png')))

alpha = alpha*255
plt.imsave("alpha.png", alpha, cmap='gray')
# plt.imshow(alpha, cmap='gray')
# plt.show()

print("Absolute Loss with ground truth - ",
      np.sum(np.abs(alpha - image_alpha))/(alpha.shape[0]*alpha.shape[1]))



def load_image(image_path):
    return cv2.imread(image_path)

def load_trimap(trimap_path):
    return cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)

def check_trimap_values(trimap):
    min_value, max_value = np.min(trimap), np.max(trimap)
    return min_value >= 0 and max_value <= 255


class TestBayesianMatte(unittest.TestCase):
    #def setUp(self):
    #    print(f'\nRunning test: {self._testMethodName}')

    def test_image_and_alpha_size(self):
        print('\n I am testing if the image size and alpha size are the same \n')
        image = np.array(Image.open('C:/Users/Chengyu/Desktop/Bayesian-Matting-main/Bayesian_Matting_Python/input_training_lowres/GT02.png'))
        image_trimap = np.array(ImageOps.grayscale(Image.open('C:/Users/Chengyu/Desktop/Bayesian-Matting-main/Bayesian_Matting_Python/trimap_training_lowres/Trimap2/GT02.png'))) 
        alpha, pixel_count, N = Bayesian_Matte(image, image_trimap)
        self.assertEqual(image.shape[0:2], alpha.shape)
   
   
    def test_image_and_trimap_size(self):
        print('\n I am testing if the image size and trimap size are the same \n')
        image = np.array(Image.open('C:/Users/Chengyu/Desktop/Bayesian-Matting-main/Bayesian_Matting_Python/input_training_lowres/GT02.png'))
        image_trimap = np.array(ImageOps.grayscale(Image.open('C:/Users/Chengyu/Desktop/Bayesian-Matting-main/Bayesian_Matting_Python/trimap_training_lowres/Trimap2/GT02.png')))
    #  ignore the image with 3 color channels because trimap is a grayscale image with a single color channel  
        self.assertEqual(image.shape[:2], image_trimap.shape)

    def test_image_loading(self):
        print('\n I am testing the image loading \n')
        input_image_path = 'C:/Users/Chengyu/Desktop/Bayesian-Matting-main/Bayesian-Matting-main/input_training_lowres/GT04.PNG'
        image = load_image(input_image_path)
        self.assertIsNotNone(image, "Image not loaded properly. Please check the file path and its format.")

    def test_odd_N_value(self):
        print('\n I am testing if the window size is odd \n')
        image = np.array(Image.open('C:/Users/Chengyu/Desktop/Bayesian-Matting-main/Bayesian_Matting_Python/input_training_lowres/GT02.png'))
        image_trimap = np.array(ImageOps.grayscale(Image.open('C:/Users/Chengyu/Desktop/Bayesian-Matting-main/Bayesian_Matting_Python/trimap_training_lowres/Trimap2/GT02.png')))
        alpha, pixel_count, N = Bayesian_Matte(image, image_trimap)   
        self.assertTrue(N % 2 == 1)

        
    def test_trimap_loading(self):
        print('\n I am testing the trimap loading \n')
        trimap_path = 'C:/Users/Chengyu/Desktop/Bayesian-Matting-main/Bayesian-Matting-main/trimap_training_lowres/Trimap1/GT04.png'
        trimap = load_trimap(trimap_path)
        self.assertIsNotNone(trimap, "Trimap not loaded properly. Please check the file path and its format.")
        
    def test_trimap_values(self):
        print('\n I am testing the trimap values \n')
        trimap_path = 'C:/Users/Chengyu/Desktop/Bayesian-Matting-main/Bayesian-Matting-main/trimap_training_lowres/Trimap1/GT04.png'
        trimap = load_trimap(trimap_path)
        self.assertTrue(check_trimap_values(trimap), "Trimap values are not appropriate. Values should be 0, 255, or between 0 and 255.")




if __name__ == '__main__':
    unittest.main()
