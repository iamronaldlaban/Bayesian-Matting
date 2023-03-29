import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import os
from numba import jit
from orchard_bouman_clust import clustFunc
 

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


def Bayesian_Matte(img, trimap, N=105, sig=8, minNeighbours=10):
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

    # We seperate the foreground specified in the trimap from the main image.
    fg_map = trimap == 1
    fg_actual = np.zeros((h, w, c))
    fg_actual = img * np.reshape(fg_map, (h, w, 1))

    # We seperate the background specified in the trimap from the main image.
    bg_map = trimap == 0
    bg_actual = np.zeros((h, w, c))
    bg_actual = img * np.reshape(bg_map, (h, w, 1))

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

        if sum(not_visited[:, 2]) == last_n:
            # ChangingWindow Size
            # Preparing the gaussian weights for window
            N += 2
            # sig += 1
            gaussian_weights = matlab_style_gauss2d((N, N), sig)
            gaussian_weights /= np.max(gaussian_weights)
            print(N)

    return a_channel, n_unknown

def image_composite(foreground, background, alpha):
        
    alpha = np.array(alpha)

    # Normalize the alpha matte to have values between 0 and 1
    alpha = alpha.astype(np.float64) / 255.0

    # Create the composite image
    composite = alpha * foreground + (1 - alpha) * background

    # Display the composite image
    plt.imshow(composite, cmap='gray')
    plt.show()



image = np.array(Image.open('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/input_training_lowres/GT02.png'))
image_trimap = np.array(ImageOps.grayscale(Image.open('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/trimap_training_lowres/Trimap2/GT02.png')))

alpha, pixel_count = Bayesian_Matte(image, image_trimap)


image_alpha = np.array(ImageOps.grayscale(Image.open('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/gt_training_lowres/GT02.png')))

alpha = alpha*255
plt.imsave("alpha.png", alpha, cmap='gray')
# plt.imshow(alpha, cmap='gray')
# plt.show()

print("Absolute Loss with ground truth - ",
      np.sum(np.abs(alpha - image_alpha))/(alpha.shape[0]*alpha.shape[1]))


# Load foreground, background, and alpha images
foreground = cv2.imread('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/input_training_lowres/GT02.png')
background = cv2.imread('Trinity.png')
alpha = cv2.imread('alpha.png')

# Fix 1: Reshape alpha to have the same shape as foreground and background
alpha = np.broadcast_to(alpha[..., np.newaxis], foreground.shape)

# Fix 2: Convert foreground and background to grayscale
foreground_gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# Blend the images using the alpha mask
composite_gray = alpha * foreground_gray + (1 - alpha) * background_gray

# Convert the composite image back to color
composite = cv2.cvtColor(composite_gray, cv2.COLOR_GRAY2BGR)

# Save the composite image
cv2.imwrite('composite.jpg', composite)



