import numpy as np
import matplotlib.pyplot as plt
import cv2
from numba import jit


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

def compositing(img, alpha, background): 

    H = alpha.shape[0]
    W = alpha.shape[1]

    # Resizing the background image to the size of the alpha channel
    background = cv2.resize(background, (W, H))

    # Converting the images to float
    img = img / 255
    alpha = alpha / 255
    background = background / 255

    # Reshaping the alpha channel to the size of the foreground image
    alpha = alpha.reshape((H, W, 1))
    alpha = np.broadcast_to(alpha, (H, W, 3))

    # Compositing the foreground and background images
    comp = img * (alpha) + background * (1 - alpha)

    return comp