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
    half_height, half_width = [(ss-1)/2 for ss in shape]
    x_coords, y_coords = np.meshgrid(np.arange(-half_width, half_width+1), np.arange(-half_height, half_height+1))
    kernel = np.exp(-(x_coords**2 + y_coords**2)/(2*sigma**2))
    kernel[kernel < np.finfo(kernel.dtype).eps*kernel.max()] = 0
    kernel /= np.sum(kernel)
    return kernel



@jit(nopython=True, cache=True)
def get_window(image, x_center, y_center, window_size):
    """
    Extracts a window of a specified size centered at a given pixel location (x_center, y_center) from an image.
    If the window goes beyond the boundary of the image, it is padded with zeros.

    Args:
        image (numpy.ndarray): A three-dimensional NumPy array representing an image, with dimensions 
                               (height, width, channels).
        x_center (int): An integer specifying the x-coordinate of the center pixel of the window.
        y_center (int): An integer specifying the y-coordinate of the center pixel of the window.
        window_size (int): An integer specifying the size of the window to extract.

    Returns:
        numpy.ndarray: A three-dimensional NumPy array representing the extracted window, with dimensions 
                       (window_size, window_size, channels). If the window goes beyond the boundary of 
                       the image, it is padded with zeros.
    """
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
def solve(mu_F, Sigma_F, mu_B, Sigma_B, C, Sigma_C, alpha_init, max_iterations = 50, min_likelihood = 1e-6):
    """
    Args:

    mu_F: Mean of foreground pixel
    Sigma_F: Covariance matrix of foreground pixel
    mu_B: Mean of background pixel
    Sigma_B: Covariance matrix of background pixel
    C: Current pixel value
    Sigma_C: Variance of current pixel value
    alpha_init: Starting alpha value
    maxIter: Maximum number of iterations to solve the value of the pixel (default value: 50)
    minLike: Minimum likelihood to reach to stop before maxIterations (default value: 1e-6)
    Returns:

    foreground_best: Foreground pixel values
    bg_best: Background pixel values
    background_best: Alpha value
    """

    # Initializing Matrices
    identity_matrix = np.eye(3)
    foreground_best = np.zeros(3)
    background_best = np.zeros(3)
    alpha_best = 0.0
    max_likelihood = -np.inf

    inverse_sigma_squared = 1 / Sigma_C ** 2

    for i in range(mu_F.shape[0]):
        foreground_mean_i = mu_F[i]
        inverse_sigma_Fi = np.linalg.inv(Sigma_F[i])

        for j in range(mu_B.shape[0]):
            background_mean_j = mu_B[j]
            inverse_sigma_Bj = np.linalg.inv(Sigma_B[j])

            alpha = alpha_init
            iteration = 1
            last_likelihood = -1.7977e+308

            while True:
                # Solving AX = b where X is composed of foreground and background pixels
                A = np.zeros((6,6))
                A[:3,:3] = inverse_sigma_Fi + identity_matrix * alpha**2 * inverse_sigma_squared
                A[:3,3:] = A[3:,:3] = identity_matrix * alpha*(1-alpha) * inverse_sigma_squared
                A[3:,3:] = inverse_sigma_Bj+identity_matrix*(1-alpha)**2 * inverse_sigma_squared

                b = np.zeros((6,1))
                b[:3] = np.reshape(inverse_sigma_Fi @ foreground_mean_i + C*(alpha) * inverse_sigma_squared,(3,1))
                b[3:] = np.reshape(inverse_sigma_Bj @ background_mean_j + C*(1-alpha) * inverse_sigma_squared,(3,1))

                # Solving for foreground and background pixels
                X = np.linalg.solve(A, b)
                foreground_pixels = np.maximum(0, np.minimum(1, X[0:3]))
                background_pixels = np.maximum(0, np.minimum(1, X[3:6]))

                # Solving for alpha
                alpha = np.maximum(0, np.minimum(1, ((np.atleast_2d(C).T-background_pixels).T @ (foreground_pixels-background_pixels))/np.sum((foreground_pixels-background_pixels)**2)))[0,0]

                # Calculating likelihood
                likelihood_C = - np.sum((np.atleast_2d(C).T -alpha*foreground_pixels-(1-alpha)*background_pixels)**2) * inverse_sigma_squared
                likelihood_fg = (- ((foreground_pixels- np.atleast_2d(foreground_mean_i).T).T @ inverse_sigma_Fi @ (foreground_pixels-np.atleast_2d(foreground_mean_i).T))/2)[0,0]
                likelihood_bg = (- ((background_pixels- np.atleast_2d(background_mean_j).T).T @ inverse_sigma_Bj @ (background_pixels-np.atleast_2d(background_mean_j).T))/2)[0,0]
                likelihood = (likelihood_C + likelihood_fg + likelihood_bg)

                # Updating variables with the best values
                if likelihood > max_likelihood:
                    alpha_best = alpha
                    max_likelihood = likelihood
                    foreground_best = foreground_pixels.ravel()
                    background_best = background_pixels.ravel()

                if iteration >= max_iterations or abs(likelihood-last_likelihood) <= min_likelihood:
                    break

                last_likelihood = likelihood
                iteration += 1

    return foreground_best, background_best, alpha_best 


def compositing(foreground, alpha_channel, background_image): 
    # Get the dimensions of the alpha channel
    height, width = alpha_channel.shape[:2]

    # Resize the background image to match the dimensions of the alpha channel
    background_image = cv2.resize(background_image, (width, height))

    # Convert the images to float
    foreground = foreground / 255.0
    alpha_channel = alpha_channel / 255.0
    background_image = background_image / 255.0

    # Reshape the alpha channel to match the dimensions of the foreground image
    alpha_channel = alpha_channel.reshape((height, width, 1))
    alpha_channel = np.broadcast_to(alpha_channel, (height, width, 3))

    # Compose the final image by blending the foreground and background based on the alpha channel
    composite_image = foreground * (alpha_channel) + background_image * (1 - alpha_channel)

    return composite_image
