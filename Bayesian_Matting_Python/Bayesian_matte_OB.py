import numpy as np
from matting_functions import matlab_style_gauss2d
from matting_functions import get_window
from orchard_bouman_clust import clustFunc 
from matting_functions import solve
from cluster_KNN import knn_cluster

def Bayesian_Matte(image, trimap, N=105, sig=8, minNeighbours=10):
    '''
    img - input image that the user will give to perform the foreground-background mapping
    trimap - the alpha mapping that is given with foreground and background determined.
    N - Window size, determines how many pixels will be sampled around the pixel to be solved, should be always odd.
    sig - wieghts of the neighbouring pixels. less means more centered.
    minNeighbours - Neigbour pixels available to solve, should be greater than 0, else inverse wont be calculated
    '''

    # We Convert the Images to float so that we are able to play with the pixel values
    image = np.array(image, dtype='float')
    trimap = np.array(trimap, dtype='float')

    # Here we normalise the Images to range from 0 and 1.
    image /= 255
    trimap /= 255

    # We get the dimensions
    h, w, c = image.shape

    # Preparing the gaussian weights for window
    gaussian_weights = matlab_style_gauss2d((N, N), sig)
    gaussian_weights /= np.max(gaussian_weights)

    # We seperate the foreground specified in the trimap from the main image.
    foreground_map = trimap == 1
    foreground_actual = np.zeros((h, w, c))
    foreground_actual = image * np.reshape(foreground_map, (h, w, 1))

    # We seperate the background specified in the trimap from the main image.
    background_map = trimap == 0
    background_actual = np.zeros((h, w, c))
    background_actual = image * np.reshape(background_map, (h, w, 1))

    # Creating empty alpha channel to fill in by the program
    unknown_map = np.logical_or(foreground_map, background_map) == False
    alpha_channel = np.zeros(unknown_map.shape)
    alpha_channel[foreground_map] = 1
    alpha_channel[unknown_map] = np.nan

    # Finding total number of unkown pixels to be calculated
    n_unknown = np.sum(unknown_map)

    # Making the datastructure for finding pixel values and saving id they have been solved yet or not.
    A, B = np.where(unknown_map == True)
    points_not_visited = np.vstack((A, B, np.zeros(A.shape))).T

    print("Solving Image with {} unsolved pixels... Please wait...".format(n_unknown))

    # running till all the pixels are solved.
    while(sum(points_not_visited[:, 2]) != n_unknown):
        last_value = sum(points_not_visited[:, 2])

        # iterating for all pixels
        for i in range(n_unknown):
            # checking if solved or not
            if points_not_visited[i, 2] == 1:
                continue

            # If not solved, we try to solve
            else:
                # We get the location of the unsolved pixel
                y, x = map(int, points_not_visited[i, :2])

                # Creating an window which states what pixels around it are solved(forground/background)
                a_window = get_window(
                    alpha_channel[:, :, np.newaxis], x, y, N)[:, :, 0]

                # Creating a window and weights of solved foreground window
                foreground_window = get_window(foreground_actual, x, y, N)
                foreground_weights = np.reshape(a_window**2 * gaussian_weights, -1)
                values_to_keep = np.nan_to_num(foreground_weights) > 0
                foreground_pixels = np.reshape(foreground_window, (-1, 3))[values_to_keep, :]
                foreground_weights = foreground_weights[values_to_keep]

                # Creating a window and weights of solved background window
                background_window = get_window(background_actual, x, y, N)
                background_weights = np.reshape((1-a_window)**2 * gaussian_weights, -1)
                values_to_keep = np.nan_to_num(background_weights) > 0
                background_pixels = np.reshape(background_window, (-1, 3))[values_to_keep, :]
                background_weights = background_weights[values_to_keep]

                # We come back to this pixel later if it doesnt has enough solved pixels around it.
                if len(background_weights) < minNeighbours or len(foreground_weights) < minNeighbours:
                    continue

                # If enough pixels, we cluster these pixels to get clustered colour centers and their covariance    matrices
                mean_foreground, cov_foreground = clustFunc(foreground_pixels, foreground_weights)
                mean_background, cov_background = clustFunc(background_pixels, background_weights)
                alpha_init = np.nanmean(a_window.ravel())

                # We try to solve our 3 equation 7 variable problem with minimum likelihood estimation
                foreground_predicted, background_predicted, alpha_predicted = solve(
                    mean_foreground, cov_foreground, mean_background, cov_background, image[y, x], 0.7, alpha_init)

                # storing the predicted values in appropriate windows for use for later pixels.
                foreground_actual[y, x] = foreground_predicted.ravel()
                background_actual[y, x] = background_predicted.ravel()
                alpha_channel[y, x] = alpha_predicted
                points_not_visited[i, 2] = 1
                if(np.sum(points_not_visited[:, 2]) % 1000 == 0):
                    print("Solved {} out of {}.".format(
                        np.sum(points_not_visited[:, 2]), len(points_not_visited)))

        if sum(points_not_visited[:, 2]) == last_value:
            # ChangingWindow Size
            # Preparing the gaussian weights for window
            N += 2
            sig += 1
            gaussian_weights = matlab_style_gauss2d((N, N), sig)
            gaussian_weights /= np.max(gaussian_weights)
            print(N)

    return alpha_channel