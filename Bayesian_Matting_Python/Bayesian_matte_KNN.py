import numpy as np
from matting_functions import matlab_style_gauss2d
from matting_functions import get_window
from orchard_bouman_clust import clustFunc 
from matting_functions import solve
from cluster_KNN import knn_cluster


def Bayesian_Matte_KNN(img, trimap, N=25, sig=8, minNeighbours=10):
    '''
    img - input image that the user will give to perform the foreground-background mapping
    trimap - the alpha mapping that is given with foreground and background determined.
    N - Window size, determines how many pixels will be sampled around the pixel to be solved, should be always odd.
    sig - wieghts of the neighbouring pixels. less means more centered.
    minNeighbours - Neigbour pixels available to solve, should be greater than 0, else inverse wont be calculated
    '''

    # We Convert the Images to float so that we are able to play with the pixel values
    img = np.array(img, dtype='float')
    trimap = np.array(trimap, dtype='float')

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
                mean_fg, cov_fg = knn_cluster(fg_pixels, fg_weights)
                mean_bg, cov_bg = knn_cluster(bg_pixels, bg_weights)
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
            sig += 1
            gaussian_weights = matlab_style_gauss2d((N, N), sig)
            gaussian_weights /= np.max(gaussian_weights)
            print(N)

    return a_channel