import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from skimage import io

# Load the input image and trimap
img_path = 'C:/Users/Chengyu/Desktop/Bayesian-Matting-main/Bayesian-Matting-main/input_training_lowres/GT04.PNG'
trimap_path = 'C:/Users/Chengyu/Desktop/Bayesian-Matting-main/Bayesian-Matting-main/trimap_training_lowres/Trimap1/GT04.png'
img = np.array(img_path, dtype='float')
trimap = np.array(trimap_path, dtype='float')

# Convert the image and trimap to double precision and normalize
img = img.astype(np.float64) / 255.0
trimap = trimap.astype(np.float64) / 255.0

# Get the size of the image
m, n, c = img.shape

# Calculate the foreground, background, and unknown pixels
fg = trimap > 0.99
bg = trimap < 0.01
unk = ~(fg | bg)

# Create the Laplacian matrix
A = sp.lil_matrix((m * n, m * n))
A.setdiag(4)

for i in range(m * n):
    if i >= n:  # add left neighbor
        A[i, i - n] = -1
    if i % n != (n - 1):  # add right neighbor
        A[i, i + 1] = -1
    if i < (m - 1) * n:  # add top neighbor
        A[i, i + n] = -1
    if i % n != 0:  # add bottom neighbor
        A[i, i - 1] = -1

A = A.tocsc()

# Calculate the alpha matte
b = np.zeros(m * n)  # initialize b vector
b[fg.ravel()] = 1  # set foreground pixels to 1
b[bg.ravel()] = 0  # set background pixels to 0
alpha = np.zeros((m, n))
alpha[unk] = spsolve(A[unk.ravel(), :][:, unk.ravel()], b[unk.ravel()])  # solve linear system for unknown pixels

# Save the output alpha matte
plt.imshow(alpha)
plt.show()
