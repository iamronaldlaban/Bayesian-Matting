import numpy as np
from sklearn.neighbors import NearestNeighbors

# class Node(object):

#     def __init__(self, matrix, w):
#         W = np.sum(w)
#         self.w = w
#         self.X = matrix
#         self.left = None
#         self.right = None
#         self.mu = np.einsum('ij,i->j', self.X, w)/W
#         diff = self.X - np.tile(self.mu, [(self.X.shape)[0], 1])
#         t = np.einsum('ij,i->ij', diff, np.sqrt(w))
#         self.cov = (t.T @ t)/W + 1e-5*np.eye(3)
#         self.N = self.X.shape[0]
#         V, D = np.linalg.eig(self.cov)
#         self.lmbda = np.max(np.abs(V))
#         self.e = D[np.argmax(np.abs(V))]

# def knn_cluster(S, w, k=5, minVar=0.05):
#     nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(S)
#     distances, indices = nbrs.kneighbors(S)
#     nodes = []
#     nodes.append(Node(S, w))
#     while max(nodes, key=lambda x: x.lmbda).lmbda > minVar:
#         nodes = split(nodes)
#     mu, sigma = [], []
#     for node in nodes:
#         mu.append(node.mu)
#         sigma.append(node.cov)
#     return np.array(mu), np.array(sigma)

# def split(nodes):
#     idx_max = max(enumerate(nodes), key=lambda x: x[1].lmbda)[0]
#     C_i = nodes[idx_max]
#     idx = C_i.X @ C_i.e <= np.dot(C_i.mu, C_i.e)
#     if np.count_nonzero(idx) == 0 or np.count_nonzero(~idx) == 0:
#         return nodes
#     C_a = Node(C_i.X[idx], C_i.w[idx])
#     C_b = Node(C_i.X[~idx], C_i.w[~idx])
#     nodes.pop(idx_max)
#     nodes.append(C_a)
#     nodes.append(C_b)
#     return nodes

# import numpy as np

# def knn_cluster(X, k, minVar=0.05):
#     n, d = X.shape
#     clusters = []
#     for i in range(n):
#         dists = np.sum((X - X[i])**2, axis=1)
#         nearest_centroid = None
#         nearest_dist = float('inf')
#         for cluster in clusters:
#             centroid = cluster['mu']
#             dist = np.sum((X[i] - centroid)**2)
#             if dist < nearest_dist:
#                 nearest_dist = dist
#                 nearest_centroid = centroid
#         if nearest_centroid is None or nearest_dist > minVar:
#             clusters.append({
#                 'X': [X[i]],
#                 'w': [1],
#                 'mu': X[i],
#                 'cov': np.eye(d),
#                 'N': 1,
#             })
#         else:
#             cluster = next(c for c in clusters if np.all(c['mu'] == nearest_centroid))
#             cluster['X'].append(X[i])
#             cluster['w'].append(1)
#             W = np.sum(cluster['w'])
#             cluster['mu'] = np.einsum('ij,i->j', cluster['X'], cluster['w']) / W
#             diff = np.array(cluster['X']) - np.tile(cluster['mu'], [(len(cluster['X'])), 1])
#             t = np.einsum('ij,i->ij', diff, np.sqrt(cluster['w']))
#             cluster['cov'] = (t.T @ t) / W + 1e-5 * np.eye(d)
#             cluster['N'] += 1

#     mu = np.array([c['mu'] for c in clusters])
#     sigma = np.array([c['cov'] for c in clusters])
#     return mu, sigma
def reshapeWeights(matrix, mask):
  matrix = matrix[:, :, 0]
  mask = mask[:, :, 0]
  mask = mask.reshape(mask.size)
  matrix = np.delete(matrix, np.where(np.isnan(mask)))
  weight_vector = matrix
  return weight_vector


def reshapePixels(matrix, mask):
  mask = mask.reshape(mask.size)
  matrix = np.delete(matrix, np.where(np.isnan(mask)))
  matrix = matrix.reshape(np.uint32(matrix.size / 3), 3)
  new_matrix = matrix.T
  return new_matrix



def calcMean(pixel, weight):
  W = np.sum(weight)
  # get the values for each of the three RGB values
  pixel_r = pixel[0, :]
  pixel_g = pixel[1, :]
  pixel_b = pixel[2, :]
  # calculate the mean for each channel
  mean_r = (np.sum(pixel_r * weight)) / W
  mean_g = (np.sum(pixel_g * weight)) / W
  mean_b = (np.sum(pixel_b * weight)) / W
  # mean_rgb = [[mean_r], [mean_g], [mean_b]]
  mean_rgb = np.stack((mean_r, mean_g, mean_b), axis=0).reshape([3, 1])
  # combine the mean
  # mean_rgb = np.around(mean_rgb, decimals=5)
  return mean_rgb


def calcCovariance(pixel, weight, mean_rgb):
  W = np.sum(weight)
  # get the values for each of the three RGB values
  pixel_r = pixel[0, :]
  pixel_g = pixel[1, :]
  pixel_b = pixel[2, :]
  # obtain the mean values for each channel
  mean_r = mean_rgb[0]
  mean_g = mean_rgb[1]
  mean_b = mean_rgb[2]
  # middle matrices
  middle_r = pixel_r - mean_r
  middle_g = pixel_g - mean_g
  middle_b = pixel_b - mean_b
  # combine those middles matrices
  middle = np.stack((middle_r, middle_g, middle_b), axis=0)
  # calculate the covariance
  covariance = ((np.multiply(weight, middle)) @ (np.transpose(middle))) / W
  # covariance = np.around(covariance, decimals=5)
  return covariance
