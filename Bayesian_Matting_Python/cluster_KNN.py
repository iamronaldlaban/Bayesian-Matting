import numpy as np
from sklearn.neighbors import NearestNeighbors

class Node(object):

    def __init__(self, matrix, w):
        W = np.sum(w)
        self.w = w
        self.X = matrix
        self.left = None
        self.right = None
        self.mu = np.einsum('ij,i->j', self.X, w)/W
        diff = self.X - np.tile(self.mu, [(self.X.shape)[0], 1])
        t = np.einsum('ij,i->ij', diff, np.sqrt(w))
        self.cov = (t.T @ t)/W + 1e-5*np.eye(3)
        self.N = self.X.shape[0]
        V, D = np.linalg.eig(self.cov)
        self.lmbda = np.max(np.abs(V))
        self.e = D[np.argmax(np.abs(V))]

def knn_cluster(S, w, k=5, minVar=0.05):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(S)
    distances, indices = nbrs.kneighbors(S)
    nodes = []
    nodes.append(Node(S, w))
    while max(nodes, key=lambda x: x.lmbda).lmbda > minVar:
        nodes = split(nodes)
    mu, sigma = [], []
    for node in nodes:
        mu.append(node.mu)
        sigma.append(node.cov)
    return np.array(mu), np.array(sigma)

def split(nodes):
    idx_max = max(enumerate(nodes), key=lambda x: x[1].lmbda)[0]
    C_i = nodes[idx_max]
    idx = C_i.X @ C_i.e <= np.dot(C_i.mu, C_i.e)
    if np.count_nonzero(idx) == 0 or np.count_nonzero(~idx) == 0:
        return nodes
    C_a = Node(C_i.X[idx], C_i.w[idx])
    C_b = Node(C_i.X[~idx], C_i.w[~idx])
    nodes.pop(idx_max)
    nodes.append(C_a)
    nodes.append(C_b)
    return nodes
