import numpy as np

class Node:
    def __init__(self, data_matrix, weights):
        total_weight = np.sum(weights)
        self.weights = weights
        self.data = data_matrix
        self.left = None
        self.right = None
        self.mean = np.einsum('ij,i->j', self.data, weights) / total_weight
        diff = self.data - np.tile(self.mean, [(self.data.shape)[0], 1])
        scaled_diff = np.einsum('ij,i->ij', diff, np.sqrt(weights))
        self.cov = (scaled_diff.T @ scaled_diff) / total_weight + 1e-5 * np.eye(3)
        self.num_data = self.data.shape[0]
        eigvals, eigvecs = np.linalg.eig(self.cov)
        self.largest_eigval = np.max(np.abs(eigvals))
        self.largest_eigvec = eigvecs[np.argmax(np.abs(eigvals))]


def clustFunc(data_matrix, weights, min_var=0.05):
    means, covariances = [], []
    nodes = []
    nodes.append(Node(data_matrix, weights))

    while max(nodes, key=lambda x: x.largest_eigval).largest_eigval > min_var:
        nodes = split(nodes)

    for i, node in enumerate(nodes):
        means.append(node.mean)
        covariances.append(node.cov)

    return np.array(means), np.array(covariances)


def split(nodes):
    max_eigval_index = max(enumerate(nodes), key=lambda x: x[1].largest_eigval)[0]
    C_i = nodes[max_eigval_index]
    idx = C_i.data @ C_i.largest_eigvec <= np.dot(C_i.mean, C_i.largest_eigvec)
    C_a = Node(C_i.data[idx], C_i.weights[idx])
    C_b = Node(C_i.data[np.logical_not(idx)], C_i.weights[np.logical_not(idx)])
    nodes.pop(max_eigval_index)
    nodes.append(C_a)
    nodes.append(C_b)
    return nodes
