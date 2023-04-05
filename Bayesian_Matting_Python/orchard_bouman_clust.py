import numpy as np

import numpy as np


def calc_total_weight(weights):
    return np.sum(weights)


def calc_mean(data, weights, total_weight):
    return np.einsum('ij,i->j', data, weights) / total_weight


def calc_cov(data, weights, total_weight):
    mean = calc_mean(data, weights, total_weight)
    diff = data - np.tile(mean, [(data.shape)[0], 1])
    scaled_diff = np.einsum('ij,i->ij', diff, np.sqrt(weights))
    return (scaled_diff.T @ scaled_diff) / total_weight + 1e-5 * np.eye(3)


def calc_largest_eig(cov):
    eigvals, eigvecs = np.linalg.eig(cov)
    largest_eigval = np.max(np.abs(eigvals))
    largest_eigvec = eigvecs[np.argmax(np.abs(eigvals))]
    return largest_eigval, largest_eigvec


def Node(data_matrix, weights):
    total_weight = calc_total_weight(weights)
    mean = calc_mean(data_matrix, weights, total_weight)
    cov = calc_cov(data_matrix, weights, total_weight)
    largest_eigval, largest_eigvec = calc_largest_eig(cov)

    return {
        "weights": weights,
        "data": data_matrix,
        "mean": mean,
        "cov": cov,
        "num_data": data_matrix.shape[0],
        "largest_eigval": largest_eigval,
        "largest_eigvec": largest_eigvec,
        "left": None,
        "right": None
    }


def clustFunc(data_matrix, weights, min_var=0.05):
    means, covariances = [], []
    nodes = []
    nodes.append(Node(data_matrix, weights))

    while max(nodes, key=lambda x: x["largest_eigval"])["largest_eigval"] > min_var:
        nodes = split(nodes)

    for i, node in enumerate(nodes):
        means.append(node["mean"])
        covariances.append(node["cov"])

    return np.array(means), np.array(covariances)


def split(nodes):
    max_eigval_index = max(enumerate(nodes), key=lambda x: x[1]["largest_eigval"])[0]
    C_i = nodes[max_eigval_index]
    idx = C_i["data"] @ C_i["largest_eigvec"] <= np.dot(C_i["mean"], C_i["largest_eigvec"])
    C_a = Node(C_i["data"][idx], C_i["weights"][idx])
    C_b = Node(C_i["data"][np.logical_not(idx)], C_i["weights"][np.logical_not(idx)])
    nodes.pop(max_eigval_index)
    nodes.append(C_a)
    nodes.append(C_b)
    return nodes
