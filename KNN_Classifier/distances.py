import numpy as np

def euclidean_distance(X, Y):
    m = X.shape[0]
    n = Y.shape[0]
    X_dots = (X * X).sum(axis=1).reshape((m, 1)) * np.ones(shape=(1, n))
    Y_dots = (Y * Y).sum(axis=1) * np.ones(shape=(m, 1))
    return np.sqrt(X_dots + Y_dots - 2 * X.dot(Y.T))


def cosine_distance(X, Y):
    dotted = X.dot(Y.T)
    matrix_norms = np.linalg.norm(X, axis=1)
    vector_norm = np.linalg.norm(Y, axis=1)
    matrix_vector_norms = np.outer(matrix_norms, vector_norm)
    neighbors = 1.0 - np.divide(dotted, matrix_vector_norms)
    return neighbors
