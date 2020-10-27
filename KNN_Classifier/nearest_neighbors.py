import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
import distances


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.metric = metric
        if self.metric not in ['euclidean', 'cosine']:
            raise TypeError(self.metric + ' not euclidean or cosine')
        self.k = k
        self.weights = weights
        if strategy in ['brute', 'kd_tree', 'ball_tree']:
            self.strategy = sklearn.neighbors.NearestNeighbors(
                algorithm=strategy, metric=self.metric)
        elif strategy == 'my_own':
            self.strategy = strategy
        else:
            raise TypeError(strategy + ' unacceptable')
        self.test_block_size = test_block_size

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        if self.strategy != 'my_own':
            self.strategy.fit(X, y)

    def find_kneighbors(self, X, return_distance):
        iter_num = np.size(X, 0) // self.test_block_size
        if np.size(X, 0) % self.test_block_size:
            iter_num += 1
        ind_list = list()
        if return_distance:
            dist_list = list()
        for i in range(iter_num):
            ind_beg = self.test_block_size * i
            ind_end = self.test_block_size * (i + 1)
            if self.strategy == 'my_own':
                if self.metric == 'euclidean':
                    distance_func = distances.euclidean_distance
                elif self.metric == 'cosine':
                    distance_func = distances.cosine_distance
                distance_matrix = distance_func(X[ind_beg:ind_end], self.X_train)
                k_ind_matrix = np.argsort(distance_matrix, axis=1)[:, :self.k]
                ind_list.append(k_ind_matrix)
                if return_distance:
                    dist_list.append(distance_matrix[
                                         np.arange(np.size(k_ind_matrix, 0))[:, np.newaxis],
                                         k_ind_matrix])
            else:
                if return_distance:
                    dist, ind = self.strategy.kneighbors(
                        X[ind_beg:ind_end], n_neighbors=self.k,
                        return_distance=return_distance)
                    dist_list.append(dist)
                    ind_list.append(ind)
                else:
                    ind_list.append(
                        self.strategy.kneighbors(
                            X[ind_beg:ind_end], n_neighbors=self.k,
                            return_distance=return_distance))
        if return_distance:
            return np.concatenate(dist_list), np.concatenate(ind_list)
        else:
            return np.concatenate(ind_list)

    def predict(self, X):
        if self.weights:
            dist, ind = self.find_kneighbors(X, return_distance=True)

            el_weights = 1 / (dist + 10 ** -5)

            res = []
            for i in range(ind.shape[0]):
                indeces = ind[i]
                w = el_weights[i]
                ind_list = []
                for index in indeces:
                    ind_list.append(self.y_train[index])
                counts = np.bincount(ind_list, weights=w)
                y = np.argmax(counts)
                res.append(y)
        else:
            ind = self.find_kneighbors(X, return_distance=False)
            res = []
            for indeces in ind:
                counts = np.bincount(indeces)
                y = np.argmax(counts)
                res.append(y)
        return np.array(res)
