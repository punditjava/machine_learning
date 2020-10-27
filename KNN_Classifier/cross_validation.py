import numpy as np
from nearest_neighbors import *


def kfold(n, n_folds=3):
    all_idx = range(n)
    folds = np.array_split(all_idx, n_folds)
    res = []
    for k in range(n_folds):
        if k == 0:
            edu = np.hstack(folds[k + 1:])
        elif k == n_folds - 1:
            edu = np.hstack(folds[:k])
        else:
            edu = np.hstack((np.hstack(folds[:k]), np.hstack(folds[k + 1:])))
        val = folds[k]
        res.append((edu, val))
    return res


def knn_cross_val_score(X, y, k_list, score, cv, **kwargs):
    acc = {}
    for key in k_list:
        acc[key] = np.array([])
    a = KNNClassifier(k=max(k_list), **kwargs)
    if cv == None:
        cv = kfold(X.shape[0], 3)
    for m, i in enumerate(cv):
        a.fit(X[i[0]], y[i[0]])
        dist, kneighbors = a.find_kneighbors(X[i[1]], True)
        for j, k in enumerate(k_list):
            ans = []
            dist_k = dist[:, :k]
            kneighbors_k = kneighbors[:, :k]
            for ind, l in enumerate(kneighbors_k):
                if a.weights:
                    count_el = np.bincount(y[i[0]][l].astype('int64'),
                                           weights=1 / (dist_k[ind] + 0.00001))
                else:
                    count_el = np.bincount(y[i[0]][l].astype('int64'))
                ans.append(str(count_el.argmax()))
            acc[k] = np.append(acc[k],
                               (np.array(ans).astype('int64') ==
                                np.array(y[i[1]]).astype('int64')).sum() / len(i[1]))
    return acc
