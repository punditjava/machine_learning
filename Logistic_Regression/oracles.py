import numpy as np
from scipy.special import expit
from scipy.spatial.distance import euclidean
from scipy.special import logsumexp
import scipy


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """

    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        """
        Задание параметров оракула.

        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef
        pass

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        return (np.logaddexp(np.zeros(y.shape),
                -y * (X.dot(w))).mean() +
                self.l2_coef * 0.5 * (euclidean(w, np.zeros(w.shape)) ** 2))

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        sigmoid_y = (expit(-y * X.dot(w)) * y)
        if isinstance(X, scipy.sparse.csr_matrix):
            return self.l2_coef * w - np.array(X.multiply(sigmoid_y[:, np.newaxis]).mean(axis=0)).flatten()
        elif isinstance(X, np.ndarray):
            return self.l2_coef * w - (X * sigmoid_y[:, np.newaxis]).mean(axis=0)