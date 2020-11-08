import numpy as np
import oracles
import time
from scipy.special import expit


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function=None, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = oracles.BinaryLogistic(**kwargs)
        self.alpha = step_alpha
        self.beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        pass

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        self._iters = 1
        if w_0 is None:
            w_0 = np.zeros((X.shape[1], ))
        self._w = w_0
        func_prev = self.loss_function.func(X, y, w=self._w)
        history = {}
        if trace:
            history = {'time': [0], 'func': [func_prev], 'acc': [0]}
            _time = time.time()
        w = self._w
        while self._iters < self.max_iter + 1:
            grad = self.loss_function.grad(X, y, w=w)
            w -= grad * self.alpha / self._iters ** self.beta
            self._iters += 1
            _func = self.loss_function.func(X, y, w=w)
            if trace:
                this_time = time.time()
                history['time'].append(this_time - _time)
                _time = this_time
                history['func'].append(_func)
                history['acc'].append(np.sum(self.predict(X) == y) / y.shape[0])
            if abs(_func - func_prev) < self.tolerance:
                break
            func_prev = _func
        if trace:
            return history
        pass

    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        return np.sign(X.dot(self._w))

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        ex_pit = expit(X.dot(self._w))
        return np.vstack((ex_pit, 1 - ex_pit))

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self.loss_function.func(X, y, w=self._w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        return self.loss_function.grad(X, y, w=self._w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self._w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, batch_size, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход


        max_iter - максимальное число итераций (эпох)

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = oracles.BinaryLogistic(**kwargs)
        self.alpha = step_alpha
        self.beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.batch = batch_size
        self.seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно происходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        np.random.seed(self.seed)
        _iters = 1
        self._classes = np.unique(y)
        if w_0 is None:
            w_0 = np.zeros((X.shape[1],))
        self._w = w_0
        func_prev = self.loss_function.func(X, y, w=self._w)
        history = {}
        if trace:
            history = {'epoch_num': [0], 'time': [0],
                       'func': [func_prev], 'weights_diff': [0], 'acc': [0]}
            _time = time.time()
            epoch = 0
        _items = 0
        w = self._w
        x_size = X.shape[0]
        idx = x_size
        while _iters < self.max_iter + 1:
            if idx == x_size:
                perm = np.random.permutation(x_size)
                idx = 0
            indexes = perm[idx: idx + self.batch]
            _items += indexes.shape[0]
            idx = indexes.shape[0]
            grad = self.loss_function.grad(X[indexes], y[indexes], w=w)
            exp = self.alpha / _iters ** self.beta
            w -= grad * exp
            _iters += 1
            _func = self.loss_function.func(X, y, w=w)
            if trace:
                epoch_curr = _items / x_size
                if epoch_curr - epoch > log_freq:
                    history['epoch_num'].append(epoch_curr)
                    this_time = time.time()
                    history['time'].append(this_time - _time)
                    _time = this_time
                    history['func'].append(_func)
                    history['weights_diff'].append(((exp * grad) ** 2).sum() ** .5)
                    history['acc'].append(np.sum(self.predict(X) == y) / y.shape[0])
            if abs(_func - func_prev) < self.tolerance:
                break
            func_prev = _func
        if trace:
            return history