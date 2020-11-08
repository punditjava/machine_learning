import numpy as np


def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    grad = np.empty(w)
    func_value = function(w)
    v_eps = np.zeros_like(w)
    for i in enumerate(v_eps):
        v_eps[i] += eps
        grad[i] = (function(w + v_eps) - func_value) / eps
        v_eps[i] -= eps
    return grad
