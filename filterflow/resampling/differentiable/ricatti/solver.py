import abc

import numpy as np
import scipy.linalg as linalg
import tensorflow as tf

LOGGER = tf.get_logger()


def _memoize(f):
    """ Memoization decorator for a function taking a single argument """

    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret

    return memodict().__getitem__


@tf.function
def _make_nil(tensor, axis):
    return tensor - tf.reduce_mean(tensor, axis, keepdims=True)


@tf.function
def _make_admissible(tensor):
    symmetric_tensor = 0.5 * (tensor + tf.transpose(tensor, perm=[0, 2, 1]))
    row_mean = tf.reduce_mean(symmetric_tensor, 1, keepdims=True)
    col_mean = tf.reduce_mean(symmetric_tensor, 2, keepdims=True)
    mean = tf.reduce_mean(symmetric_tensor, [1, 2], keepdims=True)

    return symmetric_tensor - row_mean - col_mean + mean


class BaseSolver(tf.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def solve(self, ode_fn, t_0, z_0):
        """Returns value at infinity"""


class Euler(BaseSolver):
    """A simple explicit Euler solver for infinite horizon problems"""

    def __init__(self, step_size, convergence_threshold, max_horizon=tf.constant(50.), name='Euler'):
        super(Euler, self).__init__(name)
        self.step_size = step_size
        self.convergence_threshold = convergence_threshold
        self.max_horizon = max_horizon

        self._next = tf.function(self.__next)

    @staticmethod
    def __next(t, z, ode_fn, step_size):
        z_ = z + step_size * ode_fn(t, z)
        return t + step_size, z + step_size * ode_fn(t, (z_ + z) / 2)

    def _solve(self, ode_fn, t_0, z_0):
        @tf.function
        def _next(t, x):
            return self._next(t, x, ode_fn, self.step_size)

        def body(t, z, prev_diff, _is_decreasing):
            t_, z_ = _next(t, z)
            new_diff = tf.reduce_max(tf.abs(z_ - z) / self.step_size)
            return t_, z_, new_diff, new_diff < prev_diff

        def stop(t, z, diff, is_decreasing):
            t_cond = t < self.max_horizon
            diff_cond = diff > self.convergence_threshold
            return tf.reduce_all([t_cond, diff_cond, is_decreasing])

        final_t, final_z, final_diff, final_decreasing = tf.while_loop(stop,
                                                                       body,
                                                                       [t_0,
                                                                        z_0,
                                                                        tf.constant(float('inf')),
                                                                        tf.constant(True)])
        if not final_decreasing:
            return z_0

        return final_z

    solve = tf.function(_solve)


def make_ode_fun(A, transport_matrix):
    @tf.function
    def ode_fn(_t, delta):
        delta = _make_admissible(delta)
        # Do not allow small errors to propagate
        b_delta = tf.matmul(transport_matrix, delta, transpose_a=True)
        delta_delta = tf.matmul(delta, delta, transpose_a=True)
        delta_prime = A - b_delta - tf.transpose(b_delta, [0, 2, 1]) - delta_delta
        return delta_prime

    return ode_fn


class RicattiSolver(tf.Module, metaclass=abc.ABCMeta):
    def __init__(self, name='RicattiSolver'):
        super(RicattiSolver, self).__init__(name=name)

    @abc.abstractmethod
    def __call__(self, transport_matrix, w):
        """"""


@tf.function
def _make_A(transport_matrix, w, n_particles):
    W = tf.linalg.diag(w)
    TT = tf.matmul(transport_matrix, transport_matrix, transpose_a=True)
    return _make_admissible(n_particles * W - TT)


@tf.function
def _make_B(transport_matrix):
    return tf.linalg.matrix_transpose(transport_matrix)


class NaiveSolver(RicattiSolver):
    """This is a adaptation of https://arxiv.org/pdf/1608.08179.pdf with backprop handled by unrolling autodiff"""

    def __init__(self, step_size=0.5, horizon=5., threshold=1e-3, name='RicattiSolver'):
        super(NaiveSolver, self).__init__(name=name)
        self.step_size = tf.cast(step_size, float)
        self.horizon = tf.cast(horizon, float)
        self.threshold = tf.cast(threshold, float)
        self.solver = Euler(self.step_size, self.threshold, self.horizon)

        self._routine = tf.custom_gradient(
            lambda A, B: self._routine_without_custom_grad(self.solver, A, B))

    @staticmethod
    def _routine_without_custom_grad(solver: BaseSolver, A: tf.Tensor, transport_matrix: tf.Tensor):
        ode_fn = make_ode_fun(A, transport_matrix)
        res = solver.solve(ode_fn, 0., tf.zeros_like(A))

        @tf.function
        def grad(d_delta):
            d_delta = tf.clip_by_value(d_delta, -1., 1.)
            grads = tf.gradients(res, [A, transport_matrix], d_delta)
            return grads

        return res, grad

    @tf.function
    def __call__(self, transport_matrix, w):
        n_particles = w.shape[1]
        float_n_particles = tf.cast(n_particles, float)
        A = _make_A(transport_matrix, w, float_n_particles)
        final_delta = self._routine(A, transport_matrix)
        return final_delta


@tf.function
def schur_decomposition(tensor, max_iter, sort=False):
    def body(Q, U, i):
        Q_new, R_new = tf.linalg.qr(U)
        U_new = tf.linalg.matmul(R_new, Q_new)
        return Q_new @ Q, U_new, i + 1

    def cond(_Q, _U, i):
        return i < max_iter - 1

    Q_res, U_res, _ = tf.while_loop(cond, body,
                                    [tf.eye(tensor.shape[1], batch_shape=[tensor.shape[0]]), tensor, tf.constant(0)])

    if sort:
        eigen = tf.linalg.diag_part(U_res)
        argsort = tf.argsort(eigen)
        ordered_Q = tf.gather(Q_res, argsort, batch_dims=1, axis=2)
        return ordered_Q, tf.matmul(tf.matmul(ordered_Q, tensor, transpose_a=True), ordered_Q)

    return Q_res, U_res


def np_solve_sylvester(tensors, sign_matrices, d_sign_matrices):
    res = np.empty_like(tensors)
    for k in range(tensors.shape[0]):
        tensor = tensors[k]
        sign_matrix = sign_matrices[k]
        d_sign_matrix = d_sign_matrices[k]
        q = np.dot(tensor, d_sign_matrix) - np.dot(d_sign_matrix, tensor)
        a = sign_matrix
        b = -sign_matrix
        res[k] = linalg.solve_sylvester(a, b, q)
    return res


def tf_solve_sylvester(tensor, sign, d_sign):
    d_tensor = tf.numpy_function(np_solve_sylvester, [tensor, sign, d_sign], tf.float32)
    return d_tensor


@tf.function
def _matrix_norm(tensor, axis):
    """axis = 1 means summing over lines - norm 1
       axis = 2 means summing over columns - norm sup"""
    abs_tensor = tf.abs(tensor)
    sum_ = tf.reduce_sum(abs_tensor, axis)
    return tf.reduce_max(sum_, axis=-1)


@tf.function
@tf.custom_gradient
def matrix_sign(tensor, n_iter, threshold=1e-5):
    # TODO: maybe autodiff: th5.7 in https://www.maths.manchester.ac.uk/~higham/fm/OT104HighamChapter5.pdf
    n = tensor.shape[1]

    eye = tf.eye(tensor.shape[1], batch_shape=[tensor.shape[0]])
    float_n = tf.cast(tensor.shape[1], float)

    sqrt_svd_vals = tf.stop_gradient(tf.math.sqrt(tf.linalg.svd(tensor, compute_uv=False)))
    # no need for backprop as this *should* not impact the result

    max_ = tf.reduce_max(sqrt_svd_vals)
    non_null_vals = tf.where(tf.abs(sqrt_svd_vals) > 0., sqrt_svd_vals, max_)

    min_non_zero = tf.reshape(tf.reduce_min(non_null_vals, 1), [-1, 1, 1])
    shifted_tensor = tensor + min_non_zero * eye
    newton_schulz_flag = tf.reduce_all(_matrix_norm(eye - tf.matmul(tensor, tensor), 1) < 1.)

    @tf.function
    def newton_schulz(X):
        # this is faster and more stable but only locally convergent
        X_new = 0.5 * tf.matmul(X, 3 * eye - tf.matmul(X, X))
        mask = tf.ones_like(X_new, dtype=tf.bool)
        return X_new, mask

    @tf.function
    def newton(X):
        det = tf.stop_gradient(tf.abs(tf.linalg.det(X)))
        mask = tf.reshape(det > 0, [-1, 1, 1])
        scaling_factor = tf.reshape(tf.math.pow(det, -1 / float_n), [-1, 1, 1])
        scaled_X = tf.where(mask, scaling_factor * X, tf.eye(X.shape[1], batch_shape=[X.shape[0]]))
        X_inv = tf.linalg.inv(scaled_X)
        X_new = tf.where(mask, 0.5 * (scaled_X + X_inv), X)
        return X_new, mask

    @tf.function
    def cond(_X, error, i):
        cond_val = tf.logical_and(i < n_iter, error > threshold)
        return cond_val

    @tf.function
    def body(X, error, i):
        if newton_schulz_flag:
            new_X, mask = newton_schulz(X)
        else:
            new_X, mask = newton(X)

        temp = tf.where(mask, 0.5 * (tf.matmul(new_X, new_X) + new_X),
                        tf.eye(new_X.shape[1], batch_shape=[new_X.shape[0]]))
        new_error = tf.reduce_max(tf.linalg.norm(tf.matmul(temp, temp) - temp, ord=1, axis=[1, 2]))

        return new_X, new_error, tf.add(i, 1)

    i0 = tf.constant(0)
    init_error = 2. * threshold

    sign, _final_error, final_i = tf.while_loop(cond,
                                                body,
                                                [shifted_tensor,
                                                 init_error,
                                                 i0,
                                                 ])

    @tf.function
    def grad(d_sign):
        d_sign_non_null = tf.reshape(tf.reduce_max(tf.abs(d_sign), [1, 2]) > 0, [-1, 1, 1])
        grad_val, = tf.gradients(sign, [tensor], d_sign)
        grad_val_non_nan = tf.math.is_finite(grad_val)
        grad_val = tf.where(tf.logical_and(d_sign_non_null, grad_val_non_nan), grad_val, 0.)
        return grad_val, None, None

    return sign, grad


@tf.function
def _block_matrix(I, J, K, L):
    """[[I, J]
        [K, L]]
    """
    res = tf.concat([tf.concat([I, J], -1),
                     tf.concat([K, L], -1)],
                    -2)
    return res


@tf.function
@tf.custom_gradient
def qr(projector):
    q, r = tf.linalg.qr(projector)

    @tf.function
    def grad(dq):
        qdq = tf.matmul(q, dq, transpose_a=True)
        qdq_ = qdq - tf.linalg.matrix_transpose(qdq)
        r_diag = tf.linalg.diag_part(r)
        clipped_r_diag = tf.clip_by_value(r_diag, -float('inf'), -1e-5)
        # just make sure that r is "invertible", this doesn't change the result as dq is null on the second part
        clipped_r_diag = tf.where(r_diag > 0., r_diag, clipped_r_diag)
        clipped_r = r - tf.linalg.diag(r_diag) + tf.linalg.diag(clipped_r_diag)
        tril = tf.linalg.band_part(qdq_, -1, 0)

        def _triangular_solve(x):
            """Equiv to matmul(x, adjoint(matrix_inverse(r))) if r is upper-tri."""
            t_x = tf.linalg.matrix_transpose(x)
            return tf.linalg.matrix_transpose(tf.linalg.triangular_solve(clipped_r, t_x, lower=False, adjoint=False))

        grad_val = tf.matmul(q, _triangular_solve(tril))
        return tf.clip_by_value(grad_val, -1., 1.)  # anything outside results from a numerical instability.

    return q, grad


@tf.function
def _solve_petkov(A, transport_matrix, n, n_iter):
    I = tf.eye(n, batch_shape=[transport_matrix.shape[0]])

    transport_matrix_t = tf.linalg.matrix_transpose(transport_matrix)
    hamiltonian_matrix = _block_matrix(-transport_matrix, -I, -A, transport_matrix_t)

    sign_matrix = matrix_sign(hamiltonian_matrix, n_iter)
    projector = 0.5 * (tf.eye(2 * n, batch_shape=[transport_matrix.shape[0]]) - sign_matrix)
    Q = qr(projector)

    upper_left = Q[:, :n, :n]
    lower_left = Q[:, n:, :n]

    # TODO: Solve or invert?
    # TODO: the below should use the Linear Operator paradigm for optimization - upper_left is diagonal dominated...
    delta = tf.linalg.solve(upper_left, tf.linalg.matrix_transpose(lower_left), adjoint=True)
    ode_fun = make_ode_fun(A, transport_matrix)

    ode_res = tf.reduce_mean(tf.abs(ode_fun(0., delta)))
    mask = tf.reshape(ode_res < 1e-2, [-1, 1, 1])

    delta = tf.where(mask, delta, 0.)

    return _make_admissible(delta)


class PetkovSolver(RicattiSolver):
    """https://www.researchgate.net/publication/221014504_Numerical_Solution_of_High_Order_Matrix_Riccati_Equations
    """

    def __init__(self, n_iter, use_newton_schulze=False, name='PetkovSolver'):
        super(PetkovSolver, self).__init__(name=name)
        self._n_iter = tf.cast(n_iter, tf.int32)

    @staticmethod
    @tf.function
    @tf.custom_gradient
    def _call(transport_matrix, w, n_iter):
        n = transport_matrix.shape[1]
        A = _make_A(transport_matrix, w, n)
        res = _solve_petkov(A, transport_matrix, n, n_iter)

        @tf.function
        def grad(dres):
            dres_non_null = tf.reduce_max(tf.abs(dres), [1, 2]) > 0.
            dres = _make_admissible(dres)
            d_transport, dw = tf.gradients(res, [transport_matrix, w], dres)

            d_transport = tf.where(tf.reshape(dres_non_null, [-1, 1, 1]), d_transport, 0.)
            dw = tf.where(tf.reshape(dres_non_null, [-1, 1]), dw, 0.)

            return tf.clip_by_value(d_transport, -1., 1.), tf.clip_by_value(dw, -1., 1.), None

        return res, grad

    def __call__(self, transport_matrix, w):
        return self._call(transport_matrix, w, self._n_iter)
