import abc

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
            tf.print("Ricatti solver didn't converge - this iterate was not corrected. "
                     "Try with a smaller step size or increase precision in Sinkhorn iterates")
            tf.print("final error: ", final_diff)
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
                                    [tf.eye(tensor.shape[1], batch_shape=[tensor.shape[0]]), tensor, tf.constant(0)],
                                    back_prop=True)

    if sort:
        eigen = tf.linalg.diag_part(U_res)
        argsort = tf.argsort(eigen)
        ordered_Q = tf.gather(Q_res, argsort, batch_dims=1, axis=2)
        return ordered_Q, tf.matmul(tf.matmul(ordered_Q, tensor, transpose_a=True), ordered_Q)

    return Q_res, U_res


@tf.function
def matrix_sign(tensor, n_iter, use_newton_schulze=True, threshold=1e-5):
    # TODO: maybe autodiff: th5.7 in https://www.maths.manchester.ac.uk/~higham/fm/OT104HighamChapter5.pdf

    eye = tf.eye(tensor.shape[1], batch_shape=[tensor.shape[0]])
    float_n = tf.cast(tensor.shape[1], float)

    # vals = tf.abs(tf.linalg.svd(tensor, compute_uv=False))
    # TODO: for some reason svd fails with a cryptic error... Will need to reproduce and send to tf support

    vals = tf.stop_gradient(tf.abs(tf.linalg.eigvals(tensor)))
    # we don't need the gradient, this is a numerical trick to prevent underflow
    INF = tf.fill(vals.shape, 1e6)
    non_null_vals = tf.where(tf.abs(vals) > 0., vals, INF)

    min_non_zero = tf.reshape(tf.reduce_min(non_null_vals, 1), [-1, 1, 1])
    shifted_tensor = tensor + 0.5 * min_non_zero * eye

    @tf.function
    def one_step(X):
        if use_newton_schulze:
            X_2 = tf.matmul(X, X)
            X_new = 0.5 * (tf.linalg.matmul(X, 3 * eye - X_2))
        else:
            det = tf.abs(tf.linalg.det(X))
            scaling_factor = tf.reshape(tf.math.pow(det, -1 / float_n), [-1, 1, 1])
            scaled_X = scaling_factor * X
            X_inv = tf.linalg.inv(scaled_X)
            X_new = 0.5 * (scaled_X + X_inv)
        return X_new

    def cond(_X, error, i):
        cond_val = tf.logical_and(i < n_iter, error > threshold)
        return cond_val

    def body(X, error, i):
        new_X = one_step(X)
        temp = 0.5 * (tf.matmul(new_X, new_X) + new_X)
        new_error = tf.reduce_max(tf.linalg.norm(tf.matmul(temp, temp) - temp, ord=1, axis=[1, 2]))

        return new_X, new_error, tf.add(i, 1)

    i0 = tf.constant(0)
    init_error = 2. * threshold

    sign, _final_error, _final_i = tf.while_loop(cond, body, [shifted_tensor, init_error, i0])
    mask = tf.math.is_finite(sign)
    return tf.where(mask, sign, tf.zeros_like(sign))


@tf.function
def _block_matrix(I, J, K, L):
    """[[I, J]
        [K, L]]
    """
    res = tf.concat([tf.concat([I, J], -1),
                     tf.concat([K, L], -1)],
                    -2)
    return res


@tf.custom_gradient
def _solve_petkov(A, transport_matrix, n, n_iter, use_newton_schulze):
    I = tf.eye(n, batch_shape=[transport_matrix.shape[0]])

    transport_matrix_t = tf.linalg.matrix_transpose(transport_matrix)
    hamiltonian_matrix = _block_matrix(-transport_matrix, -I, -A, transport_matrix_t)

    sign_matrix = matrix_sign(hamiltonian_matrix, n_iter, use_newton_schulze)
    projector = 0.5 * (tf.eye(2 * n, batch_shape=[transport_matrix.shape[0]]) - sign_matrix)
    Q, _ = tf.linalg.qr(projector)

    upper_left = Q[:, :n, :n]
    lower_left = Q[:, n:, :n]

    # TODO: Solve or invert?
    # delta = tf.linalg.matmul(lower_left, tf.linalg.inv(upper_left))

    delta = tf.linalg.solve(upper_left, tf.linalg.matrix_transpose(lower_left), adjoint=True)
    delta = _make_admissible(delta)

    def grad(d_delta):
        d_delta = tf.clip_by_value(d_delta, -1., 1.)
        dA, d_transport = tf.gradients(delta, [A, transport_matrix], d_delta)
        return dA, d_transport, None, None, None

    return delta, grad


class PetkovSolver(RicattiSolver):
    def __init__(self, n_iter, use_newton_schulze=False, name='PetkovSolver'):
        super(PetkovSolver, self).__init__(name=name)
        self._n_iter = tf.cast(n_iter, tf.int32)
        self._use_newton_schulze = tf.cast(use_newton_schulze, bool)

    def __call__(self, transport_matrix, w):
        n = transport_matrix.shape[1]
        A = _make_A(transport_matrix, w, n)
        res = _solve_petkov(A, transport_matrix, n, self._n_iter, self._use_newton_schulze)
        return res
