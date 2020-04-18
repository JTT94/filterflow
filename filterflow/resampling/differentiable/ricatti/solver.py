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


# @_memoize
# @tf.function
# def _get_centering_matrix(n):
#     float_n = tf.cast(n, float)
#     ones = tf.ones([n, 1], dtype=float)
#     centering_matrix = tf.eye(n, dtype=float) - tf.matmul(ones, ones, transpose_b=True) / float_n
#     return tf.expand_dims(centering_matrix, 0)


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


class RicattiSolver(tf.Module):
    """This is a adaptation of https://arxiv.org/pdf/1608.08179.pdf with backprop handled by unrolling autodiff"""

    def __init__(self, step_size=0.5, horizon=5., threshold=1e-3, name='RicattiSolver'):
        super(RicattiSolver, self).__init__(name=name)
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

    @staticmethod
    @tf.function
    def _make_A(transport_matrix, w, n_particles):
        W = tf.linalg.diag(w)
        TT = tf.matmul(transport_matrix, transport_matrix, transpose_a=True)
        return _make_admissible(n_particles * W - TT)

    @tf.function
    def __call__(self, transport_matrix, w):
        n_particles = w.shape[1]
        float_n_particles = tf.cast(n_particles, float)
        A = self._make_A(transport_matrix, w, float_n_particles)
        final_delta = self._routine(A, transport_matrix)
        return final_delta
