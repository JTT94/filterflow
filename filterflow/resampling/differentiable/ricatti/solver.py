import abc

import tensorflow as tf


@tf.function
def _make_nil(tensor):
    tensor = tensor - tf.reduce_mean(tensor, 1, keepdims=True)
    return tensor - tf.reduce_mean(tensor, 2, keepdims=True)


@tf.function
def _make_admissible(tensor):
    tensor = _make_nil(tensor)
    return 0.5 * (tensor + tf.transpose(tensor, perm=[0, 2, 1]))


class BaseSolver(tf.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def solve(self, t_0, z_0):
        """Returns value at infinity"""


class Euler(BaseSolver):
    """A simple explicit Euler solver for infinite horizon problems"""

    def __init__(self, ode_fn, step_size, convergence_threshold, max_horizon=tf.constant(50.), name='Euler'):
        super(Euler, self).__init__(name)
        self.ode_fn = ode_fn
        self.step_size = step_size
        self.convergence_threshold = convergence_threshold
        self.max_horizon = max_horizon

        self._next = tf.function(lambda t, z: self.__next(t, z, ode_fn, step_size))

    @staticmethod
    def __next(t, z, ode_fn, step_size):
        z_ = z + step_size * ode_fn(t, z)
        return t + step_size, z + step_size * ode_fn(t, (z_ + z) / 2)

    def _solve(self, t_0, z_0):
        def body(t, z, _diff):
            t_, z_ = self._next(t, z)
            return t_, z_, tf.reduce_max(tf.abs(z_ - z))

        def stop(t, z, diff):
            t_cond = t < self.max_horizon
            diff_cond = diff > self.convergence_threshold
            return tf.logical_and(t_cond, diff_cond)

        final_t, final_z, final_diff = tf.while_loop(stop, body, [t_0, z_0, 2. * self.convergence_threshold])
        return final_z

    solve = tf.function(_solve)


def make_ode_fun(A, B):
    @tf.function
    def ode_fn(_t, delta):
        b_delta = tf.matmul(B, delta)
        delta_delta = tf.matmul(delta, delta, transpose_b=True)
        delta_prime = b_delta + tf.transpose(b_delta, [0, 2, 1]) + delta_delta - A
        # _make_admissible is only there to protect against numerical instability:
        # as per theory at all time delta_prime should be symmetric and row-col summing to 0
        return -_make_admissible(delta_prime)

    return ode_fn


class RicattiSolver(tf.Module):
    """This is a adaptation of https://arxiv.org/pdf/1608.08179.pdf with backprop handled by the adjoint method"""

    def __init__(self, step_size=0.5, horizon=5., threshold=1e-3, name='RicattiSolver'):
        super(RicattiSolver, self).__init__(name=name)
        self.step_size = tf.cast(step_size, float)
        self.horizon = tf.cast(horizon, float)
        self.threshold = tf.cast(threshold, float)
        self._routine = tf.custom_gradient(
            lambda A, B: self.__routine(self.step_size, A, B, self.horizon, self.threshold))
        self._make_B = tf.function(self.__make_B)
        self._make_A = tf.function(self.__make_A)

    @staticmethod
    def __routine(step_size: tf.Tensor, A: tf.Tensor, B: tf.Tensor, horizon: tf.Tensor, threshold: tf.Tensor):
        ode_fn = make_ode_fun(A, B)
        solver = Euler(ode_fn, step_size, threshold, horizon)
        res = solver.solve(0., tf.zeros_like(A))

        def grad(d_delta):
            d_delta_ = _make_admissible(d_delta)
            return tf.gradients(res, [A, B], d_delta_)

        return res, grad

    @staticmethod
    def __make_B(transport_matrix):
        return tf.transpose(transport_matrix, perm=[0, 2, 1])

    @staticmethod
    def __make_A(transport_matrix, w, n_particles):
        W = tf.linalg.diag(w)
        TT = tf.matmul(transport_matrix, transport_matrix, transpose_a=True)
        return _make_admissible(n_particles * W - TT)

    def __call__(self, transport_matrix, w):
        n_particles = w.shape[1]
        float_n_particles = tf.cast(n_particles, float)
        B = self._make_B(transport_matrix)
        A = self.__make_A(transport_matrix, w, float_n_particles)
        final_delta = self._routine(A, B)
        return final_delta
