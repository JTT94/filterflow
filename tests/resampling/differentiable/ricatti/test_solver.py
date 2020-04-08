import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.resampling.differentiable.regularized_transport.plan import transport
from filterflow.resampling.differentiable.ricatti.solver import _make_admissible, _make_nil, RicattiSolver, make_ode_fun


class TestFunctions(tf.test.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.n_particles = 50
        self.tensor = tf.random.uniform([self.batch_size, self.n_particles, self.n_particles], -1., 1.)

    def _assert_nil(self, tensor):
        self.assertAllEqual(tensor.shape, [self.batch_size, self.n_particles, self.n_particles])
        self.assertAllClose(tf.reduce_sum(tensor, 1), tf.zeros([self.batch_size, self.n_particles]), atol=1e-5)
        self.assertAllClose(tf.reduce_sum(tensor, 2), tf.zeros([self.batch_size, self.n_particles]), atol=1e-5)

    def test_make_nil(self):
        nil = _make_nil(self.tensor)
        self._assert_nil(nil)

    def test_make_admissible(self):
        admissible = _make_admissible(self.tensor)
        self._assert_nil(admissible)
        self.assertAllClose(admissible, tf.transpose(admissible, perm=[0, 2, 1]), atol=1e-5)


class TestRicatti(tf.test.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.horizon = tf.constant(5.)
        self.batch_size = 1
        self.n_particles = 5
        self.dimension = 2

        choice = np.random.binomial(1, 0.25, [self.batch_size, self.n_particles]).astype(bool)
        w = np.random.uniform(0., 1., [self.batch_size, self.n_particles]).astype(np.float32)
        w[choice] = 1e-3
        self.w = w / w.sum(1, keepdims=True)
        self.log_w = tf.math.log(self.w)
        self.ones = np.ones_like(self.w) / self.n_particles
        self.log_uniform = tf.math.log(self.ones)

        self.np_x = np.random.normal(0.1, 0.7, [self.batch_size, self.n_particles, self.dimension]).astype(np.float32)
        self.x = tf.constant(self.np_x)
        self.transport_matrix, _ = transport(self.x, self.log_w, tf.constant(0.5), tf.constant(0.85),
                                             tf.constant(1e-3),
                                             tf.constant(500), tf.constant(self.n_particles))

        self.ricatti_instance = RicattiSolver(horizon=self.horizon)

    def test_make_A(self):
        A = self.ricatti_instance._make_A(self.transport_matrix, self.w, float(self.n_particles))

        self.assertAllClose(tf.reduce_sum(A, 1), tf.zeros([self.batch_size, self.n_particles]), atol=1e-2)
        self.assertAllClose(tf.reduce_sum(A, 2), tf.zeros([self.batch_size, self.n_particles]), atol=1e-2)

        self.assertAllClose(A, tf.transpose(A, perm=[0, 1, 2]))

    def test_ode_fn(self):
        A = self.ricatti_instance._make_A(self.transport_matrix, self.w, float(self.n_particles))
        B = self.ricatti_instance._make_B(self.transport_matrix)
        ode_fn = make_ode_fun(A, B)
        delta_0 = tf.zeros_like(A)
        delta_prime = ode_fn(0., delta_0)
        delta = delta_0 + delta_prime
        self.assertAllClose(tf.reduce_sum(delta, 1), tf.zeros([self.batch_size, self.n_particles]), atol=1e-2)
        self.assertAllClose(tf.reduce_sum(delta, 2), tf.zeros([self.batch_size, self.n_particles]), atol=1e-2)

        self.assertAllClose(delta, tf.transpose(delta, perm=[0, 1, 2]))

    def test_routine(self):
        B = self.ricatti_instance._make_B(self.transport_matrix)
        A = self.ricatti_instance._make_A(B, self.w, float(self.n_particles))

        solution = self.ricatti_instance._routine(A, B)

        x_tilde = tf.einsum('ijk,ikl->ijl', self.transport_matrix + solution, self.x)
        uncorrected_x_tilde = tf.einsum('ijk,ikl->ijl', self.transport_matrix, self.x)

        w_ = tf.expand_dims(self.w[0], 1)
        x_ = self.x[0]
        weighted_mean = tf.reduce_sum(x_ * w_, 0, keepdims=True)
        x_ = x_ - weighted_mean

        weighted_cov = tf.einsum('ij,ik->jk', w_ * x_, x_)
        uncorrected_covariance_tilde = tfp.stats.covariance(uncorrected_x_tilde, sample_axis=[1])
        corrected_covariance_tilde = tfp.stats.covariance(x_tilde, sample_axis=[1])

        self.assertAllClose(corrected_covariance_tilde[0], weighted_cov, atol=5e-2)
        self.assertNotAllClose(uncorrected_covariance_tilde[0], weighted_cov, atol=5e-2)

    def test_gradient(self):
        @tf.function
        def fun(w):
            w_ = w / tf.reduce_sum(w, 1)
            return tf.math.reduce_std(self.ricatti_instance(self.transport_matrix, w_))

        theoretical, numerical = tf.test.compute_gradient(fun, [tf.constant(self.w)])
        self.assertAllClose(theoretical[0], numerical[0], 1e-2)
