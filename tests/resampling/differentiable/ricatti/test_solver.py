import numpy as np
import scipy.linalg as linalg
import tensorflow as tf
import tensorflow_probability as tfp

tf.config.set_visible_devices([], 'GPU')

from filterflow.resampling.differentiable.regularized_transport.plan import transport
from filterflow.resampling.differentiable.ricatti.solver import _make_admissible, NaiveSolver, PetkovSolver, \
    make_ode_fun, _make_A, matrix_sign



class TestFunctions(tf.test.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.n_particles = 5
        self.tensor = tf.random.uniform([self.batch_size, self.n_particles, self.n_particles], -4., 1.)

    def _assert_nil(self, tensor):
        self.assertAllEqual(tensor.shape, [self.batch_size, self.n_particles, self.n_particles])
        self.assertAllClose(tf.reduce_sum(tensor, 1), tf.zeros([self.batch_size, self.n_particles]), atol=1e-5)
        self.assertAllClose(tf.reduce_sum(tensor, 2), tf.zeros([self.batch_size, self.n_particles]), atol=1e-5)

    def test_make_admissible(self):
        admissible = _make_admissible(self.tensor)
        self.assertAllClose(admissible, tf.transpose(admissible, perm=[0, 2, 1]), atol=1e-5)
        row_sum = tf.reduce_sum(admissible, 1)
        col_sum = tf.reduce_sum(admissible, 2)
        self.assertAllClose(row_sum, tf.zeros_like(row_sum), atol=1e-5)
        self.assertAllClose(col_sum, tf.zeros_like(col_sum), atol=1e-5)

    def test_matrix_sign(self):
        @tf.function
        def fun(tensor):
            return matrix_sign(tensor, tf.constant(100), tf.constant(1e-6))[0]
        theoretical, numerical = tf.test.compute_gradient(fun, [self.tensor], delta=1e-2)
        self.assertAllClose(theoretical[0], numerical[0], atol=1e-2)
        sc_sign = linalg.signm(self.tensor[0].numpy())
        tf_sign = fun(self.tensor)[0].numpy()
        self.assertAllClose(sc_sign, tf_sign, atol=1e-6)


def _measure_time(fun, *args):
    import time
    res = fun(*args)
    tic = time.time()
    for _ in range(10):
        _ = fun(*args)
    return time.time() - tic, res


class TestRicatti(tf.test.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.horizon = tf.constant(100.)
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
        self.transport_matrix = transport(self.x, self.log_w, tf.constant(0.5), tf.constant(0.85),
                                             tf.constant(1e-3),
                                             tf.constant(500), tf.constant(self.n_particles))

        self.naive_instance = NaiveSolver(horizon=self.horizon, threshold=tf.constant(1e-3),
                                          step_size=tf.constant(0.05))
        self.petkov_instance = PetkovSolver(tf.constant(10), use_newton_schulze=False)

    def test_make_A(self):
        A = _make_A(self.transport_matrix, self.w, float(self.n_particles))

        self.assertAllClose(tf.reduce_sum(A, 1), tf.zeros([self.batch_size, self.n_particles]), atol=1e-2)
        self.assertAllClose(tf.reduce_sum(A, 2), tf.zeros([self.batch_size, self.n_particles]), atol=1e-2)

        self.assertAllClose(A, tf.transpose(A, perm=[0, 1, 2]))

    def test_ode_fn(self):
        A = _make_A(self.transport_matrix, self.w, float(self.n_particles))
        ode_fn = make_ode_fun(A, self.transport_matrix)
        delta_0 = tf.zeros_like(A)
        delta_prime = ode_fn(0., delta_0)
        delta = delta_0 + delta_prime
        self.assertAllClose(tf.reduce_sum(delta, 1), tf.zeros([self.batch_size, self.n_particles]), atol=1e-2)
        self.assertAllClose(tf.reduce_sum(delta, 2), tf.zeros([self.batch_size, self.n_particles]), atol=1e-2)

        self.assertAllClose(delta, tf.transpose(delta, perm=[0, 1, 2]))

    def test_routine(self):
        A = self.naive_instance._make_A(self.transport_matrix, self.w, float(self.n_particles))

        solution = self.naive_instance._routine(A, self.transport_matrix)

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
        def fun_w(w):
            w_ = w / tf.reduce_sum(w, 1)
            return tf.math.reduce_std(self.naive_instance(self.transport_matrix, w_))

        theoretical, numerical = tf.test.compute_gradient(fun_w, [tf.constant(self.w)])
        self.assertAllClose(theoretical[0], numerical[0], 1e-5)

    def test_solvers_agree(self):
        naive_toc, naive_result = _measure_time(self.naive_instance, self.transport_matrix, self.w)
        sign_toc, sign_result = _measure_time(self.petkov_instance, self.transport_matrix, self.w)

        A = _make_A(self.transport_matrix, self.w, float(self.n_particles))

        import scipy.linalg as linalg
        eye = np.eye(self.n_particles)
        sc_toc, sc_sol = _measure_time(linalg.solve_continuous_are, -self.transport_matrix[0].numpy(), eye,
                                       A[0].numpy(), eye)

        print()
        print('sc_toc', sc_toc)
        print('naive_toc', naive_toc)
        print('sign_toc', sign_toc)
        self.assertAllClose(tf.transpose(sc_sol), sc_sol, atol=1e-10)
        self.assertAllClose(sign_result[0], tf.transpose(sign_result[0]), atol=1e-5)

        self.assertAllClose(naive_result[0], sc_sol, atol=1e-1)
        self.assertAllClose(sign_result[0], sc_sol, atol=1e-4)

        # ode_fn = make_ode_fun(A, self.transport_matrix)
        #
        # delta_prime_sc_sol = ode_fn(0., tf.expand_dims(tf.constant(sc_sol.astype(np.float32)), 0))
        # delta_prime_schur = ode_fn(0., schur_result)
        # delta_prime_naive = ode_fn(0., naive_result)
        # print(delta_prime_sc_sol)
        # print(delta_prime_schur)
        # print(delta_prime_naive)
        #
        # print()
        # print(tf.reduce_sum(schur_result[0], 0))
        # print(tf.reduce_sum(schur_result[0], 1))
        # print(schur_result[0])
        # print()
        # print(tf.reduce_sum(naive_result[0], 0))
        # print(tf.reduce_sum(naive_result[0], 1))
        #
        # print(naive_result[0])
