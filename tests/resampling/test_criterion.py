import math

import tensorflow as tf

from filterflow.resampling.criterion import neff, NeffCriterion


class MockState(object):
    def __init__(self, weights):
        self.weights = weights
        self.log_weights = tf.math.log(weights)
        self.n_particles = self.weights.shape.as_list()[1]


class TestNeffCriterion(tf.test.TestCase):
    def setUp(self):
        self.weights = tf.constant([[1 / 3, 1 / 3, 1 / 3],
                                    [0.05, 0.05, 0.9]])

        self.log_weights = tf.math.log(self.weights)

        self.state = MockState(self.weights)

        self._scaled_weights = 3. * self.weights
        self._scaled_log_weights = self.log_weights + math.log(3)

        self.neff_log_instance = NeffCriterion(0.5, True, True, True)
        self.neff_instance = NeffCriterion(0.5, True, False, True)

    def test_neff_normalized(self):
        flag, _ = neff(self.weights, True, False, 0.5 * 3)
        flag_log, _ = neff(self.log_weights, True, True, 0.5 * 3)

        self.assertAllEqual(flag, flag_log)
        self.assertAllEqual(flag, [False, True])

    def test_neff_unnormalized(self):
        flag, _ = neff(self._scaled_weights, False, False, 0.5 * 3)
        flag_log, _ = neff(self._scaled_log_weights, False, True, 0.5 * 3)

        self.assertAllEqual(flag, flag_log)
        self.assertAllEqual(flag, [False, True])

    def test_neff(self):
        log_flags, _ = self.neff_log_instance.apply(self.state)
        flags, _ = self.neff_instance.apply(self.state)

        self.assertAllEqual(log_flags, flags)
        self.assertAllEqual(flags, [False, True])
