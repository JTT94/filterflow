import tensorflow as tf

from filterflow.resampling.standard.multinomial import _uniform_spacings


class TestMultinomial(tf.test.TestCase):
    def setUp(self):
        tf.random.set_seed(42)

    def test_uniform_spacings(self):
        spacings = _uniform_spacings(5, 3)
        self.assertEquals(spacings.shape.as_list(), [3, 5])
        self.assertAllGreater(spacings[:, 1:] - spacings[:, :-1], 0.)
        self.assertAllInRange(spacings, 0., 1.)
