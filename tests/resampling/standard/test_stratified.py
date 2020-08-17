import tensorflow as tf

from filterflow.resampling.standard.stratified import _stratified_spacings


class TestStratified(tf.test.TestCase):
    def setUp(self):
        tf.random.set_seed(42)

    def test_stratified_spacings(self):
        spacings = _stratified_spacings(5, 3)
        self.assertEquals(spacings.shape.as_list(), [3, 5])
        self.assertAllGreater(spacings[:, 1:] - spacings[:, :-1], 0.)
        self.assertAllInRange(spacings, 0., 1.)
