import tensorflow as tf

from filterflow.resampling.standard.base import StandardResamplerBase


@tf.function
def _stratified_spacings(n_particles, batch_size):
    """ Generate non decreasing numbers x_i between [0, 1]

    :param n_particles: int
        number of particles
    :param batch_size: int
        batch size
    :return: spacings
    :rtype: tf.Tensor
    """
    z = tf.random.uniform((batch_size, n_particles))
    z = z + tf.reshape(tf.linspace(0., n_particles-1., n_particles), [1, -1])
    return z / n_particles


class StratifiedResampler(StandardResamplerBase):
    @staticmethod
    def _get_spacings(n_particles, batch_size):
        return _stratified_spacings(n_particles, batch_size)
