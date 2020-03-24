import tensorflow as tf

from filterflow.resampling.standard.base import StandardResamplerBase


@tf.function
def _uniform_spacings(n_particles, batch_size):
    """ Generate non decreasing numbers x_i between [0, 1]

    :param n_particles: int
        number of particles
    :param batch_size: int
        batch size
    :return: spacings
    :rtype: tf.Tensor
    """
    u = tf.random.uniform((batch_size, n_particles + 1))
    z = tf.cumsum(-tf.math.log(u), 1)
    res = z[:, :-1] / tf.expand_dims(z[:, -1], 1)
    return res


class MultinomialResampler(StandardResamplerBase):
    @staticmethod
    def _get_spacings(n_particles, batch_size):
        return _uniform_spacings(n_particles, batch_size)
