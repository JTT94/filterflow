import tensorflow as tf

from filterflow.resampling.standard.base import StandardResamplerBase


@tf.function
def _uniform_spacings(n_particles, batch_size, seed=None):
    """ Generate non decreasing numbers x_i between [0, 1]

    :param n_particles: int
        number of particles
    :param batch_size: int
        batch size
    :return: spacings
    :rtype: tf.Tensor
    """
    if seed is None:
        u = tf.random.uniform((batch_size, n_particles + 1))
    else:
        u = tf.random.stateless_uniform((batch_size, n_particles + 1), seed=seed)
    z = tf.cumsum(-tf.math.log(u), 1)
    res = z[:, :-1] / tf.expand_dims(z[:, -1], 1)
    return res


class MultinomialResampler(StandardResamplerBase):
    def __init__(self, on_log=True, name='MultinomialResampler'):
        super(MultinomialResampler, self).__init__(name, on_log)

    @staticmethod
    def _get_spacings(n_particles, batch_size, seed):
        return _uniform_spacings(n_particles, batch_size, seed)
