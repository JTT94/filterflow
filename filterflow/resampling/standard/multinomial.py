import tensorflow as tf

from filterflow.resampling.standard.base import StandardResampler


@tf.function
def uniform_spacings(n_particles, batch_size):
    u = tf.random.uniform((n_particles + 1, batch_size))
    z = tf.cumsum(-tf.math.log(u), 0)
    res = z[:-1, :] / z[-1, :]
    return res


class MultinomialResampler(StandardResampler):
    @staticmethod
    @tf.function
    def _get_spacings(n_particles, batch_size):
        return uniform_spacings(n_particles, batch_size)
