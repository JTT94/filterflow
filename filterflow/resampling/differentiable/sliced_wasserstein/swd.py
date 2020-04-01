import tensorflow as tf


@tf.function
def _project(z, theta):
    return tf.linalg.matmul(theta, z, transpose_b=True)


class SlicedWassersteinDistance(tf.Module):
    def __init__(self, n_slices, name='SlicedWassersteinDistance'):
        super(SlicedWassersteinDistance, self).__init__(name=name)
        self.n_slices = tf.cast(n_slices, int)

    def __call__(self, x, y, w_x, w_y):
        pass
