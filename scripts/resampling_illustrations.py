import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import flags, app
from tensorflow_probability.python.internal.samplers import split_seed

from filterflow.base import State
from filterflow.resampling import MultinomialResampler, SystematicResampler, StratifiedResampler, RegularisedTransform, \
    CorrectedRegularizedTransform
from filterflow.resampling.differentiable import PartiallyCorrectedRegularizedTransform
from filterflow.resampling.differentiable.loss import SinkhornLoss
from filterflow.resampling.differentiable.optimized import OptimizedPointCloud
from filterflow.resampling.differentiable.optimizer.sgd import SGD
from scripts.simple_linear_common import ResamplingMethodsEnum


def resampling_method_factory(resampling_method_enum, resampling_kwargs):
    if resampling_method_enum == ResamplingMethodsEnum.MULTINOMIAL:
        resampling_method = MultinomialResampler()
    elif resampling_method_enum == ResamplingMethodsEnum.SYSTEMATIC:
        resampling_method = SystematicResampler()
    elif resampling_method_enum == ResamplingMethodsEnum.STRATIFIED:
        resampling_method = StratifiedResampler()
    elif resampling_method_enum == ResamplingMethodsEnum.REGULARIZED:
        resampling_method = RegularisedTransform(**resampling_kwargs)
    elif resampling_method_enum == ResamplingMethodsEnum.VARIANCE_CORRECTED:
        regularized_resampler = RegularisedTransform(**resampling_kwargs)
        resampling_method = PartiallyCorrectedRegularizedTransform(regularized_resampler)
    elif resampling_method_enum == ResamplingMethodsEnum.OPTIMIZED:

        lr = resampling_kwargs.pop('lr', resampling_kwargs.pop('learning_rate', 0.1))
        decay = resampling_kwargs.pop('lr', 0.95)
        symmetric = resampling_kwargs.pop('symmetric', True)
        loss = SinkhornLoss(**resampling_kwargs, symmetric=symmetric)

        optimizer = SGD(loss, lr=lr, decay=decay)

        regularized_resampler = RegularisedTransform(**resampling_kwargs)
        intermediate_resampling_method = PartiallyCorrectedRegularizedTransform(regularized_resampler)

        resampling_method = OptimizedPointCloud(optimizer, intermediate_resampler=intermediate_resampling_method)
    elif resampling_method_enum == ResamplingMethodsEnum.CORRECTED:
        resampling_method = CorrectedRegularizedTransform(**resampling_kwargs)
    else:
        raise ValueError(f'resampling_method_name {resampling_method_enum} is not a valid ResamplingMethodsEnum')
    return resampling_method


def plot_scatter(df, name, n_particles, kwargs):
    if name != "initial":
        full_name = name + "_" + "_".join(f'{k}_{str(v)}' for k, v in kwargs.items())
    else:
        full_name = 'degenerate'
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    data = df.groupby(["x", "y"]).sum().reset_index()
    ax.scatter(data.x, data.y, s=100 * n_particles * data.w)
    fig.tight_layout()
    fig.savefig(os.path.join('./charts/', 'illustration_' + full_name + '.png'))


def plot_resampling(resampling_name, initial_state, resampled_state, n_particles, kwargs):
    initial_state_dataframe = pd.DataFrame(initial_state.particles[0],
                                           columns=['x', 'y'])
    initial_state_dataframe['w'] = initial_state.weights[0]

    plot_scatter(initial_state_dataframe, "initial", n_particles, {})

    resampled_state_dataframe = pd.DataFrame(resampled_state.particles[0].numpy(),
                                             columns=['x', 'y'])
    resampled_state_dataframe['w'] = resampled_state.weights[0].numpy()

    plot_scatter(resampled_state_dataframe, resampling_name, n_particles, kwargs)


def main(resampling_method_value, resampling_kwargs=None, n_particles=50, data_seed=111, resampling_seed=555):
    resampling_method_enum = ResamplingMethodsEnum(resampling_method_value)

    resampling_method = resampling_method_factory(resampling_method_enum, resampling_kwargs.copy())

    np_random_state = np.random.RandomState(seed=data_seed)
    data = np_random_state.uniform(-1., 1., [1, n_particles, 2]).astype(np.float32)
    weights = np_random_state.uniform(0., 1., (1, n_particles)).astype(np.float32) ** 2
    weights = weights / np.sum(weights)
    log_weights = np.log(weights)

    state = State(data, log_weights)
    flags = tf.ones([1], dtype=tf.bool)

    resampling_seed, = split_seed(resampling_seed, n=1)

    resampled_state = resampling_method.apply(state, flags, resampling_seed)
    plot_resampling(resampling_method_enum.name, state, resampled_state, n_particles, resampling_kwargs)


# define flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('resampling_method', ResamplingMethodsEnum.VARIANCE_CORRECTED, 'resampling_method')
flags.DEFINE_float('epsilon', 0.05, 'epsilon')
flags.DEFINE_float('scaling', 0.85, 'scaling')
flags.DEFINE_float('learning_rate', -1., 'learning_rate')
flags.DEFINE_float('decay', 0.5, 'log_learning_rate_min')
flags.DEFINE_float('convergence_threshold', 1e-6, 'convergence_threshold')
flags.DEFINE_integer('n_particles', 50, 'n_particles', lower_bound=4)
flags.DEFINE_integer('max_iter', 500, 'max_iter', lower_bound=1)
flags.DEFINE_integer('data_seed', 25, 'data_seed')
flags.DEFINE_integer('resampling_seed', 50, 'resampling_seed')


def flag_main(argb):
    print('resampling_method: {0}'.format(ResamplingMethodsEnum(FLAGS.resampling_method).name))
    print('epsilon: {0}'.format(FLAGS.epsilon))
    print('convergence_threshold: {0}'.format(FLAGS.convergence_threshold))
    print('n_particles: {0}'.format(FLAGS.convergence_threshold))
    print('scaling: {0}'.format(FLAGS.scaling))
    print('max_iter: {0}'.format(FLAGS.max_iter))
    print('learning_rate: {0}'.format(FLAGS.learning_rate))
    print('decay: {0}'.format(FLAGS.learning_rate))

    kwargs = {'epsilon': FLAGS.epsilon, 'learning_rate': FLAGS.learning_rate, 'scaling': FLAGS.scaling,
              'convergence_threshold': FLAGS.convergence_threshold, 'max_iter': FLAGS.max_iter, 'decay': FLAGS.decay}
    main(FLAGS.resampling_method,
         n_particles=FLAGS.n_particles,
         resampling_kwargs=kwargs,
         resampling_seed=FLAGS.resampling_seed,
         data_seed=FLAGS.data_seed)


if __name__ == '__main__':
    app.run(flag_main)
