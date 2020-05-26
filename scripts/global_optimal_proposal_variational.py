import enum
import os, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pykalman
import tensorflow as tf
from absl import flags, app
from tensorflow_probability.python.internal import samplers
from tqdm import tqdm

tf.config.set_visible_devices([], 'GPU')

from filterflow.base import State
from filterflow.models.optimal_proposal_linear_gaussian import make_filter
from filterflow.resampling import MultinomialResampler, SystematicResampler, StratifiedResampler, RegularisedTransform
from filterflow.resampling.criterion import NeverResample, AlwaysResample, NeffCriterion
from filterflow.resampling.differentiable import PartiallyCorrectedRegularizedTransform
from filterflow.resampling.differentiable.loss import SinkhornLoss
from filterflow.resampling.differentiable.optimized import OptimizedPointCloud
from filterflow.resampling.differentiable.optimizer.sgd import SGD

from scripts.optimal_proposal_common import get_data, ResamplingMethodsEnum, get_observation_matrix, \
    get_observation_covariance, get_transition_covariance, get_transition_matrix


@tf.function
def run_smc(smc, state, observations_dataset, T, mu, beta, log_sigma, seed):
    if seed is None:
        temp_seed = tf.random.uniform((), 0, 2 ** 32, tf.int32)
        seed, = samplers.split_seed(temp_seed, n=1, salt='init')
    if tf.size(seed) == 0:
        seed = tf.stack([seed, 0])

    iterator = iter(observations_dataset)
    sigma = tf.math.exp(log_sigma)
    for t in tf.range(T):
        seed, seed1, seed2 = samplers.split_seed(seed, n=3, salt='update')
        obs = iterator.get_next()
        state = smc.update(state, obs, [mu, beta, sigma], seed1, seed2)
    res = tf.reduce_mean(state.log_likelihoods)
    return res


@tf.function
def routine(pf, initial_state, observations_dataset, T, mu, beta, log_sigma, seed):
    with tf.GradientTape() as tape:
        tape.watch([mu, beta, log_sigma])
        log_likelihood = run_smc(pf, initial_state, observations_dataset, T, mu, beta, log_sigma, seed)
        res = -log_likelihood
    return res, tape.gradient(res, [mu, beta, log_sigma])


def get_gradient_descent_function(seed):
    # This is a trick because tensorflow doesn't allow you to create variables inside a decorated function

    def gradient_descent(pf, initial_state, observations_dataset, T, n_iter, optimizer, mu, beta, log_sigma,
                         initial_values):
        variables = [mu, beta, log_sigma]
        reset_operations = [k.assign(v) for k, v in zip(variables, initial_values)]
        loss = tf.TensorArray(dtype=tf.float32, size=n_iter + 1, dynamic_size=False)

        with tf.control_dependencies(reset_operations):
            for i in tf.range(n_iter):
                loss_value, grads = routine(pf, initial_state, observations_dataset, T, mu, beta, log_sigma,
                                            seed)
                loss = loss.write(tf.cast(i, tf.int32), loss_value)
                optimizer.apply_gradients(zip(grads, variables))
        final_log_likelihood = run_smc(pf, initial_state, observations_dataset, T, mu, beta, log_sigma, seed)
        loss = loss.write(tf.cast(n_iter, tf.int32), -final_log_likelihood)
        return [tf.convert_to_tensor(var) for var in variables], loss.stack()

    return gradient_descent


def compare_learning_rates(pf, initial_state, observations_dataset, T, mu, beta, log_sigma, initial_values,
                           n_iter, optimizer_maker, learning_rates, filter_seed, use_xla):
    loss_profiles = []
    for learning_rate in tqdm(learning_rates):
        optimizer = optimizer_maker(learning_rate=learning_rate)
        gradient_descent_function = tf.function(get_gradient_descent_function(filter_seed),
                                                experimental_compile=use_xla)
        final_variables, loss_profile = gradient_descent_function(pf, initial_state, observations_dataset, T, n_iter,
                                                                  optimizer, mu, beta, log_sigma,
                                                                  initial_values)
        loss_profiles.append(loss_profile.numpy())

    return loss_profiles


def plot_losses(loss_profiles_df, filename, savefig, dx, dy, dense, T):
    fig, ax = plt.subplots(figsize=(5, 5))
    loss_profiles_df.style.float_format = '${:,.1f}'.format
    loss_profiles_df.plot(ax=ax, legend=False)

    # ax.set_ylim(250, 700)
    ax.legend()
    fig.tight_layout()
    if savefig:
        fig.savefig(os.path.join('./charts/', f'global_variational_different_lr_loss_{filename}_dx_{dx}_dy_{dy}_dense_{dense}_T_{T}.png'))
    else:
        fig.suptitle(f'variational_different_loss_{filename}_dx_{dx}_dy_{dy}_dense_{dense}_T_{T}')
        plt.show()


def plot_variables(variables_df, filename, savefig):
    fig, ax = plt.subplots(figsize=(5, 5))
    variables_df.plot(ax=ax)
    fig.tight_layout()
    if savefig:
        fig.savefig(os.path.join('./charts/', f'global_variational_different_lr_variables_{filename}.png'))
    else:
        fig.suptitle(f'variational_different_lr_variables_{filename}')
        plt.show()


def main(resampling_method_value, resampling_neff, learning_rates=(1e-4, 1e-3), resampling_kwargs=None,
         alpha=0.42, dx=10, dy=3, observation_covariance=0.1, dense=False, T=20, batch_size=1, n_particles=25,
         data_seed=0, n_iter=50, savefig=False, filter_seed=0, use_xla=False):
    transition_matrix = get_transition_matrix(alpha, dx)
    transition_covariance = get_transition_covariance(dx)
    observation_matrix = get_observation_matrix(dx, dy, dense)
    observation_covariance = get_observation_covariance(observation_covariance, dy)

    resampling_method_enum = ResamplingMethodsEnum(resampling_method_value)

    np_random_state = np.random.RandomState(seed=data_seed)
    data = get_data(transition_matrix, observation_matrix, transition_covariance, observation_covariance, T,
                    np_random_state)
    observation_dataset = tf.data.Dataset.from_tensor_slices(data)

    if resampling_kwargs is None:
        resampling_kwargs = {}

    if resampling_neff == 0.:
        resampling_criterion = NeverResample()
    elif resampling_neff == 1.:
        resampling_criterion = AlwaysResample()
    else:
        resampling_criterion = NeffCriterion(resampling_neff, True)

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

        loss = SinkhornLoss(**resampling_kwargs, symmetric=True)
        optimizer = SGD(loss, lr=lr, decay=0.95)
        regularized_resampler = RegularisedTransform(**resampling_kwargs)

        resampling_method = OptimizedPointCloud(optimizer, intermediate_resampler=regularized_resampler)
    else:
        raise ValueError(f'resampling_method_name {resampling_method_enum} is not a valid ResamplingMethodsEnum')

    observation_matrix = tf.convert_to_tensor(observation_matrix)
    transition_covariance_chol = tf.linalg.cholesky(transition_covariance)
    observation_covariance_chol = tf.linalg.cholesky(observation_covariance)

    initial_particles = np_random_state.normal(0., 1., [batch_size, n_particles, dx]).astype(np.float32)
    initial_state = State(initial_particles)

    smc = make_filter(observation_matrix, transition_matrix, observation_covariance_chol,
                      transition_covariance_chol, resampling_method, resampling_criterion)

    scale = 0.5

    def optimizer_maker(learning_rate):
        # tf.function doesn't like creating variables. This is a way to create them outside the graph
        # We can't reuse the same optimizer because it would be giving a warmed-up momentum to the ones run later
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        return optimizer

    mu_init = scale * np_random_state.normal(0.5, 1., [dx]).astype(np.float32)
    beta_init = 0.5 + scale * np_random_state.normal(0., 1., [dx]).astype(np.float32)
    log_sigma_init = scale * np_random_state.normal(-0.5, 1., [dx]).astype(np.float32)

    mu = tf.Variable(mu_init, trainable=True)
    beta = tf.Variable(beta_init, trainable=True)
    log_sigma = tf.Variable(log_sigma_init, trainable=True)

    initial_values = [mu_init, beta_init, log_sigma_init]

    losses = compare_learning_rates(smc, initial_state, observation_dataset, T, mu, beta, log_sigma,
                                    initial_values, n_iter, optimizer_maker, learning_rates, filter_seed, use_xla)

    losses_df = pd.DataFrame(np.stack(losses).T, columns=np.log10(learning_rates))
    losses_df.columns.name = 'log learning rate'
    losses_df.columns.epoch = 'epoch'

    plot_losses(losses_df, resampling_method_enum.name, savefig, dx, dy, dense, T)


FLAGS = flags.FLAGS

flags.DEFINE_integer('resampling_method', ResamplingMethodsEnum.MULTINOMIAL, 'resampling_method')
flags.DEFINE_float('epsilon', 0.75, 'epsilon')
flags.DEFINE_float('resampling_neff', 0.5, 'resampling_neff')
flags.DEFINE_float('scaling', 0.75, 'scaling')
flags.DEFINE_float('log_learning_rate_min', -8, 'log_learning_rate_min')
flags.DEFINE_float('log_learning_rate_max', -5, 'log_learning_rate_max')
flags.DEFINE_integer('n_learning_rates', 4, 'log_learning_rate_max')
flags.DEFINE_float('convergence_threshold', 1e-3, 'convergence_threshold')
flags.DEFINE_integer('n_particles', 4, 'n_particles', lower_bound=4)
flags.DEFINE_integer('batch_size', 4, 'batch_size', lower_bound=1)
flags.DEFINE_integer('n_iter', 100, 'n_iter', lower_bound=10)
flags.DEFINE_integer('max_iter', 500, 'max_iter', lower_bound=1)
flags.DEFINE_integer('dx', 25, 'dx', lower_bound=1)
flags.DEFINE_integer('dy', 25, 'dy', lower_bound=1)
flags.DEFINE_integer('T', 20, 'T', lower_bound=1)
flags.DEFINE_boolean('savefig', True, 'Save fig')
flags.DEFINE_boolean('use_xla', False, 'Use XLA (experimental)')
flags.DEFINE_boolean('dense', True, 'dense')
flags.DEFINE_boolean('is_global_proposal', False, 'Use one proposal per time step or a global one?')
flags.DEFINE_integer('seed', 25, 'seed')


def flag_main(argb):
    print('resampling_method: {0}'.format(ResamplingMethodsEnum(FLAGS.resampling_method).name))
    print('epsilon: {0}'.format(FLAGS.epsilon))
    print('resampling_neff: {0}'.format(FLAGS.resampling_neff))
    print('convergence_threshold: {0}'.format(FLAGS.convergence_threshold))
    print('n_particles: {0}'.format(FLAGS.n_particles))
    print('batch_size: {0}'.format(FLAGS.batch_size))
    print('n_iter: {0}'.format(FLAGS.n_iter))
    print('T: {0}'.format(FLAGS.T))
    print('savefig: {0}'.format(FLAGS.savefig))
    print('scaling: {0}'.format(FLAGS.scaling))
    print('max_iter: {0}'.format(FLAGS.max_iter))
    print('dx: {0}'.format(FLAGS.dx))
    print('dy: {0}'.format(FLAGS.dy))
    print('dense: {0}'.format(FLAGS.dense))
    print('use_xla: {0}'.format(FLAGS.use_xla))
    learning_rates = np.logspace(FLAGS.log_learning_rate_min, FLAGS.log_learning_rate_max, FLAGS.n_learning_rates,
                                 base=10).astype(np.float32)
    main(FLAGS.resampling_method,
         resampling_neff=FLAGS.resampling_neff,
         T=FLAGS.T,
         n_particles=FLAGS.n_particles,
         batch_size=FLAGS.batch_size,
         savefig=FLAGS.savefig,
         n_iter=FLAGS.n_iter,
         dx=FLAGS.dx,
         dy=FLAGS.dy,
         dense=FLAGS.dense,
         learning_rates=learning_rates,
         resampling_kwargs=dict(epsilon=FLAGS.epsilon,
                                scaling=FLAGS.scaling,
                                convergence_threshold=FLAGS.convergence_threshold,
                                max_iter=FLAGS.max_iter),
         filter_seed=FLAGS.seed,
         use_xla=FLAGS.use_xla)


if __name__ == '__main__':
    app.run(flag_main)
