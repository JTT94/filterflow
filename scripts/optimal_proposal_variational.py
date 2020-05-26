import enum
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pykalman
import tensorflow as tf
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


def get_data(transition_matrix, observation_matrix, transition_covariance, observation_covariance, T=100,
             random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    kf = pykalman.KalmanFilter(transition_matrix, observation_matrix, transition_covariance, observation_covariance)
    sample = kf.sample(T, random_state=random_state)
    data = sample[1].data.astype(np.float32)
    return data.reshape(T, 1, 1, -1)


def get_observation_matrix(dx, dy, dense=False, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    if dense:
        return random_state.normal(0., 1., [dy, dx]).astype(np.float32)
    return np.eye(dy, dx, dtype=np.float32)


def get_transition_matrix(alpha, dx):
    arange = np.arange(1, dx + 1, dtype=np.float32)
    return alpha ** (np.abs(arange[None, :] - arange[:, None]) + 1)


def get_transition_covariance(dx):
    return np.eye(dx, dtype=np.float32)


def get_observation_covariance(r, dy):
    return r * np.eye(dy, dtype=np.float32)


class ResamplingMethodsEnum(enum.IntEnum):
    MULTINOMIAL = 0
    SYSTEMATIC = 1
    STRATIFIED = 2
    REGULARIZED = 3
    VARIANCE_CORRECTED = 4
    OPTIMIZED = 5


@tf.function
def run_smc(smc, state, observations_dataset, T, mu_ts, beta_ts, log_sigma_ts, seed):
    if seed is None:
        temp_seed = tf.random.uniform((), 0, 2 ** 32, tf.int32)
        seed, = samplers.split_seed(temp_seed, n=1, salt='init')
    if tf.size(seed) == 0:
        seed = tf.stack([seed, 0])

    iterator = iter(observations_dataset)
    for t in tf.range(T):
        mu_t = mu_ts[t]
        beta_t = beta_ts[t]
        sigma_t = tf.math.exp(log_sigma_ts[t])
        seed, seed1, seed2 = samplers.split_seed(seed, n=3, salt='update')
        obs = iterator.get_next()
        state = smc.update(state, obs, [mu_t, beta_t, sigma_t], seed1, seed2)
    res = tf.reduce_mean(state.log_likelihoods)
    return res


@tf.function
def routine(pf, initial_state, observations_dataset, T, mu_ts, beta_ts, log_sigma_ts, seed):
    with tf.GradientTape() as tape:
        tape.watch([mu_ts, beta_ts, log_sigma_ts])
        log_likelihood = run_smc(pf, initial_state, observations_dataset, T, mu_ts, beta_ts, log_sigma_ts, seed)
        res = -log_likelihood
    return res, tape.gradient(res, [mu_ts, beta_ts, log_sigma_ts])


def get_gradient_descent_function(seed):
    # This is a trick because tensorflow doesn't allow you to create variables inside a decorated function

    @tf.function
    def gradient_descent(pf, initial_state, observations_dataset, T, n_iter, optimizer, mu_ts, beta_ts, log_sigma_ts,
                         initial_values):
        variables = [mu_ts, beta_ts, log_sigma_ts]
        reset_operations = [k.assign(v) for k, v in zip(variables, initial_values)]
        loss = tf.TensorArray(dtype=tf.float32, size=n_iter + 1, dynamic_size=False)

        with tf.control_dependencies(reset_operations):
            for i in tf.range(n_iter):
                loss_value, grads = routine(pf, initial_state, observations_dataset, T, mu_ts, beta_ts, log_sigma_ts, seed)
                loss = loss.write(tf.cast(i, tf.int32), loss_value)
                optimizer.apply_gradients(zip(grads, variables))
        final_log_likelihood = run_smc(pf, initial_state, observations_dataset, T, mu_ts, beta_ts, log_sigma_ts, seed)
        loss = loss.write(tf.cast(n_iter, tf.int32), -final_log_likelihood)
        return [tf.convert_to_tensor(var) for var in variables], loss.stack()

    return gradient_descent


def compare_learning_rates(pf, initial_state, observations_dataset, T, mu_ts, beta_ts, log_sigma_ts, initial_values,
                           n_iter, optimizer_maker, learning_rates, filter_seed):
    loss_profiles = []
    for learning_rate in tqdm(learning_rates):
        optimizer = optimizer_maker(learning_rate=learning_rate)
        gradient_descent_function = get_gradient_descent_function(filter_seed)
        final_variables, loss_profile = gradient_descent_function(pf, initial_state, observations_dataset, T, n_iter,
                                                                  optimizer, mu_ts, beta_ts, log_sigma_ts,
                                                                  initial_values)
        loss_profiles.append(loss_profile.numpy())
    return loss_profiles


def plot_losses(loss_profiles_df, filename, savefig):
    fig, ax = plt.subplots(figsize=(5, 5))
    loss_profiles_df.plot(ax=ax, legend=False)
    fig.tight_layout()
    if savefig:
        fig.savefig(os.path.join('./charts/', f'variational_different_lr_loss_{filename}.png'))
    else:
        fig.suptitle(f'variational_different_loss_{filename}')
        plt.show()


def plot_variables(variables_df, filename, savefig):
    fig, ax = plt.subplots(figsize=(5, 5))
    variables_df.plot(ax=ax)
    fig.tight_layout()
    if savefig:
        fig.savefig(os.path.join('./charts/', f'variational_different_lr_variables_{filename}.png'))
    else:
        fig.suptitle(f'variational_different_lr_variables_{filename}')
        plt.show()


def main(resampling_method_value, resampling_neff, learning_rates=(1e-4, 1e-3), resampling_kwargs=None,
         alpha=0.42, dx=10, dy=3, observation_covariance=0.1, dense=False,
         T=100, batch_size=1, n_particles=25,
         data_seed=0, n_iter=50, savefig=False, filter_seed=0):
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

    mu_ts_init = scale * np_random_state.normal(0., 1., [T, dx]).astype(np.float32)
    beta_ts_init = 1. + scale * np_random_state.normal(0., 1., [T, dx]).astype(np.float32)
    log_sigma_ts_init = scale * np_random_state.normal(0., 1., [T, dx]).astype(np.float32)

    mu_ts = tf.Variable(mu_ts_init, trainable=True)
    beta_ts = tf.Variable(beta_ts_init, trainable=True)
    log_sigma_ts = tf.Variable(log_sigma_ts_init, trainable=True)

    def optimizer_maker(learning_rate):
        # tf.function doesn't like creating variables. This is a way to create them outside the graph
        # We can't reuse the same optimizer because it would be giving a warmed-up momentum to the ones run later
        optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
        return optimizer

    initial_values = [mu_ts_init, beta_ts_init, log_sigma_ts_init]

    losses = compare_learning_rates(smc, initial_state, observation_dataset, T, mu_ts, beta_ts, log_sigma_ts,
                                    initial_values, n_iter, optimizer_maker, learning_rates, filter_seed)
    losses_df = pd.DataFrame(np.stack(losses).T, columns=learning_rates)
    losses_df.columns.name = 'learning rate'
    losses_df.columns.epoch = 'epoch'

    plot_losses(losses_df, resampling_method_enum.name, savefig)


if __name__ == '__main__':
    learning_rates = np.logspace(-4., -2., 5, base=10).astype(np.float32)
    main(ResamplingMethodsEnum.REGULARIZED, 0.5, T=20, n_particles=4, batch_size=4, learning_rates=learning_rates,
         n_iter=250, resampling_kwargs=dict(epsilon=0.75, scaling=0.75, convergence_threshold=1e-3), filter_seed=42, savefig=True)
