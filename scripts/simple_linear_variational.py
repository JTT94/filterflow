import enum
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pykalman
import tensorflow as tf
from tqdm import tqdm

tf.config.set_visible_devices([], 'GPU')

from filterflow.base import State
from filterflow.models.simple_linear_gaussian import make_filter
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
    return sample[1].data.astype(np.float32)


class ResamplingMethodsEnum(enum.IntEnum):
    MULTINOMIAL = 0
    SYSTEMATIC = 1
    STRATIFIED = 2
    REGULARIZED = 3
    VARIANCE_CORRECTED = 4
    OPTIMIZED = 5


@tf.function
def routine(pf, initial_state, resampling_correction, observations_dataset, T, gradient_variables):
    with tf.GradientTape() as tape:
        tape.watch(gradient_variables)
        final_state = pf(initial_state, observations_dataset, n_observations=T, return_final=True)
        log_likelihood = tf.reduce_mean(final_state.log_likelihoods)
        correction_term = tf.reduce_mean(final_state.resampling_correction)
        if resampling_correction:
            res_for_grad = log_likelihood + correction_term
        else:
            res_for_grad = log_likelihood
    return log_likelihood, tape.gradient(res_for_grad, gradient_variables)


def get_gradient_descent_function():
    # This is a trick because tensorflow doesn't allow you to create variables inside a decorated function
    @tf.function
    def gradient_descent(pf, initial_state, observations_dataset, T, gradient_variables, n_iter, optimizer):
        loss = tf.TensorArray(dtype=tf.float32, size=n_iter + 1, dynamic_size=False)
        for i in tf.range(n_iter):
            with tf.GradientTape() as tape:
                tape.watch(gradient_variables)
                final_state = pf(initial_state, observations_dataset, T, return_final=True)
                loss_value = -tf.reduce_mean(final_state.log_likelihoods)
            loss = loss.write(tf.cast(i, tf.int32), loss_value)
            grads = tape.gradient(loss_value, gradient_variables)
            optimizer.apply_gradients(zip(grads, gradient_variables))
        final_state = pf(initial_state, observations_dataset, T, return_final=True)
        loss = loss.write(tf.cast(n_iter, tf.int32), -tf.reduce_mean(final_state.log_likelihoods))
        return [tf.convert_to_tensor(var) for var in gradient_variables], loss.stack()

    return gradient_descent


def compare_learning_rates(pf, initial_state, observations_dataset, T, gradient_variables, initial_values, n_iter,
                           optimizer_maker, learning_rates):
    loss_profiles = []
    reset_ops = [k.assign(v) for k, v in zip(gradient_variables, initial_values)]
    for learning_rate in tqdm(learning_rates):
        optimizer = optimizer_maker(learning_rate=learning_rate)
        gradient_descent_function = get_gradient_descent_function()
        with tf.control_dependencies([reset_ops]):
            final_variables, loss_profile = gradient_descent_function(pf, initial_state, observations_dataset, T,
                                                                      gradient_variables, n_iter, optimizer)
        loss_profiles.append(loss_profile.numpy())
    return loss_profiles


def plot_losses(loss_profiles_df, filename, savefig):
    fig, ax = plt.subplots(figsize=(5, 5))
    loss_profiles_df.plot(ax=ax)
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
         T=100, batch_size=1, n_particles=25,
         data_seed=0, n_iter=50, savefig=False):
    transition_matrix = 0.5 * np.eye(2, dtype=np.float32)
    transition_covariance = np.eye(2, dtype=np.float32)
    observation_matrix = np.eye(2, dtype=np.float32)
    observation_covariance = 0.1 * np.eye(2, dtype=np.float32)

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

    init_transition_matrix = (0.5 * np.eye(2) + 0.1 * np_random_state.randn(2, 2)).astype(np.float32)

    modifiable_transition_matrix = tf.Variable(init_transition_matrix, trainable=True)
    observation_matrix = tf.convert_to_tensor(observation_matrix)
    transition_covariance_chol = tf.linalg.cholesky(transition_covariance)
    observation_covariance_chol = tf.linalg.cholesky(observation_matrix)

    initial_particles = np_random_state.normal(0., 1., [batch_size, n_particles, 2]).astype(np.float32)
    initial_state = State(initial_particles)

    smc = make_filter(observation_matrix, modifiable_transition_matrix, observation_covariance_chol,
                      transition_covariance_chol,
                      resampling_method, resampling_criterion)

    variables = [modifiable_transition_matrix]

    def optimizer_maker(learning_rate):
        # tf.function doesn't like creating variables. This is a way to create them outside the graph
        # We can't reuse the same optimizer because it would be giving a warmed-up momentum to the ones run later
        optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
        return optimizer

    initial_values = [init_transition_matrix]

    losses = compare_learning_rates(smc, initial_state, observation_dataset, T, variables,
                                    initial_values, n_iter, optimizer_maker, learning_rates)
    losses_df = pd.DataFrame(np.stack(losses).T, columns=learning_rates)
    losses_df.columns.name = 'learning rate'
    losses_df.columns.epoch = 'epoch'

    plot_losses(losses_df, resampling_method_enum.name, savefig)


if __name__ == '__main__':
    learning_rates = np.logspace(-5, 1, 10).astype(np.float32)
    main(ResamplingMethodsEnum.REGULARIZED, 0.5, T=100, n_particles=4, batch_size=4, learning_rates=learning_rates, n_iter=500,
         resampling_kwargs=dict(epsilon=0.5, scaling=0.75, convergence_threshold=1e-4))
