import enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm

import os, sys


import attr
import datetime

import seaborn

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import sonnet as snt

tf.config.set_visible_devices([], 'GPU')

# add to path
sys.path.append("../")

from filterflow.base import State
from filterflow.models.vrnn import make_filter, VRNNState, ObservationModelEnum
from filterflow.resampling import MultinomialResampler, SystematicResampler, StratifiedResampler, RegularisedTransform
from filterflow.resampling.criterion import NeverResample, AlwaysResample, NeffCriterion
from filterflow.resampling.differentiable import PartiallyCorrectedRegularizedTransform
from filterflow.resampling.differentiable.loss import SinkhornLoss
from filterflow.resampling.differentiable.optimized import OptimizedPointCloud
from filterflow.resampling.differentiable.optimizer.sgd import SGD
from data.data import create_pianoroll_dataset


def get_data():
    # get data
    data_dir = "../data/piano_data"
    path = os.path.join(data_dir, 'jsb.pkl')
    inputs_tensor, targets_tensor, lens, mean = create_pianoroll_dataset(path, split='train', batch_size=1)

    inputs_tensor = tf.expand_dims(inputs_tensor, 1)
    targets_tensor = tf.expand_dims(targets_tensor, 1)

    T = targets_tensor.shape.as_list()[0]
    observation_size = targets_tensor.shape.as_list()[-1]

    return targets_tensor, inputs_tensor, T, observation_size

class ResamplingMethodsEnum(enum.IntEnum):
    MULTINOMIAL = 0
    SYSTEMATIC = 1
    STRATIFIED = 2
    REGULARIZED = 3
    VARIANCE_CORRECTED = 4
    OPTIMIZED = 5


def get_gradient_descent_function():
    # This is a trick because tensorflow doesn't allow you to create variables inside a decorated function
    @tf.function
    def gradient_descent(pf, initial_state, observations_dataset, inputs_dataset, T, gradient_variables, n_iter, optimizer, seed):
        loss = tf.TensorArray(dtype=tf.float32, size=n_iter + 1, dynamic_size=False)
        for i in tf.range(n_iter):
            with tf.GradientTape() as tape:
                tape.watch(gradient_variables)
                final_state = pf(initial_state, observations_dataset, T, inputs_dataset, return_final=True, seed=seed)
                loss_value = -tf.reduce_mean(final_state.log_likelihoods)
            loss = loss.write(tf.cast(i, tf.int32), loss_value)
            grads = tape.gradient(loss_value, gradient_variables)
            optimizer.apply_gradients(zip(grads, gradient_variables))
        final_state = pf(initial_state, observations_dataset, T, return_final=True, seed=seed)
        loss = loss.write(tf.cast(n_iter, tf.int32), -tf.reduce_mean(final_state.log_likelihoods))
        return [tf.convert_to_tensor(var) for var in gradient_variables], loss.stack()

    return gradient_descent


def compare_learning_rates(pf, initial_state, observations_dataset, inputs_dataset, T, gradient_variables, initial_values, n_iter,
                           optimizer_maker, learning_rates, seed):
    loss_profiles = []
    reset_ops = [k.assign(v) for k, v in zip(gradient_variables, initial_values)]
    for learning_rate in tqdm(learning_rates):
        optimizer = optimizer_maker(learning_rate=learning_rate)
        gradient_descent_function = get_gradient_descent_function()
        with tf.control_dependencies([reset_ops]):
            final_variables, loss_profile = gradient_descent_function(pf, 
                                                                      initial_state, 
                                                                      observations_dataset, 
                                                                      inputs_dataset, 
                                                                      T,
                                                                      gradient_variables, 
                                                                      n_iter, 
                                                                      optimizer, 
                                                                      seed)
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


def main(resampling_method_value, 
         resampling_neff, 
         learning_rates=(1e-4, 1e-3), 
         resampling_kwargs=None,
         T=100, 
         batch_size=1, 
         latent_size=32, 
         rnn_hidden_size=16, 
         latent_encoder_layers = [32], 
         latent_encoded_size = 32, 
         n_particles=25,
         data_seed=0,
         seed=1,
         n_iter=50, 
         savefig=False):




    # get data
    np_random_state = np.random.RandomState(seed=data_seed)
    targets_tensor, inputs_tensor, T, observation_size = get_data()
    observation_dataset = tf.data.Dataset.from_tensor_slices(targets_tensor)
    inputs_data = tf.data.Dataset.from_tensor_slices(inputs_tensor)

    inputs_iter = iter(inputs_data)
    for i in range(200):
        inp = inputs_iter.get_next()
        if i >0:
            print(inp==prev)
        prev = inp
        
        print(inp==prev)

    # get resampling method
    resampling_method_enum = ResamplingMethodsEnum(resampling_method_value)

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


    


    observation_model_name = ObservationModelEnum.BERNOULLI
    smc = make_filter(latent_size, observation_size, rnn_hidden_size, latent_encoder_layers,
                latent_encoded_size, observation_model_name, resampling_method, resampling_criterion)

    # init smc and variables
    # initial state
    dimension = latent_size
    normal_dist = tfp.distributions.Normal(0., 1.)
    initial_latent_state = tf.zeros([batch_size, n_particles, dimension])
    initial_latent_state = tf.cast(initial_latent_state, dtype=float)
    latent_encoded = smc._transition_model.latent_encoder(initial_latent_state)

    # initial rnn_state
    initial_rnn_state = [normal_dist.sample([batch_size,n_particles,rnn_hidden_size])]*2
    initial_rnn_state = tf.concat(initial_rnn_state, axis=-1)

    # rnn_out
    initial_rnn_out = tf.zeros([batch_size, n_particles, rnn_hidden_size])

    initial_weights = tf.ones((batch_size, n_particles), dtype=float) / tf.cast(n_particles, float)
    log_likelihoods = tf.zeros(batch_size, dtype=float)
    initial_state = VRNNState(particles=initial_latent_state, 
                            log_weights = tf.math.log(initial_weights),
                            weights=initial_weights, 
                            log_likelihoods=log_likelihoods,
                            rnn_state = initial_rnn_state,
                            rnn_out = initial_rnn_out,
                            latent_encoded = latent_encoded)
    variables = list(smc.trainable_variables)

    def optimizer_maker(learning_rate):
        # tf.function doesn't like creating variables. This is a way to create them outside the graph
        # We can't reuse the same optimizer because it would be giving a warmed-up momentum to the ones run later
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        return optimizer

    initial_values = [var.value() for var in variables]

    losses = compare_learning_rates(smc, initial_state, observation_dataset, inputs_data,  T, variables,
                                    initial_values, n_iter, optimizer_maker, learning_rates, seed)
    losses_df = pd.DataFrame(np.stack(losses).T, columns=learning_rates)
    losses_df.columns.name = 'learning rate'
    losses_df.columns.epoch = 'epoch'

    plot_losses(losses_df, resampling_method_enum.name, savefig)

if __name__ == '__main__':
    learning_rates = np.logspace(-5, -1, 5, base=10).astype(np.float32)
    main(ResamplingMethodsEnum.REGULARIZED, 
         resampling_neff=0.5, 
         learning_rates=(1e-4, 1e-3), 
         T=100, 
         batch_size=1, 
         latent_size=32, 
         rnn_hidden_size=16, 
         latent_encoder_layers = [32], 
         latent_encoded_size = 32, 
         n_particles=25,
         data_seed=0,
         n_iter=50, 
         resampling_kwargs=dict(epsilon=0.5, scaling=0.75, convergence_threshold=1e-1))
