import copy
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from filterflow import std
from filterflow.base import State
from filterflow.models.simple_linear_gaussian import make_filter
from filterflow.resampling import MultinomialResampler, SystematicResampler, StratifiedResampler, RegularisedTransform
from filterflow.resampling.criterion import NeverResample, AlwaysResample, NeffCriterion
from filterflow.resampling.differentiable import PartiallyCorrectedRegularizedTransform
from filterflow.resampling.differentiable.loss import SinkhornLoss
from filterflow.resampling.differentiable.optimized import OptimizedPointCloud
from filterflow.resampling.differentiable.optimizer.sgd import SGD
from scripts.simple_linear_common import get_data, kf_loglikelihood, ResamplingMethodsEnum


@tf.function(experimental_relax_shapes=True)
def get_states(pf, initial_state, observations_dataset, T, filter_seed):
    states = pf(initial_state, observations_dataset, n_observations=T, return_final=False, seed=filter_seed)
    return states


def kalman_main(kf, data, savefig):
    _, cov = kf.filter(data)
    stdevs = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
    stdevs = stdevs.mean(0, keepdims=True)

    stdevs_df = pd.DataFrame(stdevs,
                             columns=[r'$\sigma(x_1)$', r'$\sigma(x_2)$'],
                             index=["Kalman Filter"]).T.reset_index()
    if savefig:
        filename = f'kalman_std_values.tex'
        stdevs_df.to_latex(buf=os.path.join('./tables/', filename),
                           float_format='{:,.3f}'.format, escape=False, index=False)
    else:
        print(stdevs_df.to_latex(float_format='{:,.3f}'.format, escape=False, index=False))


def main(resampling_method_value, resampling_neff, resampling_kwargs=None, T=150, batch_size=50, n_particles=25,
         data_seed=0, filter_seed=555, savefig=False):
    transition_matrix = 0.5 * np.eye(2, dtype=np.float32)
    transition_covariance = np.eye(2, dtype=np.float32)
    observation_matrix = np.eye(2, dtype=np.float32)
    observation_covariance = 0.1 * np.eye(2, dtype=np.float32)

    resampling_method_enum = ResamplingMethodsEnum(resampling_method_value)

    np_random_state = np.random.RandomState(seed=data_seed)
    data, kf = get_data(transition_matrix, observation_matrix, transition_covariance, observation_covariance, T,
                        np_random_state)
    observation_dataset = tf.data.Dataset.from_tensor_slices(data)

    if resampling_method_enum == ResamplingMethodsEnum.KALMAN:
        return kalman_main(kf, data, savefig)

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

    initial_particles = np_random_state.normal(0., 1., [batch_size, n_particles, 2]).astype(np.float32)
    initial_state = State(tf.constant(initial_particles))

    smc = make_filter(observation_matrix, transition_matrix, observation_covariance_chol,
                      transition_covariance_chol, resampling_method, resampling_criterion)

    states = get_states(smc,
                        initial_state,
                        observation_dataset,
                        tf.constant(T),
                        tf.constant(filter_seed))

    stddevs = std(states, keepdims=False).numpy()
    stddevs_df = stddevs

    # elbos_df = stdevs_df.describe().loc[['mean', 'std']].reset_index()
    #
    # if savefig:
    #     filename = f"{resampling_method_enum.name}_batchsize_{batch_size}_epsilon_{resampling_kwargs.get('epsilon')}_N_{n_particles}_stddev_values.tex"
    #     elbos_df.to_latex(buf=os.path.join('./tables/', filename),
    #                       float_format='{:,.3f}'.format, escape=False, index=False)
    # else:
    #     print(elbos_df.to_latex(float_format='{:,.3f}'.format, escape=False, index=False))


if __name__ == '__main__':
    for resampling_method in {ResamplingMethodsEnum.VARIANCE_CORRECTED, ResamplingMethodsEnum.REGULARIZED}:
        for n_particles in {25, 50, 100}:
            main(resampling_method, 0.9, T=150, n_particles=n_particles, batch_size=100,
                 resampling_kwargs=dict(epsilon=0.5, scaling=0.95, convergence_threshold=1e-3),
                 filter_seed=555, data_seed=111, savefig=True)

    main(ResamplingMethodsEnum.KALMAN, 0.5, T=150, n_particles=25, batch_size=100,
         resampling_kwargs=dict(epsilon=0.5, scaling=0.75, convergence_threshold=1e-3),
         filter_seed=555, data_seed=111, savefig=True)
