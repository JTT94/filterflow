import copy
import os

import numpy as np
import pandas as pd
import tensorflow as tf

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
def get_elbos(pf, initial_state, observations_dataset, T, modifiable_transition_values, values, filter_seed):
    elbos = tf.TensorArray(dtype=tf.float32, size=values.shape[0])
    for i in tf.range(values.shape[0]):
        val = values[i]
        assign_op = modifiable_transition_values.assign(tf.linalg.diag(val))
        with tf.control_dependencies([assign_op]):
            final_state = pf(initial_state, observations_dataset, n_observations=T, return_final=True, seed=filter_seed)
        elbos = elbos.write(tf.cast(i, tf.int32), final_state.log_likelihoods / tf.cast(T, float))
    return elbos.stack()


def kalman_main(kf, data, values, T, savefig):
    log_likelihoods = []
    for val in values:
        transition_matrix = np.diag(val)
        kf_copy = copy.copy(kf)
        kf_copy.transition_matrices = transition_matrix
        log_likelihoods.append(kf_loglikelihood(kf_copy, data) / T)
    likelihoods_df = pd.Series(log_likelihoods,
                               name=r'$\ell(\theta_1, \theta_2)$',
                               index=pd.Index(values[:, 0],
                                              name=r'$\theta_1, \theta_2$')).to_frame().reset_index()
    if savefig:
        filename = f'kalman_likelihoods_values.tex'
        likelihoods_df.to_latex(buf=os.path.join('./tables/', filename),
                                float_format='{:,.3f}'.format, escape=False, index=False)
    else:
        print(likelihoods_df.to_latex(float_format='{:,.3f}'.format, escape=False, index=False))


def main(resampling_method_value, resampling_neff, resampling_kwargs=None, T=150, batch_size=50, n_particles=25,
         data_seed=0, values=(0.25, 0.5, 0.75), filter_seed=555, savefig=False):
    transition_matrix = 0.5 * np.eye(2, dtype=np.float32)
    transition_covariance = np.eye(2, dtype=np.float32)
    observation_matrix = np.eye(2, dtype=np.float32)
    observation_covariance = 0.1 * np.eye(2, dtype=np.float32)

    values = np.array(list(zip(values, values)), dtype=np.float32)

    resampling_method_enum = ResamplingMethodsEnum(resampling_method_value)

    np_random_state = np.random.RandomState(seed=data_seed)
    data, kf = get_data(transition_matrix, observation_matrix, transition_covariance, observation_covariance, T,
                        np_random_state)
    observation_dataset = tf.data.Dataset.from_tensor_slices(data)

    if resampling_method_enum == 6:
        return kalman_main(kf, data, values, T, savefig)

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
    observation_covariance_chol = tf.linalg.cholesky(observation_covariance)

    initial_particles = np_random_state.normal(0., 1., [batch_size, n_particles, 2]).astype(np.float32)
    initial_state = State(tf.constant(initial_particles))

    smc = make_filter(observation_matrix, modifiable_transition_matrix, observation_covariance_chol,
                      transition_covariance_chol, resampling_method, resampling_criterion)

    elbos = get_elbos(smc,
                      initial_state,
                      observation_dataset,
                      tf.constant(T),
                      modifiable_transition_matrix,
                      tf.constant(values),
                      tf.constant(filter_seed))

    elbos_df = pd.DataFrame(elbos.numpy(), pd.Index(values[:, 0], name=r'$\theta_1$, $\theta_2$'))

    elbos_df = elbos_df.T.describe().T[['mean', 'std']].reset_index()

    if savefig:
        filename = f"{resampling_method_enum.name}_batchsize_{batch_size}_N_{n_particles}_epsilon_{resampling_kwargs.get('epsilon')}_likelihoods_values.tex"
        elbos_df.to_latex(buf=os.path.join('./tables/', filename),
                          float_format='{:,.3f}'.format, escape=False, index=False)
    else:
        print(elbos_df.to_latex(float_format='{:,.3f}'.format, escape=False, index=False))


if __name__ == '__main__':
    for n_particles in {25, 50, 100}:
        print(n_particles)
        main(ResamplingMethodsEnum.MULTINOMIAL, 0.5, T=150, n_particles=n_particles, batch_size=100,
             resampling_kwargs=dict(epsilon=0.5, scaling=0.9, convergence_threshold=1e-3),
             filter_seed=555, data_seed=111, savefig=True)

    for n_particles in {25, 50, 100}:
        for resampling_method in {ResamplingMethodsEnum.VARIANCE_CORRECTED, ResamplingMethodsEnum.REGULARIZED}:
            for eps in {0.25, 0.5, 0.75}:
                main(resampling_method, 0.5, T=150, n_particles=n_particles, batch_size=100,
                     resampling_kwargs=dict(epsilon=eps, scaling=0.9, convergence_threshold=1e-3),
                     filter_seed=555, data_seed=111, savefig=True)

    main(ResamplingMethodsEnum.KALMAN, 0.5, T=150, n_particles=25, batch_size=100,
         resampling_kwargs=dict(epsilon=0.5, scaling=0.75, convergence_threshold=1e-3),
         filter_seed=555, data_seed=111, savefig=True)
