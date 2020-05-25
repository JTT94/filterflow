import enum
import os
import sys
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pykalman
import seaborn
import tensorflow as tf
import tqdm
from mpl_toolkits import mplot3d

sys.path.append("../")
from filterflow.base import State
from filterflow.models.simple_linear_gaussian import make_filter
from filterflow.resampling import MultinomialResampler, SystematicResampler, StratifiedResampler, RegularisedTransform
from filterflow.resampling.criterion import NeverResample, AlwaysResample, NeffCriterion
from filterflow.resampling.differentiable import PartiallyCorrectedRegularizedTransform
from filterflow.resampling.differentiable.loss import SinkhornLoss
from filterflow.resampling.differentiable.optimized import OptimizedPointCloud
from filterflow.resampling.differentiable.optimizer.sgd import SGD

_ = mplot3d  # Importing this monkey patches matplotlib to allow for 3D plots


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


# DO NOT DECORATE
def get_surface(mesh, modifiable_transition_matrix, pf, initial_state, use_correction_term, observations_dataset, T,
                seed):
    likelihoods = tf.TensorArray(size=len(mesh), dtype=tf.float32, dynamic_size=False, element_shape=[])
    gradients = tf.TensorArray(size=len(mesh), dtype=tf.float32, dynamic_size=False, element_shape=[2])
    for i, val in enumerate(tqdm.tqdm(mesh)):
        tf_val = tf.constant(val)
        transition_matrix = tf.linalg.diag(tf_val)
        assign_op = modifiable_transition_matrix.assign(transition_matrix)
        with tf.control_dependencies([assign_op]):
            tf.random.set_seed(seed)
            # sadly this can only be done in eager mode for the time being
            # (will be corrected with stateless operations in next tf versions)
            ll, ll_grad = routine(pf, initial_state, use_correction_term, observations_dataset, T,
                                  modifiable_transition_matrix)
        likelihoods = likelihoods.write(tf.cast(i, tf.int32), ll)
        gradients = gradients.write(tf.cast(i, tf.int32), tf.linalg.diag_part(ll_grad))
    return likelihoods.stack(), gradients.stack()


# DO NOT DECORATE
def get_surface_finite_difference(mesh, modifiable_transition_matrix, pf, initial_state, use_correction_term,
                                  observations_dataset, T, seed, epsilon=1e-3):
    likelihoods = tf.TensorArray(size=len(mesh), dtype=tf.float32, dynamic_size=False, element_shape=[])
    gradients = tf.TensorArray(size=len(mesh), dtype=tf.float32, dynamic_size=False, element_shape=[2])
    for i, val in enumerate(tqdm.tqdm(mesh)):

        tf_val = tf.constant(val)
        transition_matrix = tf.linalg.diag(tf_val)
        assign_op = modifiable_transition_matrix.assign(transition_matrix)
        with tf.control_dependencies([assign_op]):
            tf.random.set_seed(seed)
            ll, ll_grad = routine(pf, initial_state, use_correction_term, observations_dataset, T,
                                  modifiable_transition_matrix)

        ll_eps_list = []
        for n_val in range(mesh.shape[1]):
            tf_val_eps = tf.constant([val[k] + (epsilon if k == n_val else 0.) for k in range(mesh.shape[1])],
                                     dtype=tf.float32)
            transition_matrix = tf.linalg.diag(tf_val_eps)
            assign_op = modifiable_transition_matrix.assign(transition_matrix)
            with tf.control_dependencies([assign_op]):
                tf.random.set_seed(seed)
                ll_eps, _ = routine(pf, initial_state, use_correction_term, observations_dataset, T,
                                    modifiable_transition_matrix)
                ll_eps_list.append((ll_eps - ll) / epsilon)

        likelihoods = likelihoods.write(tf.cast(i, tf.int32), ll)
        gradients = gradients.write(tf.cast(i, tf.int32), tf.convert_to_tensor(ll_eps_list, dtype=tf.float32))
    return likelihoods.stack(), gradients.stack()


def plot_surface(mesh, mesh_size, data, method_name, resampling_kwargs, savefig):
    seaborn.set()
    fig = plt.figure(figsize=(10, 10))

    x = mesh[:, 0].reshape([mesh_size, mesh_size])
    y = mesh[:, 1].reshape([mesh_size, mesh_size])

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.plot_surface(x, y, data.reshape([mesh_size, mesh_size]), cmap='viridis', edgecolor='none')
    fig.tight_layout()

    if savefig:
        filename = method_name + '_' + str(resampling_kwargs)
        fig.savefig(os.path.join('./charts/', f'surface_{filename}.png'))
    else:
        fig.suptitle(f'surface_{method_name}')
        plt.show()


def plot_vector_field(mesh, mesh_size, data, grad_data, method_name, resampling_kwargs, savefig):
    fig, ax = plt.subplots(figsize=(10, 10))

    x = mesh[:, 0].reshape([mesh_size, mesh_size])
    y = mesh[:, 1].reshape([mesh_size, mesh_size])

    contour = ax.contour(x, y, data.reshape([mesh_size, mesh_size]))
    ax.clabel(contour, inline=1, fontsize=10)
    ax.quiver(mesh[:, 0], mesh[:, 1], grad_data[:, 0], grad_data[:, 1])
    fig.tight_layout()
    if savefig:
        filename = method_name + '_' + str(resampling_kwargs)
        fig.savefig(os.path.join('./charts/', f'field_{filename}.png'))
    else:
        fig.suptitle(f'field_{method_name}')
        plt.show()


def main(resampling_method_value, resampling_neff, resampling_kwargs=None, T=100, batch_size=1, n_particles=25,
         data_seed=0, filter_seed=1, mesh_size=10, savefig=False):
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

    modifiable_transition_matrix = tf.Variable(transition_matrix, trainable=False)
    observation_matrix = tf.convert_to_tensor(observation_matrix)
    transition_covariance_chol = tf.linalg.cholesky(transition_covariance)
    observation_covariance_chol = tf.linalg.cholesky(observation_matrix)

    initial_particles = np_random_state.normal(0., 1., [batch_size, n_particles, 2]).astype(np.float32)
    initial_state = State(initial_particles)

    smc = make_filter(observation_matrix, modifiable_transition_matrix, observation_covariance_chol,
                      transition_covariance_chol,
                      resampling_method, resampling_criterion)

    x_linspace = np.linspace(0.25, 0.75, mesh_size).astype(np.float32)
    y_linspace = np.linspace(0.25, 0.75, mesh_size).astype(np.float32)
    mesh = np.asanyarray([(x, y) for x in x_linspace for y in y_linspace])

    if resampling_method.DIFFERENTIABLE:
        log_likelihoods, gradients = get_surface(mesh, modifiable_transition_matrix, smc, initial_state, False,
                                                 observation_dataset, T, filter_seed)
    else:
        log_likelihoods, gradients = get_surface_finite_difference(mesh, modifiable_transition_matrix, smc,
                                                                   initial_state, False, observation_dataset, T,
                                                                   filter_seed)

    plot_surface(mesh, mesh_size, log_likelihoods.numpy(), resampling_method_enum.name, resampling_kwargs, savefig)
    plot_vector_field(mesh, mesh_size, log_likelihoods.numpy(), gradients.numpy(), resampling_method_enum.name,
                      resampling_kwargs, savefig)


def fun_to_distribute(epsilon):
    main(ResamplingMethodsEnum.REGULARIZED, 0.5, T=150, mesh_size=20, savefig=True,
         resampling_kwargs=dict(epsilon=epsilon, scaling=0.5, convergence_threshold=1e-2))


if __name__ == '__main__':
    epsilons = [0.25, 0.5, 0.75, 1.]
    for epsilon in epsilons:
        print('Epsilon: {0}'.format(epsilon))
        fun_to_distribute(epsilon)
