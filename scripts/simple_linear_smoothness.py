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
from absl import flags, app

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
def routine(pf, initial_state, resampling_correction, observations_dataset, T, gradient_variables, seed=None):
    with tf.GradientTape() as tape:
        tape.watch(gradient_variables)
        final_state = pf(initial_state, observations_dataset, n_observations=T, return_final=True, seed=seed)
        log_likelihood = tf.reduce_mean(final_state.log_likelihoods)
        correction_term = tf.reduce_mean(final_state.resampling_correction)
        if resampling_correction:
            res_for_grad = log_likelihood + correction_term
        else:
            res_for_grad = log_likelihood
    return log_likelihood, tape.gradient(res_for_grad, gradient_variables)


# DO NOT DECORATE
def get_surface(mesh, modifiable_transition_matrix, pf, initial_state, use_correction_term, observations_dataset, T,
                seed, use_tqdm=False):
    likelihoods = tf.TensorArray(size=len(mesh), dtype=tf.float32, dynamic_size=False, element_shape=[])
    gradients = tf.TensorArray(size=len(mesh), dtype=tf.float32, dynamic_size=False, element_shape=[2])
    i = 0
    for val in mesh:
        tf_val = tf.convert_to_tensor(val)
        transition_matrix = tf.linalg.diag(tf_val)
        assign_op = modifiable_transition_matrix.assign(transition_matrix)
        with tf.control_dependencies([assign_op]):
            # sadly this can only be done in eager mode for the time being
            # (will be corrected with stateless operations in next tf versions)
            ll, ll_grad = routine(pf, initial_state, use_correction_term, observations_dataset, T,
                                  modifiable_transition_matrix, seed)
        likelihoods = likelihoods.write(tf.cast(i, tf.int32), ll)
        gradients = gradients.write(tf.cast(i, tf.int32), tf.linalg.diag_part(ll_grad))
        i += 1
        tf.print('\r', 'Step:', i, '/', mesh.shape[0], end='')
    return likelihoods.stack(), gradients.stack()


# DO NOT DECORATE
def get_surface_finite_difference(mesh, modifiable_transition_matrix, pf, initial_state, use_correction_term,
                                  observations_dataset, T, seed, epsilon=1e-2, use_tqdm=False):
    likelihoods = tf.TensorArray(size=len(mesh), dtype=tf.float32, dynamic_size=False, element_shape=[])
    gradients = tf.TensorArray(size=len(mesh), dtype=tf.float32, dynamic_size=False, element_shape=[2])

    i = 0
    for val in mesh:
        tf_val = tf.convert_to_tensor(val)
        transition_matrix = tf.linalg.diag(tf_val)
        assign_op = modifiable_transition_matrix.assign(transition_matrix)
        with tf.control_dependencies([assign_op]):
            ll, ll_grad = routine(pf, initial_state, use_correction_term, observations_dataset, T,
                                  modifiable_transition_matrix, seed)

        ll_eps_list = tf.TensorArray(tf.float32, size=mesh.shape[1])
        for n_val in tf.range(mesh.shape[1]):
            tf_val_eps = tf.tensor_scatter_nd_add(val, [[n_val]], [epsilon])

            transition_matrix = tf.linalg.diag(tf_val_eps)
            assign_op = modifiable_transition_matrix.assign(transition_matrix)
            with tf.control_dependencies([assign_op]):
                ll_eps, _ = routine(pf, initial_state, use_correction_term, observations_dataset, T,
                                    modifiable_transition_matrix, seed)
                ll_eps_list = ll_eps_list.write(tf.cast(n_val, dtype=tf.int32), (ll_eps - ll) / epsilon)

        likelihoods = likelihoods.write(tf.cast(i, tf.int32), ll)
        gradients = gradients.write(tf.cast(i, tf.int32), tf.convert_to_tensor(ll_eps_list.stack(), dtype=tf.float32))
        i += 1
        tf.print('\r', 'Step:', i, '/', mesh.shape[0], end='')

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
        filename = method_name + '_' + str(resampling_kwargs.get('epsilon', 'other'))
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
        filename = method_name + '_' + str(resampling_kwargs.get('epsilon', 'other'))
        fig.savefig(os.path.join('./charts/', f'field_{filename}.png'))
    else:
        fig.suptitle(f'field_{method_name}')
        plt.show()


def main(resampling_method_value, resampling_neff, resampling_kwargs=None, T=100, batch_size=1, n_particles=25,
         data_seed=0, filter_seed=1, mesh_size=10, savefig=True, use_tqdm=False, use_xla=False):
    transition_matrix = 0.5 * np.eye(2, dtype=np.float32)
    transition_covariance = 0.5 * np.eye(2, dtype=np.float32)
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
    observation_covariance_chol = tf.linalg.cholesky(observation_covariance)

    initial_particles = np_random_state.normal(0., 1., [batch_size, n_particles, 2]).astype(np.float32)
    initial_state = State(initial_particles)

    smc = make_filter(observation_matrix, modifiable_transition_matrix, observation_covariance_chol,
                      transition_covariance_chol,
                      resampling_method, resampling_criterion)

    x_linspace = np.linspace(0.25, 0.75, mesh_size).astype(np.float32)
    y_linspace = np.linspace(0.25, 0.75, mesh_size).astype(np.float32)
    mesh = np.asanyarray([(x, y) for x in x_linspace for y in y_linspace])

    if resampling_method.DIFFERENTIABLE:
        get_method = tf.function(get_surface, experimental_compile=use_xla)
    else:
        get_method = tf.function(get_surface, experimental_compile=use_xla)

    log_likelihoods, gradients = get_method(mesh, modifiable_transition_matrix, smc,
                                            initial_state, False, observation_dataset, T,
                                            filter_seed, use_tqdm)

    plot_surface(mesh, mesh_size, log_likelihoods.numpy(), resampling_method_enum.name, resampling_kwargs, savefig)
    plot_vector_field(mesh, mesh_size, log_likelihoods.numpy(), gradients.numpy(), resampling_method_enum.name,
                      resampling_kwargs, savefig)


def fun_to_distribute(epsilon):
    main(ResamplingMethodsEnum.REGULARIZED, resampling_neff=0.5, T=100, mesh_size=5,
         resampling_kwargs=dict(epsilon=epsilon, scaling=0.75, convergence_threshold=1e-2), savefig=True,
         use_tqdm=False)


# define flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('resampling_method', ResamplingMethodsEnum.REGULARIZED, 'resampling_method')
flags.DEFINE_float('epsilon', 0.5, 'epsilon')
flags.DEFINE_float('resampling_neff', 0.5, 'resampling_neff')
flags.DEFINE_float('scaling', 0.75, 'scaling')
flags.DEFINE_float('convergence_threshold', 1e-3, 'convergence_threshold')
flags.DEFINE_integer('n_particles', 25, 'n_particles', lower_bound=4)
flags.DEFINE_integer('max_iter', 500, 'max_iter', lower_bound=1)
flags.DEFINE_integer('T', 150, 'T', lower_bound=1)
flags.DEFINE_integer('mesh_size', 10, 'mesh_size', lower_bound=1)
flags.DEFINE_boolean('savefig', False, 'Save fig')
flags.DEFINE_integer('seed', 25, 'seed')
flags.DEFINE_boolean('use_xla', False, 'Use XLA (experimental)')


def flag_main(argb):
    print('resampling_method: {0}'.format(ResamplingMethodsEnum(FLAGS.resampling_method).name))
    print('epsilon: {0}'.format(FLAGS.epsilon))
    print('resampling_neff: {0}'.format(FLAGS.resampling_neff))
    print('convergence_threshold: {0}'.format(FLAGS.convergence_threshold))
    print('n_particles: {0}'.format(FLAGS.n_particles))
    print('T: {0}'.format(FLAGS.T))
    print('mesh_size: {0}'.format(FLAGS.mesh_size))
    print('savefig: {0}'.format(FLAGS.savefig))
    print('scaling: {0}'.format(FLAGS.scaling))
    print('max_iter: {0}'.format(FLAGS.max_iter))
    print('use_xla: {0}'.format(FLAGS.use_xla))

    main(FLAGS.resampling_method,
         resampling_neff=FLAGS.resampling_neff,
         T=FLAGS.T,
         mesh_size=FLAGS.mesh_size,
         n_particles=FLAGS.n_particles,
         savefig=FLAGS.savefig,
         use_tqdm=True,
         resampling_kwargs=dict(epsilon=FLAGS.epsilon,
                                scaling=FLAGS.scaling,
                                convergence_threshold=FLAGS.convergence_threshold,
                                max_iter=FLAGS.max_iter),
         filter_seed=FLAGS.seed,
         use_xla=FLAGS.use_xla)


if __name__ == '__main__':
    app.run(flag_main)
