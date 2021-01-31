import enum
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pykalman
import seaborn
import tensorflow as tf
from absl import flags, app
from mpl_toolkits import mplot3d
from tqdm import tqdm

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
    return sample[1].data.astype(np.float32), kf


class ResamplingMethodsEnum(enum.IntEnum):
    MULTINOMIAL = 0
    SYSTEMATIC = 1
    STRATIFIED = 2
    REGULARIZED = 3
    VARIANCE_CORRECTED = 4
    OPTIMIZED = 5
    KALMAN = 6


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
    return log_likelihood, tape.gradient(res_for_grad, gradient_variables), final_state.ess


# DO NOT DECORATE
def get_surface(mesh, modifiable_transition_matrix, pf, initial_state, use_correction_term, observations_dataset, T,
                seed, use_tqdm=False):
    likelihoods = tf.TensorArray(size=len(mesh), dtype=tf.float32, dynamic_size=False, element_shape=[])
    gradients = tf.TensorArray(size=len(mesh), dtype=tf.float32, dynamic_size=False, element_shape=[2])
    i = 0
    for val in mesh:
        matrix = tf.linalg.diag(val) + tf.constant([[0., 1.],
                                                    [0., 0.]], dtype=tf.float32)
        transition_matrix = tf.convert_to_tensor(matrix)
        assign_op = modifiable_transition_matrix.assign(transition_matrix)
        with tf.control_dependencies([assign_op]):
            # sadly this can only be done in eager mode for the time being
            # (will be corrected with stateless operations in next tf versions)
            ll, ll_grad, ess = routine(pf, initial_state, use_correction_term, observations_dataset, T,
                                       modifiable_transition_matrix, seed)
        likelihoods = likelihoods.write(tf.cast(i, tf.int32), ll)
        gradients = gradients.write(tf.cast(i, tf.int32), tf.linalg.diag_part(ll_grad))
        i += 1
        tf.print('\r', 'Step:', i, '/', mesh.shape[0], ", final ESS", ess, end='')
    return likelihoods.stack(), gradients.stack()


# DO NOT DECORATE
def get_surface_finite_difference(mesh, modifiable_transition_matrix, pf, initial_state, use_correction_term,
                                  observations_dataset, T, seed, use_tqdm=False, diff_epsilon=1e-2):
    likelihoods = tf.TensorArray(size=len(mesh), dtype=tf.float32, dynamic_size=False, element_shape=[])
    gradients = tf.TensorArray(size=len(mesh), dtype=tf.float32, dynamic_size=False, element_shape=[2])

    i = 0
    for val in mesh:
        matrix = tf.linalg.diag(val) + tf.constant([[0., 1.],
                                                    [0., 0.]], dtype=tf.float32)
        transition_matrix = tf.convert_to_tensor(matrix)
        # transition_matrix = tf.linalg.diag(tf_val)
        assign_op = modifiable_transition_matrix.assign(transition_matrix)
        with tf.control_dependencies([assign_op]):
            ll, ll_grad = routine(pf, initial_state, use_correction_term, observations_dataset, T,
                                  modifiable_transition_matrix, seed)

        ll_eps_list = tf.TensorArray(tf.float32, size=mesh.shape[1])
        for n_val in tf.range(mesh.shape[1]):
            tf_val_eps = tf.tensor_scatter_nd_add(val, [[n_val]], [diff_epsilon])

            transition_matrix = tf.linalg.diag(tf_val_eps) + tf.constant([[0., 1.],
                                                                          [0., 0.]], dtype=tf.float32)
            assign_op = modifiable_transition_matrix.assign(transition_matrix)
            with tf.control_dependencies([assign_op]):
                ll_eps, _ = routine(pf, initial_state, use_correction_term, observations_dataset, T,
                                    modifiable_transition_matrix, seed)
                ll_eps_list = ll_eps_list.write(tf.cast(n_val, dtype=tf.int32), (ll_eps - ll) / diff_epsilon)

        likelihoods = likelihoods.write(tf.cast(i, tf.int32), ll)
        gradients = gradients.write(tf.cast(i, tf.int32), tf.convert_to_tensor(ll_eps_list.stack(), dtype=tf.float32))
        i += 1
        tf.print('\r', 'Step:', i, '/', mesh.shape[0], end='')

    return likelihoods.stack(), gradients.stack()


def kf_loglikelihood(kf, np_obs):
    # There is an underlying bug in pykalman
    from scipy.linalg import solve_triangular as sc_solve
    from unittest import mock

    def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False,
                         overwrite_b=False, debug=None, check_finite=True):
        a = getattr(a, 'data', a)
        b = getattr(b, 'data', b)

        return sc_solve(a, b, trans, lower, unit_diagonal,
                        overwrite_b, debug, check_finite)

    with mock.patch('pykalman.utils.linalg.solve_triangular') as m:
        m.side_effect = solve_triangular
        return kf.loglikelihood(np_obs)


def get_surface_kf(mesh, kf, np_obs, epsilon, use_tqdm=False):
    likelihoods = np.empty(mesh.shape[0])
    gradients = np.empty(mesh.shape)

    n_params = mesh.shape[1]

    iterable = enumerate(mesh)
    if use_tqdm:
        iterable = tqdm(iterable, total=mesh.shape[0])

    for i, val in iterable:
        transition_matrix = np.diag(val) + np.eye(2, 2, 1, dtype=np.float32)
        kf.transition_matrices = transition_matrix

        ll = kf_loglikelihood(kf, np_obs)
        likelihoods[i] = ll
        ll_eps_list = np.empty(n_params)
        for n_val in range(n_params):
            eps_increment = np.array([epsilon * (1 if i == n_val else 0) for i in range(n_params)])
            val_eps = val + eps_increment
            transition_matrix = np.diag(val_eps) + np.eye(2, 2, 1, dtype=np.float32)
            kf.transition_matrices = transition_matrix
            ll_eps_list[n_val] = kf_loglikelihood(kf, np_obs)
        gradients[i] = (ll_eps_list - ll) / epsilon

    return likelihoods, gradients


def plot_surface(mesh, mesh_size, data, method_name, resampling_kwargs, n_particles, savefig):
    seaborn.set()
    fig = plt.figure(figsize=(10, 10))

    x = mesh[:, 0].reshape([mesh_size, mesh_size])
    y = mesh[:, 1].reshape([mesh_size, mesh_size])

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.plot_surface(x, y, data.reshape([mesh_size, mesh_size]), cmap='viridis', edgecolor='none')
    fig.tight_layout()

    if savefig:
        filename = "{0}_{1}_{2}_N_{3}".format(method_name,
                                              resampling_kwargs.get('epsilon'),
                                              resampling_kwargs.get('other'),
                                              n_particles)

        fig.savefig(os.path.join('./charts/', f'surface_{filename}.png'))
    else:
        fig.suptitle(f'surface_{method_name}')
        plt.show()


def plot_vector_field(mesh, mesh_size, data, grad_data, method_name, resampling_kwargs, n_particles, savefig):
    fig, ax = plt.subplots(figsize=(10, 10))

    x = mesh[:, 0].reshape([mesh_size, mesh_size])
    y = mesh[:, 1].reshape([mesh_size, mesh_size])

    contour = ax.contour(x, y, data.reshape([mesh_size, mesh_size]))
    ax.clabel(contour, inline=1, fontsize=10)
    ax.quiver(mesh[:, 0], mesh[:, 1], grad_data[:, 1], grad_data[:, 0])
    fig.tight_layout()
    if savefig:
        filename = "{0}_{1}_{2}_N_{3}".format(method_name,
                                              resampling_kwargs.get('epsilon'),
                                              resampling_kwargs.get('other'),
                                              n_particles)

        fig.savefig(os.path.join('./charts/', f'field_{filename}.png'))
    else:
        fig.suptitle(f'field_{method_name}')
        plt.show()


def kalman_main(kf, data, mesh, mesh_size, epsilon, use_tqdm, savefig):
    likelihoods, gradients = get_surface_kf(mesh, kf, data, epsilon, use_tqdm)
    plot_surface(mesh, mesh_size, likelihoods, 'kalman', {}, None, savefig)
    plot_vector_field(mesh, mesh_size, likelihoods, gradients, 'kalman', {}, None, savefig)


def main(resampling_method_value, resampling_neff, resampling_kwargs=None, T=100, batch_size=1, n_particles=25,
         data_seed=0, filter_seed=1, mesh_size=10, savefig=True, use_tqdm=False, use_xla=False, diff_epsilon=1e-1,
         optimal_proposal=False):

    v = 1.
    t = .1
    transition_matrix = np.array([[1., 1.],
                                  [0., 1.]],
                                 dtype=np.float32)

    transition_covariance = v ** 2 * np.array([[1 / 3, 1 / 2],
                                               [1 / 2, 1.]], dtype=np.float32)

    observation_matrix = np.array([[1., 0]], dtype=np.float32)
    observation_covariance = np.array([[t ** 2]], dtype=np.float32)

    resampling_method_enum = ResamplingMethodsEnum(resampling_method_value)

    x_linspace = np.linspace(0.95, 1., mesh_size).astype(np.float32)
    y_linspace = np.linspace(0.95, 1., mesh_size).astype(np.float32)
    mesh = np.asanyarray([(x, y) for x in x_linspace for y in y_linspace])

    np_random_state = np.random.RandomState(seed=data_seed)
    data, kf = get_data(transition_matrix, observation_matrix, transition_covariance, observation_covariance, T,
                        np_random_state)

    if resampling_method_enum == ResamplingMethodsEnum.KALMAN:
        return kalman_main(kf, data, mesh, mesh_size, 1e-2, use_tqdm, savefig)

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

    initial_particles = np_random_state.normal(0., .01, [batch_size, n_particles, 2]).astype(np.float32)
    initial_state = State(initial_particles)

    smc = make_filter(observation_matrix, modifiable_transition_matrix, observation_covariance_chol,
                      transition_covariance_chol,
                      resampling_method, resampling_criterion, optimal_proposal=optimal_proposal)

    # if resampling_method.DIFFERENTIABLE:
    get_method = tf.function(get_surface, experimental_compile=use_xla)
    # else:
    #     fun = partial(get_surface_finite_difference, diff_epsilon=diff_epsilon)
    #     get_method = tf.function(fun, experimental_compile=use_xla)

    log_likelihoods, gradients = get_method(mesh, modifiable_transition_matrix, smc,
                                            initial_state, False, observation_dataset, T,
                                            filter_seed, use_tqdm)

    plot_surface(mesh, mesh_size, log_likelihoods.numpy(), resampling_method_enum.name, resampling_kwargs, n_particles,
                 savefig)
    plot_vector_field(mesh, mesh_size, log_likelihoods.numpy(), gradients.numpy(), resampling_method_enum.name,
                      resampling_kwargs, n_particles, savefig)


# define flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('resampling_method', ResamplingMethodsEnum.REGULARIZED, 'resampling_method')
flags.DEFINE_float('epsilon', 0.25, 'epsilon')
flags.DEFINE_float('resampling_neff', 0.9999, 'resampling_neff')
flags.DEFINE_float('scaling', 0.85, 'scaling')
flags.DEFINE_float('convergence_threshold', 1e-6, 'convergence_threshold')
flags.DEFINE_float('diff_epsilon', 1e-2, 'epsilon for finite diff')
flags.DEFINE_integer('n_particles', 50, 'n_particles', lower_bound=4)
flags.DEFINE_integer('batch_size', 1, 'batch_size', lower_bound=1)
flags.DEFINE_integer('max_iter', 500, 'max_iter', lower_bound=1)
flags.DEFINE_integer('T', 100, 'T', lower_bound=1)
flags.DEFINE_integer('mesh_size', 20, 'mesh_size', lower_bound=1)
flags.DEFINE_boolean('savefig', True, 'Save fig')
flags.DEFINE_integer('seed', 1234, 'seed')
flags.DEFINE_integer('data_seed', 123, 'data_seed')
flags.DEFINE_boolean('use_xla', False, 'Use XLA (experimental)')
flags.DEFINE_boolean('optimal_proposal', True, 'Use locally optimal proposals')


def flag_main(argb):
    print('resampling_method: {0}'.format(ResamplingMethodsEnum(FLAGS.resampling_method).name))
    print('batch_size: {0}'.format(FLAGS.batch_size))
    print('epsilon: {0}'.format(FLAGS.epsilon))
    print('resampling_neff: {0}'.format(FLAGS.resampling_neff))
    print('convergence_threshold: {0}'.format(FLAGS.convergence_threshold))
    print('n_particles: {0}'.format(FLAGS.n_particles))
    print('T: {0}'.format(FLAGS.T))
    print('mesh_size: {0}'.format(FLAGS.mesh_size))
    print('savefig: {0}'.format(FLAGS.savefig))
    print('scaling: {0}'.format(FLAGS.scaling))
    print('seed: {0}'.format(FLAGS.seed))
    print('data_seed: {0}'.format(FLAGS.data_seed))
    print('max_iter: {0}'.format(FLAGS.max_iter))
    print('use_xla: {0}'.format(FLAGS.use_xla))
    print('diff_epsilon: {0}'.format(FLAGS.diff_epsilon))
    print('optimal_proposal: {0}'.format(FLAGS.optimal_proposal))
    with tf.device("/GPU:0"):
        main(FLAGS.resampling_method,
             batch_size=FLAGS.batch_size,
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
         data_seed=FLAGS.data_seed,
         use_xla=FLAGS.use_xla,
         diff_epsilon=FLAGS.diff_epsilon,
         optimal_proposal=FLAGS.optimal_proposal)


if __name__ == '__main__':
    app.run(flag_main)
