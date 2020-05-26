import enum
import os

import matplotlib.pyplot as plt
import numpy as np
import pykalman
import tensorflow as tf
from absl import flags, app

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
def routine(pf, initial_state, resampling_correction, observations_dataset, T, gradient_variables, seed):
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
@tf.function
def values_and_gradient(x, modifiable_transition_matrix, pf, initial_state,
                        observations_dataset, T, seed):
    tf_val = tf.convert_to_tensor(x)
    transition_matrix = tf.linalg.diag(tf_val)
    assign_op = modifiable_transition_matrix.assign(transition_matrix)
    with tf.control_dependencies([assign_op]):
        # sadly this can only be done in eager mode for the time being
        # (will be corrected with stateless operations in next tf versions)
        ll, ll_grad = routine(pf, initial_state, False, observations_dataset, T,
                              modifiable_transition_matrix, seed)
    return -ll, -tf.linalg.diag_part(ll_grad)


# DO NOT DECORATE
@tf.function
def values_and_gradient_finite_diff(x, modifiable_transition_matrix, pf, initial_state, observations_dataset, T, seed,
                                    epsilon=1e-2):
    tf_val = tf.convert_to_tensor(x)
    transition_matrix = tf.linalg.diag(tf_val)
    assign_op = modifiable_transition_matrix.assign(transition_matrix)
    with tf.control_dependencies([assign_op]):
        ll, _ = routine(pf, initial_state, False, observations_dataset, T, modifiable_transition_matrix, seed=seed)

    ll_eps_list = tf.TensorArray(dtype=tf.float32, size=x.shape[0])

    for n_val in tf.range(x.shape[0]):
        tf_val_eps = tf.tensor_scatter_nd_add(tf_val, [[n_val]], [epsilon])

        transition_matrix = tf.linalg.diag(tf_val_eps)
        assign_op = modifiable_transition_matrix.assign(transition_matrix)
        with tf.control_dependencies([assign_op]):
            ll_eps, _ = routine(pf, initial_state, False, observations_dataset, T,
                                modifiable_transition_matrix, seed=seed)
            ll_eps_list = ll_eps_list.write(tf.cast(n_val, tf.int32), (ll_eps - ll) / epsilon)

    return -ll, -ll_eps_list.stack()


@tf.function
def gradient_descent(loss_fun, x0, learning_rate, n_iter):
    loss = tf.TensorArray(dtype=tf.float32, size=n_iter + 1, dynamic_size=False)
    val = tf.identity(x0)
    for i in tf.range(n_iter):
        loss_val, gradient_val = loss_fun(val)
        loss = loss.write(tf.cast(i, tf.int32), loss_val)
        val -= learning_rate * gradient_val
        tf.print('\rStep ', i + 1, '/', n_iter, end='')
    loss_val, gradient_val = loss_fun(val)
    loss = loss.write(tf.cast(n_iter, tf.int32), loss_val)
    return val, loss.stack()


def plot_loss(data, final_val, filename, savefig):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data)
    ax.set_xlabel('epoch')
    ax.set_ylabel('-log-likelihood')
    ax.text(0.75, 0.75, f'final value {final_val}', transform=ax.transAxes)
    fig.tight_layout()
    if savefig:
        fig.savefig(os.path.join('./charts/', f'mle_{filename}.png'))
    else:
        fig.suptitle(f'mle_{filename}')
        plt.show()


def main(resampling_method_value, resampling_neff, resampling_kwargs=None, T=100, batch_size=1, n_particles=25,
         data_seed=0, filter_seed=1, learning_rate=0.001, n_iter=50, savefig=False, use_xla=False):
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
    observation_covariance_chol = tf.linalg.cholesky(observation_covariance)

    initial_particles = np_random_state.normal(0., 1., [batch_size, n_particles, 2]).astype(np.float32)
    initial_state = State(initial_particles)

    smc = make_filter(observation_matrix, modifiable_transition_matrix, observation_covariance_chol,
                      transition_covariance_chol,
                      resampling_method, resampling_criterion)

    x0 = tf.constant([0.25, 0.25])

    if resampling_method.DIFFERENTIABLE:
        loss_fun = lambda x: values_and_gradient(x, modifiable_transition_matrix, smc,
                                                 initial_state, observation_dataset, T,
                                                 filter_seed)
    else:
        loss_fun = lambda x: values_and_gradient_finite_diff(x, modifiable_transition_matrix, smc,
                                                             initial_state, observation_dataset, T,
                                                             filter_seed)

    final_value, loss = gradient_descent(tf.function(loss_fun, experimental_compile=use_xla), x0, learning_rate, n_iter)
    plot_loss(loss, final_value, resampling_method_enum.name, savefig)


# define flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('resampling_method', ResamplingMethodsEnum.REGULARIZED, 'resampling_method')
flags.DEFINE_float('epsilon', 0.25, 'epsilon')
flags.DEFINE_float('resampling_neff', 0.5, 'resampling_neff')
flags.DEFINE_float('scaling', 0.75, 'scaling')
flags.DEFINE_float('learning_rate', 1e-4, 'learning_rate', upper_bound=1e-1)
flags.DEFINE_float('convergence_threshold', 1e-3, 'convergence_threshold')
flags.DEFINE_integer('n_particles', 25, 'n_particles', lower_bound=4)
flags.DEFINE_integer('batch_size', 1, 'batch_size', lower_bound=1)
flags.DEFINE_integer('n_iter', 50, 'n_iter', lower_bound=10)
flags.DEFINE_integer('max_iter', 500, 'max_iter', lower_bound=1)
flags.DEFINE_integer('T', 150, 'T', lower_bound=1)
flags.DEFINE_boolean('savefig', False, 'Save fig')
flags.DEFINE_boolean('use_xla', False, 'Use XLA (experimental)')
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
    print('learning_rate: {0}'.format(FLAGS.learning_rate))
    print('use_xla: {0}'.format(FLAGS.use_xla))

    main(FLAGS.resampling_method,
         resampling_neff=FLAGS.resampling_neff,
         T=FLAGS.T,
         n_particles=FLAGS.n_particles,
         batch_size=FLAGS.batch_size,
         savefig=FLAGS.savefig,
         learning_rate=FLAGS.learning_rate,
         n_iter=FLAGS.n_iter,
         resampling_kwargs=dict(epsilon=FLAGS.epsilon,
                                scaling=FLAGS.scaling,
                                convergence_threshold=FLAGS.convergence_threshold,
                                max_iter=FLAGS.max_iter),
         filter_seed=FLAGS.seed,
         use_xla=FLAGS.use_xla)


if __name__ == '__main__':
    app.run(flag_main)

