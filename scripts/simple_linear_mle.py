import os

from absl import flags, app
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow_probability.python.internal.samplers import split_seed

from filterflow.base import State
from filterflow.models.simple_linear_gaussian import make_filter
from filterflow.resampling import MultinomialResampler, SystematicResampler, StratifiedResampler, RegularisedTransform
from filterflow.resampling.criterion import NeverResample, AlwaysResample, NeffCriterion
from filterflow.resampling.differentiable import PartiallyCorrectedRegularizedTransform
from filterflow.resampling.differentiable.loss import SinkhornLoss
from filterflow.resampling.differentiable.optimized import OptimizedPointCloud
from filterflow.resampling.differentiable.optimizer.sgd import SGD
from scripts.simple_linear_common import get_data, kf_loglikelihood, ResamplingMethodsEnum


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
def gradient_descent(loss_fun, x0, observation_dataset, learning_rate, n_iter, filter_seed, change_seed):
    loss = tf.TensorArray(dtype=tf.float32, size=n_iter + 1, dynamic_size=False)
    val = tf.identity(x0)
    filter_seed, seed = split_seed(filter_seed, n=2, salt='gradient_descent')
    for i in tf.range(n_iter):
        loss_val, gradient_val = loss_fun(val, observation_dataset, seed=seed)
        if change_seed:
            filter_seed, seed = split_seed(filter_seed, n=2)
        loss = loss.write(tf.cast(i, tf.int32), loss_val)
        val -= learning_rate * gradient_val
        tf.print('\rStep ', i + 1, '/', n_iter, end='')
    loss_val, gradient_val = loss_fun(val, observation_dataset, seed)
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


def main(resampling_method_value, resampling_neff, resampling_kwargs=None,
         T=100, batch_size=1, n_particles=25,
         data_seed=0, filter_seed=1, learning_rate=0.001, n_iter=50,
         savefig=False, use_xla=False, batch_data=1, assume_differentiable=False, change_seed=False):
    transition_matrix = 0.5 * np.eye(2, dtype=np.float32)
    transition_covariance = np.eye(2, dtype=np.float32)
    observation_matrix = np.eye(2, dtype=np.float32)
    observation_covariance = 0.1 * np.eye(2, dtype=np.float32)

    resampling_method_enum = ResamplingMethodsEnum(resampling_method_value)

    np_random_state = np.random.RandomState(seed=data_seed)
    data = []
    np_data = []

    assert batch_data > 0
    for _ in range(batch_data):
        a_data, kf = get_data(transition_matrix, observation_matrix, transition_covariance, observation_covariance, T,
                              np_random_state)
        data.append(tf.data.Dataset.from_tensor_slices(a_data))
        np_data.append(a_data)

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

    if resampling_method.DIFFERENTIABLE or assume_differentiable:
        loss_fun = lambda x, observation_dataset, seed: values_and_gradient(x, modifiable_transition_matrix, smc,
                                                                            initial_state, observation_dataset, T,
                                                                            seed)
    else:
        loss_fun = lambda x, observation_dataset, seed: values_and_gradient_finite_diff(x, modifiable_transition_matrix,
                                                                                        smc,
                                                                                        initial_state,
                                                                                        observation_dataset, T,
                                                                                        seed)

    final_values = []
    losses = []
    kalman_params = []

    def kf_likelihood_fun(val, data):
        import copy
        kf_copy = copy.copy(kf)
        kf_copy.transition_matrices = np.diag(val)
        return -kf_loglikelihood(kf_copy, data)

    fun = tf.function(loss_fun, experimental_compile=use_xla)

    for observation_dataset, np_dataset in zip(data, np_data):
        final_value, loss = gradient_descent(fun, x0, observation_dataset, tf.constant(learning_rate),
                                             tf.constant(n_iter), tf.constant(filter_seed), tf.constant(change_seed))
        final_values.append(final_value.numpy())
        losses.append(loss.numpy())
        kf_params = minimize(kf_likelihood_fun, x0.numpy(), args=(np_dataset,))
        kalman_params.append(kf_params.x)

    final_values = np.vstack(final_values)
    kalman_params = np.vstack(kalman_params)

    plt.hist(final_values - kalman_params, bins=10)
    plt.show()
    plot_loss(loss, final_value, resampling_method_enum.name, savefig)


# define flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('resampling_method', ResamplingMethodsEnum.MULTINOMIAL, 'resampling_method')
flags.DEFINE_float('epsilon', 0.25, 'epsilon')
flags.DEFINE_float('resampling_neff', 0.5, 'resampling_neff')
flags.DEFINE_float('scaling', 0.75, 'scaling')
flags.DEFINE_float('learning_rate', 1e-4, 'learning_rate', upper_bound=1e-1)
flags.DEFINE_float('convergence_threshold', 1e-3, 'convergence_threshold')
flags.DEFINE_integer('n_particles', 25, 'n_particles', lower_bound=4)
flags.DEFINE_integer('batch_size', 4, 'batch_size', lower_bound=1)
flags.DEFINE_integer('n_iter', 50, 'n_iter', lower_bound=10)
flags.DEFINE_integer('max_iter', 500, 'max_iter', lower_bound=1)
flags.DEFINE_integer('T', 150, 'T', lower_bound=1)
flags.DEFINE_boolean('savefig', False, 'Save fig')
flags.DEFINE_integer('batch_data', 10, 'Data samples', lower_bound=1)
flags.DEFINE_boolean('use_xla', False, 'Use XLA (experimental)')
flags.DEFINE_boolean('assume_differentiable', False, 'Assume that all schemes are differentiable')
flags.DEFINE_boolean('change_seed', False, 'change seed between each gradient descent step')
flags.DEFINE_integer('seed', 25, 'seed')


def flag_main(argb):
    print('resampling_method: {0}'.format(ResamplingMethodsEnum(FLAGS.resampling_method).name))
    print('assume_differentiable: {0}'.format(FLAGS.assume_differentiable))
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
    print('batch_data: {0}'.format(FLAGS.batch_data))
    print('change_seed: {0}'.format(FLAGS.change_seed))

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
         use_xla=FLAGS.use_xla,
         batch_data=FLAGS.batch_data,
         assume_differentiable=FLAGS.assume_differentiable,
         change_seed=FLAGS.change_seed)


if __name__ == '__main__':
    app.run(flag_main)
