import os
import sys

sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import flags, app
from tensorflow_probability.python.internal.samplers import split_seed
from tqdm import tqdm

from filterflow.base import State
from filterflow.models.optimal_proposal_linear_gaussian import make_filter, make_optimal_filter
from filterflow.resampling import MultinomialResampler, SystematicResampler, StratifiedResampler, RegularisedTransform
from filterflow.resampling.criterion import NeverResample, AlwaysResample, NeffCriterion
from filterflow.resampling.differentiable import PartiallyCorrectedRegularizedTransform
from filterflow.resampling.differentiable.loss import SinkhornLoss
from filterflow.resampling.differentiable.optimized import OptimizedPointCloud
from filterflow.resampling.differentiable.optimizer.sgd import SGD

import pickle

from scripts.optimal_proposal_common import get_data, ResamplingMethodsEnum, get_observation_matrix, \
    get_observation_covariance, get_transition_covariance, get_transition_matrix

def pickle_obj(obj, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

@tf.function
def routine(pf, initial_state, observations_dataset, T, log_phi_x, phi_y, seed):
    with tf.GradientTape() as tape:
        tape.watch([log_phi_x, phi_y])
        final_state = pf(initial_state, observations_dataset, T, seed=seed, return_final=True)
        res = -tf.reduce_mean(final_state.log_likelihoods)
    return res, tape.gradient(res, [log_phi_x, phi_y]), tf.reduce_mean(final_state.ess)


def get_gradient_descent_function():
    # This is a trick because tensorflow doesn't allow you to create variables inside a decorated function
    @tf.function
    def gradient_descent(pf, initial_state, observations_dataset, T, n_iter, optimizer, log_phi_x, phi_y,
                         initial_values, change_seed, seed):
        variables = [log_phi_x, phi_y]
        reset_operations = [k.assign(v) for k, v in zip(variables, initial_values)]
        loss = tf.TensorArray(dtype=tf.float32, size=n_iter, dynamic_size=False)
        ess = tf.TensorArray(dtype=tf.float32, size=n_iter, dynamic_size=False)

        filter_seed, seed = split_seed(seed, n=2, salt='gradient_descent')

        with tf.control_dependencies(reset_operations):
            for i in tf.range(n_iter):

                loss_value, grads, average_ess = routine(pf, initial_state, observations_dataset, T, log_phi_x, phi_y,
                                                         seed)
                if change_seed:
                    filter_seed, seed = split_seed(filter_seed, n=2)
                loss = loss.write(tf.cast(i, tf.int32), loss_value)
                ess = ess.write(tf.cast(i, tf.int32), average_ess)
                grads = [tf.clip_by_value(grad, -100., 100.) for grad in grads]
                optimizer.apply_gradients(zip(grads, variables))
                tf.print('\rStep', i, '/', n_iter, end='')

        return [tf.convert_to_tensor(var) for var in variables], loss.stack(), ess.stack()

    return gradient_descent


def compare_learning_rates(pf, initial_state, observations_dataset, T, log_phi_x, phi_y, initial_values,
                           n_iter, optimizer_maker, learning_rates, filter_seed, use_xla, change_seed):
    loss_profiles = []
    ess_profiles = []
    for learning_rate in tqdm(learning_rates):
        optimizer = optimizer_maker(learning_rate=learning_rate)
        gradient_descent_function = get_gradient_descent_function()
        final_variables, loss_profile, ess_profile = gradient_descent_function(pf, initial_state, observations_dataset,
                                                                               T, n_iter,
                                                                               optimizer, log_phi_x, phi_y,
                                                                               initial_values, change_seed, filter_seed)
        loss_profiles.append(-loss_profile.numpy() / T)
        ess_profiles.append(ess_profile.numpy())
    return loss_profiles, ess_profiles



def plot_losses_vs_ess(loss_profiles_df, ess_profiles_df, filename, savefig, dx, dy, dense, T, n_particles, change_seed,
                       batch_size, optimal_filter_val, kalman_val, n_iter, mse_table, n_data):
    fig, ax = plt.subplots(figsize=(5, 3))
    loss_profiles_df.style.float_format = '${:,.1f}'.format
    loss_profiles_df.plot(ax=ax, legend=False)

    ax.axhline(y=optimal_filter_val, color="k", linestyle=':')
    ax.axhline(y=kalman_val, color="k")

    ax.set_xlim(0, n_iter)

    ax1 = ax.twinx()
    ess_profiles_df.plot.area(ax=ax1, legend=False, linestyle='--', alpha=0.33, stacked=False)

    # ax.set_ylim(-2.5, -1.7)
    ax1.set_ylim(1, n_particles)

    csv_fp = os.path.join('./charts/',
                                 f'global_variational_different_loss_df_lr_loss_{filename}_dx_{dx}_dy_{dy}_dense_{dense}_T_{T}_change_seed_{change_seed}.csv')
    loss_profiles_df.to_csv(csv_fp)
    
    csv_fp = os.path.join('./charts/',
                                 f'global_variational_different_ess_df_lr_loss_{filename}_dx_{dx}_dy_{dy}_dense_{dense}_T_{T}_change_seed_{change_seed}.csv')
    ess_profiles_df.to_csv(csv_fp)

    # ax.legend()
    fig.tight_layout()
    filename = f'global_variational_different_lr_loss_ess_{filename}_N_{n_particles}_dx_{dx}_dy_{dy}_dense_{dense}_T_{T}_change_seed_{change_seed}_batch_size_{batch_size}_ndata_{n_data}'
    if savefig:
        fig.savefig(os.path.join('./charts/',
                                 filename + '.png'))
        mse_table.to_csv(os.path.join('./tables/', filename + '.csv'),
                         float_format='%.5f')
    else:
        print(mse_table)
        fig.suptitle(f'variational_different_loss_ess_{filename}_dx_{dx}_dy_{dy}_dense_{dense}_T_{T}')
        plt.show()


def plot_variables(variables_df, filename, savefig):
    fig, ax = plt.subplots(figsize=(5, 5))
    variables_df.plot(ax=ax)
    fig.tight_layout()
    if savefig:
        fig.savefig(os.path.join('./charts/', f'global_variational_different_lr_variables_{filename}.png'))
    else:
        fig.suptitle(f'variational_different_lr_variables_{filename}')
        plt.show()


def resampling_method_factory(resampling_method_enum, resampling_kwargs):
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
    return resampling_method


def main(resampling_method_value, resampling_neff, learning_rates=(1e-4, 1e-3), resampling_kwargs=None,
         alpha=0.42, dx=10, dy=3, observation_covariance=1., dense=False, T=20, batch_size=1, n_particles=25,
         data_seed=0, n_data=50, n_iter=50, savefig=False, filter_seed=0, use_xla=False, change_seed=True):
    transition_matrix = get_transition_matrix(alpha, dx)
    transition_covariance = get_transition_covariance(dx)
    observation_matrix = get_observation_matrix(dx, dy, dense)
    observation_covariance = get_observation_covariance(observation_covariance, dy)

    resampling_method_enum = ResamplingMethodsEnum(resampling_method_value)

    np_random_state = np.random.RandomState(seed=data_seed)

    observation_matrix = tf.convert_to_tensor(observation_matrix)
    transition_covariance_chol = tf.linalg.cholesky(transition_covariance)
    observation_covariance_chol = tf.linalg.cholesky(observation_covariance)

    initial_particles = np_random_state.normal(0., 1., [batch_size, n_particles, dx]).astype(np.float32)
    initial_state = State(initial_particles)

    if resampling_neff == 0.:
        resampling_criterion = NeverResample()
    elif resampling_neff == 1.:
        resampling_criterion = AlwaysResample()
    else:
        resampling_criterion = NeffCriterion(resampling_neff, True)

    optimal_smc = make_optimal_filter(observation_matrix, transition_matrix, observation_covariance_chol,
                                      transition_covariance_chol, MultinomialResampler(), resampling_criterion)

    if resampling_kwargs is None:
        resampling_kwargs = {}

    resampling_method = resampling_method_factory(resampling_method_enum, resampling_kwargs)

    datas = []
    lls = []
    observation_datasets = []
    optimal_lls = []

    log_phi_x_0 = tf.ones(dx)
    phi_y_0 = tf.zeros(dy)

    for _ in range(n_data):
        data, ll = get_data(transition_matrix, observation_matrix, transition_covariance, observation_covariance, T,
                            np_random_state)
        datas.append(data)
        lls.append(ll / T)
        observation_dataset = tf.data.Dataset.from_tensor_slices(data)
        observation_datasets.append(observation_dataset)
        final_state = optimal_smc(initial_state, observation_dataset, T, None, True, filter_seed)
        optimal_lls.append(final_state.log_likelihoods.numpy().mean() / T)

    log_phi_x = tf.Variable(log_phi_x_0, trainable=True)
    phi_y = tf.Variable(phi_y_0, trainable=True)

    smc = make_filter(observation_matrix, transition_matrix, observation_covariance_chol,
                      transition_covariance_chol, resampling_method, resampling_criterion,
                      log_phi_x, phi_y)

    def optimizer_maker(learning_rate):
        # tf.function doesn't like creating variables. This is a way to create them outside the graph
        # We can't reuse the same optimizer because it would be giving a warmed-up momentum to the ones run later
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        return optimizer

    initial_values = [log_phi_x_0, phi_y_0]
    losses_list = []
    ess_profiles_list = []
    mean_errors = []
    for observation_dataset in observation_datasets:
        try:
            losses, ess_profiles = compare_learning_rates(smc, initial_state, observation_dataset, T, log_phi_x, phi_y,
                                                          initial_values, n_iter, optimizer_maker, learning_rates,
                                                          filter_seed,
                                                          use_xla, change_seed)
        except:
            print('one dataset failed, ignoring')
            continue
        losses_df = pd.DataFrame(np.stack(losses).T, columns=np.log10(learning_rates))
        ess_df = pd.DataFrame(np.stack(ess_profiles).T, columns=np.log10(learning_rates))

        losses_df.columns.name = 'log learning rate'
        losses_df.columns.epoch = 'epoch'

        ess_df.columns.name = 'log learning rate'
        ess_df.columns.epoch = 'epoch'

        losses_list.append(losses_df)
        ess_profiles_list.append(ess_df)

        delta_phi_m_1 = tf.linalg.diag(tf.exp(-log_phi_x))
        diff_cov = optimal_smc._proposal_model._sigma - delta_phi_m_1 @ transition_covariance
        approx_error = tf.linalg.diag_part(diff_cov).numpy()
        mean_error = np.sqrt(np.nanmean(approx_error ** 2))
        mean_errors.append(mean_error)

    losses_data = pd.concat(losses_list, axis=1)
    ess_data = pd.concat(ess_profiles_list, axis=1)

    mean_data = pd.DataFrame([[np.mean(mean_errors)]], index=pd.MultiIndex.from_tuples([(batch_size, n_particles)]),
                             columns=pd.MultiIndex.from_tuples([(resampling_method_enum.name, change_seed)]))

    losses_data = losses_data.groupby(axis=1, level=0).mean()
    ess_data = ess_data.groupby(axis=1, level=0).mean()

    # plot_losses(losses_df, resampling_method_enum.name, savefig, dx, dy, dense, T, change_seed)
    plot_losses_vs_ess(losses_data, ess_data, resampling_method_enum.name, savefig, dx, dy, dense, T, n_particles,
                       change_seed, batch_size, np.mean(optimal_lls), np.mean(lls), n_iter, mean_data, n_data)
    print(tf.exp(log_phi_x))


FLAGS = flags.FLAGS

flags.DEFINE_integer('resampling_method', ResamplingMethodsEnum.REGULARIZED, 'resampling_method')
flags.DEFINE_float('epsilon', 0.25, 'epsilon')
flags.DEFINE_float('resampling_neff', 0.5, 'resampling_neff')
flags.DEFINE_float('scaling', 0.9, 'scaling')
flags.DEFINE_float('log_learning_rate_min', np.log10(0.05), 'log_learning_rate_min')
flags.DEFINE_float('log_learning_rate_max', np.log10(0.05), 'log_learning_rate_max')
flags.DEFINE_integer('n_learning_rates', 1, 'log_learning_rate_max', lower_bound=1, upper_bound=1)
flags.DEFINE_integer('n_data', 10, 'n_data', lower_bound=1)
flags.DEFINE_boolean('change_seed', True, 'change seed between each gradient descent step')
flags.DEFINE_float('convergence_threshold', 1e-4, 'convergence_threshold')
flags.DEFINE_integer('n_particles', 25, 'n_particles', lower_bound=4)
flags.DEFINE_integer('batch_size', 4, 'batch_size', lower_bound=1)
flags.DEFINE_integer('n_iter', 100, 'n_iter', lower_bound=10)
flags.DEFINE_integer('max_iter', 500, 'max_iter', lower_bound=1)
flags.DEFINE_integer('dx', 25, 'dx', lower_bound=1)
flags.DEFINE_integer('dy', 1, 'dy', lower_bound=1)
flags.DEFINE_integer('T', 100, 'T', lower_bound=1)
flags.DEFINE_boolean('savefig', True, 'Save fig')
flags.DEFINE_boolean('use_xla', False, 'Use XLA (experimental)')
flags.DEFINE_boolean('dense', False, 'dense')
flags.DEFINE_integer('seed', 77, 'seed')


def flag_main(argb):
    for name, value in FLAGS.flag_values_dict().items():
        print(name, f'{repr(value)}')
    learning_rates = np.logspace(FLAGS.log_learning_rate_min, FLAGS.log_learning_rate_max, FLAGS.n_learning_rates,
                                 base=10).astype(np.float32)
    main(FLAGS.resampling_method,
         resampling_neff=FLAGS.resampling_neff,
         T=FLAGS.T,
         n_data=FLAGS.n_data,
         n_particles=FLAGS.n_particles,
         batch_size=FLAGS.batch_size,
         savefig=FLAGS.savefig,
         n_iter=FLAGS.n_iter,
         dx=FLAGS.dx,
         dy=FLAGS.dy,
         dense=FLAGS.dense,
         learning_rates=learning_rates,
         change_seed=FLAGS.change_seed,
         resampling_kwargs=dict(epsilon=FLAGS.epsilon,
                                scaling=FLAGS.scaling,
                                convergence_threshold=FLAGS.convergence_threshold,
                                max_iter=FLAGS.max_iter),
         filter_seed=FLAGS.seed,
         use_xla=FLAGS.use_xla)


if __name__ == '__main__':
    app.run(flag_main)
