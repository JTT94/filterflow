import enum
import os
import sys

sys.path.append("./")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

from filterflow.models.seir import make_filter
from filterflow.resampling import MultinomialResampler, SystematicResampler, StratifiedResampler, RegularisedTransform
from filterflow.resampling.criterion import NeverResample, AlwaysResample, NeffCriterion
from filterflow.resampling.differentiable import PartiallyCorrectedRegularizedTransform
from filterflow.resampling.differentiable.loss import SinkhornLoss
from filterflow.resampling.differentiable.optimized import OptimizedPointCloud
from filterflow.resampling.differentiable.optimizer.sgd import SGD


def get_data():
    url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
    df = pd.read_csv(url)
    df.dateRep = pd.to_datetime(df.dateRep, dayfirst=True)
    df = df.loc[(df.geoId.isin(['UK'])) & (df.deaths > 0)].sort_values('dateRep')
    observations = df[df.geoId == 'UK']['deaths'].iloc[::-1].values.astype(np.float32)
    return observations


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
def values_and_gradient(start_values, gradient_variables, pf, initial_state,
                        observations_dataset, T, seed):
    assign_ops = [variable.assign(val) for variable, val in zip(gradient_variables, start_values)]

    with tf.control_dependencies([assign_ops]):
        tf.random.set_seed(seed)
        # sadly this can only be done in eager mode for the time being
        # (will be corrected with stateless operations in next tf versions)
        ll, ll_grad = routine(pf, initial_state, False, observations_dataset, T,
                              gradient_variables)
        # ll_grad list to tensor
        ll_grad_tensor = tf.stack(ll_grad)
    return -ll, -ll_grad_tensor


def gradient_descent(loss_fun, init_values, learning_rate, n_iter):
    loss = tf.TensorArray(dtype=tf.float32, size=n_iter + 1, dynamic_size=False)
    params = tf.TensorArray(dtype=tf.float32, size=n_iter + 1, dynamic_size=False)
    val = init_values
    for i in tqdm.trange(n_iter):
        loss_val, gradient_val = loss_fun(val)
        loss = loss.write(tf.cast(i, tf.int32), loss_val)
        val -= learning_rate * gradient_val
        params = params.write(tf.cast(n_iter, tf.int32), val)
        tf.print('Loss: ', loss_val, 'Gradient: ', gradient_val)
    loss_val, gradient_val = loss_fun(val)
    loss = loss.write(tf.cast(n_iter, tf.int32), loss_val)
    params = params.write(tf.cast(n_iter, tf.int32), val)

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


def main(make_filter, data, resampling_method_value, resampling_neff, resampling_kwargs=None, T=100, batch_size=1,
         n_particles=25,
         data_seed=0, filter_seed=1, learning_rate=0.02, n_iter=50, savefig=False):
    resampling_method_enum = ResamplingMethodsEnum(resampling_method_value)

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

    # init variables and filter
    initial_state, init_values, gradient_variables, smc = make_filter(model_kwargs,
                                                                      batch_size,
                                                                      n_particles,
                                                                      resampling_method,
                                                                      resampling_criterion)

    loss_fun = lambda x: values_and_gradient(x, gradient_variables, smc,
                                             initial_state, observation_dataset, T,
                                             filter_seed)

    final_value, loss = gradient_descent(loss_fun, init_values, learning_rate, n_iter)
    plot_loss(loss, final_value, resampling_method_enum.name, savefig)


if __name__ == '__main__':
    model_kwargs = {'alpha': 0.5,
                    'beta': 0.5,
                    'gamma': 0.15,
                    'delta': 0.001,
                    'log_sig': np.float32(np.log(0.1)),
                    'population_size': 66488991.0}
    dataset = get_data()
    main(make_filter=make_filter,
         data=dataset,
         resampling_method_value=ResamplingMethodsEnum.REGULARIZED,
         resampling_neff=0.5,
         T=len(dataset),
         n_particles=100,
         learning_rate=0.02,
         batch_size=50, resampling_kwargs={'epsilon': 0.5},
         savefig=True)
