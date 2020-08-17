import enum

import numpy as np
import pykalman

from scripts.base import kf_loglikelihood


def get_data(transition_matrix, observation_matrix, transition_covariance, observation_covariance, T=100,
             random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    kf = pykalman.KalmanFilter(transition_matrix, observation_matrix, transition_covariance, observation_covariance)
    sample = kf.sample(T, random_state=random_state)
    data = sample[1].data.astype(np.float32)

    return data.reshape(T, 1, 1, -1), kf_loglikelihood(kf, data)


def get_observation_matrix(dx, dy, dense=False, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    if dense:
        return random_state.normal(0., 1., [dy, dx]).astype(np.float32)
    return np.eye(dy, dx, dtype=np.float32)


def get_transition_matrix(alpha, dx):
    arange = np.arange(1, dx + 1, dtype=np.float32)
    return alpha ** (np.abs(arange[None, :] - arange[:, None]) + 1)


def get_transition_covariance(dx):
    return np.eye(dx, dtype=np.float32)


def get_observation_covariance(r, dy):
    return r * np.eye(dy, dtype=np.float32)


class ResamplingMethodsEnum(enum.IntEnum):
    MULTINOMIAL = 0
    SYSTEMATIC = 1
    STRATIFIED = 2
    REGULARIZED = 3
    VARIANCE_CORRECTED = 4
    OPTIMIZED = 5
