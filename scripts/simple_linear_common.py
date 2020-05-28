import enum

import pykalman
import numpy as np


class ResamplingMethodsEnum(enum.IntEnum):
    MULTINOMIAL = 0
    SYSTEMATIC = 1
    STRATIFIED = 2
    REGULARIZED = 3
    VARIANCE_CORRECTED = 4
    OPTIMIZED = 5
    KALMAN = 6


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


def get_data(transition_matrix, observation_matrix, transition_covariance, observation_covariance, T=100,
             random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    kf = pykalman.KalmanFilter(transition_matrix, observation_matrix, transition_covariance, observation_covariance)
    sample = kf.sample(T, random_state=random_state)
    return sample[1].data.astype(np.float32), kf
