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


