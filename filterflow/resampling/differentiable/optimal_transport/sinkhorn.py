import tensorflow as tf

from filterflow.resampling.differentiable.optimal_transport.utils import squared_distances, softmin

MACHINE_PRECISION = 1e-10


@tf.function
def sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, epsilon, threshold, max_iter):
    # initialisation
    a_y_init = softmin(epsilon, cost_yx, log_alpha)
    b_x_init = softmin(epsilon, cost_xy, log_beta)

    def apply_one(a_y, b_x):
        at_y = softmin(epsilon, cost_yx, log_alpha + b_x / epsilon)
        bt_x = softmin(epsilon, cost_xy, log_beta + a_y / epsilon)

        a_y_new = .5 * (a_y + at_y)
        b_x_new = .5 * (b_x + bt_x)

        a_y_diff = tf.reduce_max(tf.abs(a_y_new - a_y))
        b_x_diff = tf.reduce_max(tf.abs(b_x_new - b_x))

        return a_y_new, b_x_new, tf.maximum(a_y_diff, b_x_diff)

    def stop_condition(i, u, v, update_size):
        n_iter_cond = i < max_iter - 1
        stable_cond = update_size > threshold
        precision_cond = tf.logical_and(tf.reduce_min(u) > MACHINE_PRECISION,
                                        tf.reduce_min(v) > MACHINE_PRECISION)
        return tf.reduce_all([n_iter_cond, stable_cond, precision_cond])

    def body(i, u, v, _update_size):
        new_u, new_v, new_update_size = apply_one(u, v)
        return i + 1, new_u, new_v, new_update_size

    n_iter = tf.constant(0)
    initial_update_size = 2 * threshold
    _total_iter, converged_a_y, converged_b_x, last_update_size = tf.while_loop(stop_condition, body,
                                                                                loop_vars=[n_iter,
                                                                                           a_y_init,
                                                                                           b_x_init,
                                                                                           initial_update_size])

    # We do a last extrapolation for the gradient - leverages fixed point + implicit function theorem
    a_y = softmin(epsilon, cost_yx, log_alpha + tf.stop_gradient(converged_b_x) / epsilon)
    b_x = softmin(epsilon, cost_xy, log_beta + tf.stop_gradient(converged_a_y) / epsilon)

    return a_y, b_x


@tf.function
def sinkhorn_potentials(log_alpha, x, log_beta, y, epsilon, threshold, max_iter):
    cost_xy = 0.5 * squared_distances(x, tf.stop_gradient(y))
    cost_yx = 0.5 * squared_distances(y, tf.stop_gradient(x))
    return sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, epsilon, threshold, max_iter)
