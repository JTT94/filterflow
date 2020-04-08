import tensorflow as tf

from filterflow.resampling.differentiable.regularized_transport.utils import cost, softmin, diameter


# This is very much adapted from Feydy's geomloss work. Hopefully these should merge into one library...

@tf.function
def _simple_sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, epsilon, threshold, max_iter):
    epsilon = tf.ones([log_alpha.shape[0], 1]) * epsilon

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

    def stop_condition(i, _u, _v, update_size):
        n_iter_cond = i < max_iter - 1
        stable_cond = update_size > threshold
        return tf.reduce_all([n_iter_cond, stable_cond])

    def body(i, u, v, _update_size):
        new_u, new_v, new_update_size = apply_one(u, v)
        return i + 1, new_u, new_v, new_update_size

    n_iter = tf.constant(0)
    initial_update_size = 2 * threshold

    total_iter, converged_a_y, converged_b_x, last_update_size = tf.while_loop(stop_condition, body,
                                                                               loop_vars=[n_iter,
                                                                                          a_y_init,
                                                                                          b_x_init,
                                                                                          initial_update_size])

    # We do a last extrapolation for the gradient - leverages fixed point + implicit function theorem
    a_y, b_x, _ = apply_one(tf.stop_gradient(converged_a_y), tf.stop_gradient(converged_b_x))
    return a_y, b_x, total_iter


@tf.function
def sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy, epsilon, particles_diameter, scaling,
                  threshold, max_iter):
    # TODO: implement the symmetric version
    batch_size = log_alpha.shape[0]
    convergence_flag = tf.ones([batch_size], dtype=bool)
    epsilon_0 = particles_diameter ** 2
    scaling_factor = scaling ** 2

    a_y_init = softmin(epsilon_0, cost_yx, log_alpha)
    b_x_init = softmin(epsilon_0, cost_xy, log_beta)

    a_x_init = softmin(epsilon_0, cost_xx, log_alpha)
    b_y_init = softmin(epsilon_0, cost_yy, log_beta)

    def stop_condition(i, _a_y, _b_x, _a_x, _b_y, not_converged, running_epsilon):
        n_iter_cond = i < max_iter - 1
        epsilon_condition = running_epsilon > epsilon
        return tf.logical_and(n_iter_cond, tf.reduce_any([not_converged, epsilon_condition]))

    def apply_one(a_y, b_x, a_x, b_y, not_converged, running_epsilon):
        running_epsilon_ = tf.reshape(running_epsilon, [-1, 1])
        not_converged_ = tf.reshape(not_converged, [-1, 1])

        # TODO: Hopefully one day tensorflow controlflow will be lazy and not strict...
        at_y = tf.where(not_converged_, softmin(running_epsilon, cost_yx, log_alpha + b_x / running_epsilon_), a_y)
        bt_x = tf.where(not_converged_, softmin(running_epsilon, cost_xy, log_beta + a_y / running_epsilon_), b_x)

        at_x = tf.where(not_converged_, softmin(running_epsilon, cost_xx, log_alpha + b_x / running_epsilon_), a_x)
        bt_y = tf.where(not_converged_, softmin(running_epsilon, cost_yy, log_beta + a_y / running_epsilon_), b_y)

        a_y_new = .5 * (a_y + at_y)
        b_x_new = .5 * (b_x + bt_x)

        a_x_new = .5 * (a_x + at_x)
        b_y_new = .5 * (b_y + bt_y)

        a_y_diff = tf.reduce_max(tf.abs(a_y_new - a_y), 1)
        b_x_diff = tf.reduce_max(tf.abs(b_x_new - b_x), 1)

        is_stable = tf.logical_and(a_y_diff < threshold, b_x_diff < threshold)
        # Cross iterates converge slower than symmetric updates
        not_converged = tf.logical_not(is_stable)

        return a_y_new, b_x_new, a_x_new, b_y_new, not_converged

    def body(i, a_y, b_x, a_x, b_y, not_converged, running_epsilon):
        new_a_y, new_b_x, new_a_x, new_b_y, new_not_converged = apply_one(a_y, b_x, a_x, b_y, not_converged,
                                                                          running_epsilon)
        return i + 1, new_a_y, new_b_x, new_a_x, new_b_y, new_not_converged, tf.math.maximum(
            running_epsilon * scaling_factor, epsilon)

    n_iter = tf.constant(0)

    total_iter, converged_a_y, converged_b_x, converged_a_x, converged_b_y, _, final_epsilon = tf.while_loop(
        stop_condition,
        body,
        loop_vars=[n_iter,
                   tf.stop_gradient(a_y_init),
                   tf.stop_gradient(b_x_init),
                   tf.stop_gradient(a_x_init),
                   tf.stop_gradient(b_y_init),
                   convergence_flag,
                   epsilon_0])

    # We do a last extrapolation for the gradient - leverages fixed point + implicit function theorem
    final_a_y, final_b_x, final_a_x, final_b_y, _ = apply_one(tf.stop_gradient(converged_a_y),
                                                              tf.stop_gradient(converged_b_x),
                                                              tf.stop_gradient(converged_a_x),
                                                              tf.stop_gradient(converged_b_y),
                                                              convergence_flag,
                                                              final_epsilon)
    return final_a_y, final_b_x, final_a_x, final_b_y, total_iter


@tf.function
def sinkhorn_potentials(log_alpha, x, log_beta, y, epsilon, scaling, threshold, max_iter):
    cost_xy = cost(x, y)
    cost_yx = cost(y, x)
    cost_xx = cost(x, x)
    cost_yy = cost(y, x)
    diameter_ = tf.stop_gradient(diameter(x, y))
    a_y, b_x, a_x, b_y, total_iter = sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy, epsilon, diameter_,
                                         scaling, threshold, max_iter)

    return a_y, b_x, a_x, b_y, total_iter
