import tensorflow as tf

from filterflow.resampling.differentiable.regularized_transport.utils import cost, softmin, max_min


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
        return tf.logical_and(n_iter_cond, tf.reduce_all(stable_cond))

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
    batch_size = log_alpha.shape[0]
    continue_flag = tf.ones([batch_size], dtype=bool)
    epsilon_0 = particles_diameter ** 2
    scaling_factor = scaling ** 2

    a_y_init = softmin(epsilon_0, cost_yx, log_alpha)
    b_x_init = softmin(epsilon_0, cost_xy, log_beta)

    a_x_init = softmin(epsilon_0, cost_xx, log_alpha)
    b_y_init = softmin(epsilon_0, cost_yy, log_beta)

    def stop_condition(i, _a_y, _b_x, _a_x, _b_y, continue_, _running_epsilon):
        n_iter_cond = i < max_iter - 1
        return tf.logical_and(n_iter_cond, tf.reduce_all(continue_))

    def apply_one(a_y, b_x, a_x, b_y, continue_, running_epsilon):
        running_epsilon_ = tf.reshape(running_epsilon, [-1, 1])
        continue_reshaped = tf.reshape(continue_, [-1, 1])
        # TODO: Hopefully one day tensorflow controlflow will be lazy and not strict...
        at_y = tf.where(continue_reshaped, softmin(running_epsilon, cost_yx, log_alpha + b_x / running_epsilon_), a_y)
        bt_x = tf.where(continue_reshaped, softmin(running_epsilon, cost_xy, log_beta + a_y / running_epsilon_), b_x)

        at_x = tf.where(continue_reshaped, softmin(running_epsilon, cost_xx, log_alpha + a_x / running_epsilon_), a_x)
        bt_y = tf.where(continue_reshaped, softmin(running_epsilon, cost_yy, log_beta + b_y / running_epsilon_), b_y)

        a_y_new = (a_y + at_y) / 2
        b_x_new = (b_x + bt_x) / 2

        a_x_new = (a_x + at_x) / 2
        b_y_new = (b_y + bt_y) / 2

        a_y_diff = tf.reduce_max(tf.abs(a_y_new - a_y), 1)
        b_x_diff = tf.reduce_max(tf.abs(b_x_new - b_x), 1)

        local_continue = tf.logical_or(a_y_diff > threshold, b_x_diff > threshold)
        return a_y_new, b_x_new, a_x_new, b_y_new, local_continue

    def body(i, a_y, b_x, a_x, b_y, continue_, running_epsilon):
        new_a_y, new_b_x, new_a_x, new_b_y, local_continue = apply_one(a_y, b_x, a_x, b_y, continue_,
                                                                       running_epsilon)
        new_epsilon = tf.maximum(running_epsilon * scaling_factor, epsilon)
        global_continue = tf.logical_or(new_epsilon < running_epsilon, local_continue)

        return i + 1, new_a_y, new_b_x, new_a_x, new_b_y, global_continue, new_epsilon

    n_iter = tf.constant(0)

    total_iter, converged_a_y, converged_b_x, converged_a_x, converged_b_y, _, final_epsilon = tf.while_loop(
        stop_condition,
        body,
        loop_vars=[n_iter,
                   a_y_init,
                   b_x_init,
                   a_x_init,
                   b_y_init,
                   continue_flag,
                   epsilon_0])

    converged_a_y, converged_b_x, converged_a_x, converged_b_y, = tf.nest.map_structure(tf.stop_gradient,
                                                                                        (converged_a_y,
                                                                                         converged_b_x,
                                                                                         converged_a_x,
                                                                                         converged_b_y))
    epsilon_ = tf.reshape(epsilon, [-1, 1])
    final_a_y = softmin(epsilon, cost_yx, log_alpha + converged_b_x / epsilon_)
    final_b_x = softmin(epsilon, cost_xy, log_beta + converged_a_y / epsilon_)
    final_a_x = softmin(epsilon, cost_xx, log_alpha + converged_a_x / epsilon_)
    final_b_y = softmin(epsilon, cost_yy, log_beta + converged_b_y / epsilon_)

    return final_a_y, final_b_x, final_a_x, final_b_y, total_iter + 2


@tf.function
def sinkhorn_potentials(log_alpha, x, log_beta, y, epsilon, scaling, threshold, max_iter):
    cost_xy = cost(x, tf.stop_gradient(y))
    cost_yx = cost(y, tf.stop_gradient(x))
    cost_xx = cost(x, tf.stop_gradient(x))
    cost_yy = cost(y, tf.stop_gradient(y))
    scale = tf.stop_gradient(max_min(x, y))
    a_y, b_x, a_x, b_y, total_iter = sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy, epsilon,
                                                   scale, scaling, threshold, max_iter)

    return a_y, b_x, a_x, b_y, total_iter
