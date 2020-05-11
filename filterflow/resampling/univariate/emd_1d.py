import tensorflow as tf


def map_fn(fn, elems, *args, **kwargs):
    if not len(elems):
        return []
    return tf.unstack(tf.map_fn(fn, tf.stack(elems), *args, **kwargs))


def map_zip_fn(fn, elems_1, elems_2, *args, **kwargs):
    if not len(elems_1) or not len(elems_2):
        return []
    return tf.unstack(tf.map_fn(fn, (tf.stack(elems_1), tf.stack(elems_2)), dtype=elems_1[0].dtype, *args, **kwargs))


@tf.function
def transport_1d(w, x, *additional_tensors):
    # TODO: homogenisation of plan and cost calculation
    """
    Computes Pi @ x where Pi is the optimal transport plan
    :param w: weights
    :param x: locations
    :return: Pi @ x
    """
    batch_dims = x.shape[: -1]
    n_batch_dims = len(batch_dims)
    n = x.shape[-1]

    uniform = tf.fill(w.shape[:-1] + [1], 1 / tf.cast(n, float))

    sorting_index = tf.argsort(x, -1)

    i_s = tf.zeros(sorting_index.shape[:-1] + [1], dtype=sorting_index.dtype)
    j_s = tf.zeros(sorting_index.shape[:-1] + [1], dtype=sorting_index.dtype)

    @tf.function
    def sort(z):
        res = tf.gather(z, sorting_index, batch_dims=n_batch_dims)
        return res

    sorted_x = sort(x)
    sorted_w = sort(w)

    sorted_additional_tensors = map_fn(sort, additional_tensors)
    ranges = [tf.range(dim) for dim in batch_dims]

    mesh_grid = tf.meshgrid(*ranges, indexing="ij")

    mesh_grid = [tf.tile(coordinate[..., tf.newaxis], [1] * (n_batch_dims + 1)) for coordinate in mesh_grid]
    temp = tf.zeros_like(x)
    temp_additional_tensors = [tf.zeros_like(tensor) for tensor in additional_tensors]

    @tf.function
    def where_one(mask, tensor_1, tensor_2):
        n_dim_mask = len(mask.shape)
        n_dim = len(tensor_1.shape)
        reshaped_mask = tf.reshape(mask, mask.shape + [1] * (n_dim - n_dim_mask))
        return tf.where(reshaped_mask, tensor_1, tensor_2)

    @tf.function
    def where(mask, tensors_1, tensors_2):
        return [where_one(mask, t1, t2) for t1, t2 in zip(tensors_1, tensors_2)]

    @tf.function
    def case_0(alpha_i, beta_j, tilde_x, i, j, *args):
        return (alpha_i, beta_j, tilde_x, i, j) + args

    @tf.function
    def case_1(alpha_i, beta_j, sorted_x_j, i, j, *args):
        update = sorted_x_j * alpha_i
        args_update = map_fn(lambda z: z * tf.expand_dims(alpha_i, -1), args)

        beta_j = beta_j - alpha_i
        return [uniform, beta_j, update, i + 1, j] + args_update

    @tf.function
    def case_2(alpha_i, beta_j, sorted_x_j, i, j, *args):
        update = sorted_x_j * beta_j
        args_update = map_fn(lambda z: z * tf.expand_dims(beta_j, -1), args)
        alpha_i = alpha_i - beta_j
        beta_j = tf.gather(sorted_w, tf.minimum(j + 1, n - 1), batch_dims=n_batch_dims)
        return [alpha_i, beta_j, update, i, j + 1] + args_update

    @tf.function
    def i_j_condition(i, j):
        return tf.logical_and(i < n, j < n)

    @tf.function
    def body_if_not_finished(alpha_i, beta_j, tilde_x, i, j, *args):
        mask = tf.logical_or(alpha_i < beta_j, j == n - 1)
        sorted_x_j = tf.gather(sorted_x, j, batch_dims=n_batch_dims)
        sorted_additional_tensors_j = map_fn(lambda z: tf.gather(z, j, batch_dims=n_batch_dims),
                                             sorted_additional_tensors)

        res = where(mask,
                    case_1(alpha_i, beta_j, sorted_x_j, i, j, *sorted_additional_tensors_j),
                    case_2(alpha_i, beta_j, sorted_x_j, i, j, *sorted_additional_tensors_j))
        alpha_i, beta_j, update, i_after, j, *args_updates = res
        indices = tf.stack(mesh_grid + [i], -1)
        tilde_x = tf.tensor_scatter_nd_add(tilde_x, indices, update)
        updated_args = map_zip_fn(lambda z: tf.tensor_scatter_nd_add(z[0], indices, z[1]), args, args_updates)
        return [alpha_i, beta_j, tilde_x, i_after, j] + updated_args

    @tf.function
    def body(alpha_i, beta_j, tilde_x, i, j, *args):
        mask = i_j_condition(i, j)
        res = where(mask,
                    body_if_not_finished(alpha_i, beta_j, tilde_x, i, j, *args),
                    case_0(alpha_i, beta_j, tilde_x, i, j, *args))

        alpha_i, beta_j, tilde_x, i, j, *updated_args = res

        return [alpha_i, beta_j, tilde_x, i, j] + updated_args

    @tf.function
    def cond(_alpha_i, _beta_j, _tilde_x, i, j, *_args):
        return tf.reduce_all(i_j_condition(i, j))

    w_0 = tf.expand_dims(sorted_w[..., 0], -1)
    _, _, res, _, _, *additional_res = tf.while_loop(cond, body, [uniform, w_0, temp, i_s,
                                                                  j_s] + temp_additional_tensors)

    if len(additional_res):
        return [res] + additional_res
    else:
        return res
