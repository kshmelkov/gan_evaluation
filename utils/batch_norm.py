import tensorflow as tf
from utils import slim_utils

from tensorflow.contrib.framework.python.ops import add_arg_scope

slim = tf.contrib.slim


@add_arg_scope
def cond_batch_norm(inputs, axes=[0, 1, 2], is_training=None,
                    fused=False, labels=None,
                    n_labels=None, updates_collections=tf.GraphKeys.UPDATE_OPS,
                    epsilon=1e-5, scale=True, decay=0.999, scope=None,
                    reuse=None):
    # TODO adapt it for NCHW data format and restore data_format argument
    if axes != [0, 1, 2]:
        raise ValueError('axes other than [0, 1, 2] are unsupported')
    with tf.variable_scope(scope or "CondBatchNorm", reuse=reuse) as vs:
        mean, variance = tf.nn.moments(inputs, axes, keep_dims=True)
        shape = mean.get_shape().as_list()  # shape is [1,1,1,n]
        params_shape = [n_labels, shape[-1]]
        offset_m = slim.model_variable('offset', shape=params_shape,
                                       initializer=tf.zeros_initializer())
        scale_m = slim.model_variable('scale', shape=params_shape,
                                      initializer=tf.ones_initializer())
        offset = tf.nn.embedding_lookup(offset_m, labels)
        scale = tf.nn.embedding_lookup(scale_m, labels)

        moving_mean = slim.model_variable(
            'moving_mean',
            shape=shape,
            initializer=tf.zeros_initializer(),
            trainable=False)
        moving_variance = slim.model_variable(
            'moving_variance',
            shape=shape,
            trainable=False,
            initializer=tf.ones_initializer())

        is_training_value = slim_utils.constant_value(is_training)
        need_moments = is_training_value is None or is_training_value
        if need_moments:
            moving_vars_fn = lambda: (moving_mean, moving_variance)
            if updates_collections is None:
                def _force_updates():
                    """Internal function forces updates moving_vars if is_training."""
                    update_moving_mean = slim_utils.assign_moving_average(
                        moving_mean, mean, decay)
                    update_moving_variance = slim_utils.assign_moving_average(
                        moving_variance, variance, decay)
                    with tf.control_dependencies(
                            [update_moving_mean, update_moving_variance]):
                        return tf.identity(mean), tf.identity(variance)
                mean, variance = slim_utils.smart_cond(is_training, _force_updates,
                                            moving_vars_fn)
            else:
                def _delay_updates():
                    """Internal function that delay updates moving_vars if is_training."""
                    update_moving_mean = slim_utils.assign_moving_average(
                        moving_mean, mean, decay)
                    update_moving_variance = slim_utils.assign_moving_average(
                        moving_variance, variance, decay)
                    return update_moving_mean, update_moving_variance

                update_mean, update_variance = tf.cond(
                    is_training, _delay_updates, moving_vars_fn)
                tf.add_to_collections(updates_collections, update_mean)
                tf.add_to_collections(updates_collections, update_variance)
                # Use computed moments during training and moving_vars otherwise.
                vars_fn = lambda: (mean, variance)
                mean, variance = slim_utils.smart_cond(is_training, vars_fn, moving_vars_fn)
        else:
            mean, variance = moving_mean, moving_variance

        result = tf.nn.batch_normalization(inputs, mean, variance,
                                           offset[:, None, None, :],
                                           scale[:, None, None, :], epsilon)
        return result
