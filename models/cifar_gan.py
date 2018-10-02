# from config import args
from config_gan import args

import tensorflow as tf
from utils.sn_conv import convolution as conv2d, spectral_normed_weight
from utils.batch_norm import cond_batch_norm

from utils.generic_utils import get_glorot_uniform_initializer as get_initializer

slim = tf.contrib.slim


def get_normalization_fn():
    if args.conditional_bn and not args.unconditional:
        return cond_batch_norm
    else:
        return slim.batch_norm


def batch_norm_params(is_training):
    return {
        "decay": args.bn_decay,
        "epsilon": 1e-5,
        "scale": True,
        "updates_collections": None,
        "is_training": is_training
    }


@slim.add_arg_scope
def gen_resblock(inputs, depth=128, upsample=False, scope=None, is_training=True,
                 normalizer_fn=None, normalizer_params=None):
    with tf.variable_scope(scope, 'gen_resblock', [inputs]) as sc:
        shortcut = inputs
        if upsample or inputs.shape[-1] != depth:
            _, h, w, _ = shortcut.shape
            if upsample:
                shortcut = tf.image.resize_nearest_neighbor(shortcut, (h*2, w*2))
            shortcut = conv2d(shortcut, depth, scope="conv_sc", kernel_size=[1, 1],
                              weights_initializer=get_initializer(relu=False))

        net = inputs
        if normalizer_fn is not None:
            net = normalizer_fn(net, **normalizer_params, scope="bn1")
        net = tf.nn.relu(net)

        if upsample:
            _, h, w, _ = inputs.shape
            net = tf.image.resize_nearest_neighbor(net, (h*2, w*2))
        net = conv2d(net, depth, scope='conv1')
        if normalizer_fn is not None:
            net = normalizer_fn(net, **normalizer_params, scope="bn2")
        net = tf.nn.relu(net)
        net = conv2d(net, depth, scope='conv2')

        output = shortcut + net
        return output


@slim.add_arg_scope
def dis_resblock(inputs, depth=128, downsample=False, scope=None, is_training=True, first=False):
    with tf.variable_scope(scope, 'dis_resblock', [inputs]) as sc:
        net = inputs
        if not first:
            net = tf.nn.relu(net)
        net = conv2d(net, depth, scope='conv1')
        net = tf.nn.relu(net)
        net = conv2d(net, depth, scope='conv2')

        shortcut = inputs
        if first and downsample:
            shortcut = slim.avg_pool2d(shortcut, [2, 2])
            shortcut = conv2d(shortcut, depth, scope="conv_sc", kernel_size=[1, 1],
                              weights_initializer=get_initializer(relu=False))
            net = slim.avg_pool2d(net, [2, 2])
        else:
            if inputs.shape[-1] != depth or downsample:
                shortcut = conv2d(shortcut, depth, scope="conv_sc", kernel_size=[1, 1],
                                  weights_initializer=get_initializer(relu=False))
            if downsample:
                shortcut = slim.avg_pool2d(shortcut, [2, 2])
                net = slim.avg_pool2d(net, [2, 2])

        output = shortcut + net
        return output


def gen_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.fully_connected, conv2d],
                        weights_initializer=get_initializer(),
                        activation_fn=None,
                        normalizer_fn=None,
                        normalizer_params=None,
                        outputs_collections=outputs_collections):
        with slim.arg_scope([gen_resblock],
                            depth=args.gen_depth,
                            is_training=is_training,
                            normalizer_fn=get_normalization_fn(),
                            normalizer_params=batch_norm_params(is_training)):
            with slim.arg_scope([conv2d],
                                kernel_size=[3, 3],
                                padding="SAME") as arg_scp:
                return arg_scp


def disc_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([conv2d],
                        weights_initializer=tf.variance_scaling_initializer(distribution='uniform',
                                                                            mode='fan_avg'),
                        activation_fn=None,
                        normalizer_fn=None,
                        normalizer_params=None,
                        spectral_normalization=args.spectral_normalization,
                        kernel_size=[3, 3], stride=1, padding="SAME",
                        outputs_collections=outputs_collections) as arg_scp:
        with slim.arg_scope([dis_resblock],
                            is_training=is_training,
                            ) as arg_scp:
            return arg_scp


def generator(z, is_training, y=None, scope=None, num_classes=None):
    inputs = tf.concat(z, 1)
    if args.projection:
        inputs = z[0]
    if args.unconditional:
        labels = None
    else:
        labels = tf.argmax(z[1], axis=1)
    if num_classes is None:
        num_classes = args.num_classes
    with tf.variable_scope(scope or "generator") as scp:
        end_pts_collection = scp.name+"end_pts"
        with slim.arg_scope(gen_arg_scope(is_training, end_pts_collection)):
            with slim.arg_scope([cond_batch_norm],
                                n_labels=num_classes,
                                labels=labels):
                gf_dim = args.gen_linear_dim
                net = slim.fully_connected(inputs, 4*4*gf_dim, scope="projection",
                                           weights_initializer=get_initializer(relu=False))
                net = tf.reshape(net, [-1, 4, 4, gf_dim])
                net = gen_resblock(net, upsample=True, scope='res1')
                net = gen_resblock(net, upsample=True, scope='res2')
                net = gen_resblock(net, upsample=True, scope='res3')
                net = get_normalization_fn()(net, **batch_norm_params(is_training), scope="bn_final")
                net = tf.nn.relu(net)
                net = conv2d(net, 3,
                             activation_fn=tf.nn.tanh,
                             normalizer_fn=None,
                             normalizer_params=None,
                             weights_initializer=get_initializer(relu=False),
                             scope="conv_final")
                end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
    return net, end_pts


def discriminator(inputs, is_training, gen_input=None, reuse=None, scope=None, num_classes=None):
    with tf.variable_scope(scope or "discriminator", values=[inputs], reuse=reuse) as scp:
        end_pts_collection = scp.name+"end_pts"
        if num_classes is None:
            num_classes = args.num_classes
        with slim.arg_scope(disc_arg_scope(is_training, end_pts_collection)):
            net = inputs
            net = dis_resblock(net, first=True, downsample=True, scope='res1')
            net = dis_resblock(net, downsample=True, scope='res2')
            net = dis_resblock(net, scope='res3')
            net = dis_resblock(net, scope='res4')
            net = tf.nn.relu(net)
            if args.sum_pooling:
                net = tf.reduce_sum(net, [1, 2], keepdims=True)
            else:
                net = tf.reduce_mean(net, [1, 2], keepdims=True)
            activations = tf.squeeze(net, [1, 2], name="squeeze")  # [batch_size, num_filters]

            gan_logits = conv2d(net, 1, kernel_size=[1, 1],
                                activation_fn=None,
                                normalizer_fn=None,
                                normalizer_params=None,
                                weights_initializer=get_initializer(relu=False),
                                scope="fc1")
            class_logits = conv2d(net, num_classes, kernel_size=[1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  normalizer_params=None,
                                  weights_initializer=get_initializer(relu=False),
                                  scope="fc1_ac")
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            gan_logits = tf.squeeze(gan_logits, [1, 2], name="squeeze")
            class_logits = tf.squeeze(class_logits, [1, 2], name="squeeze")
            if gen_input is not None and args.projection:
                y = gen_input[1]
                embedding_W = slim.model_variable('embedding', shape=[num_classes, net.shape[-1]],
                                                  initializer=get_initializer(relu=False))
                if args.spectral_normalization:
                    upd_coll = None if not reuse else "NO_OPS"
                    embedding_W = spectral_normed_weight(embedding_W, update_collection=upd_coll)
                embedding = tf.matmul(y, embedding_W)  # [batch_size, num_filters]
                gan_logits += tf.reduce_sum(embedding * activations, axis=1, keepdims=True)
    return gan_logits, class_logits, end_pts
