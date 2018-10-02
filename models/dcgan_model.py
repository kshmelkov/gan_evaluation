from config_gan import args

import tensorflow as tf

from utils.sn_conv import convolution as conv2d
from utils.sn_conv import convolution2d_transpose as conv2d_transpose

from utils.generic_utils import get_glorot_uniform_initializer as get_initializer

slim = tf.contrib.slim


@slim.add_arg_scope
def lrelu(inputs, leak=0.1, scope="lrelu"):
    """
    https://github.com/tensorflow/tensorflow/issues/4079
    """
    with tf.variable_scope(scope):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * inputs + f2 * abs(inputs)


def batch_norm_params(is_training):
    return {
        "decay": args.bn_decay,
        "epsilon": 1e-5,
        "scale": True,
        "updates_collections": None,
        "is_training": is_training
    }


def gen_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([conv2d_transpose, slim.fully_connected],
                        weights_initializer=get_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params(is_training),
                        outputs_collections=outputs_collections):
        with slim.arg_scope([conv2d_transpose],
                            kernel_size=[4, 4], stride=2, padding="SAME") as arg_scp:
            return arg_scp


def disc_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([conv2d],
                        weights_initializer=get_initializer(),
                        activation_fn=lrelu,
                        normalizer_fn=None,
                        normalizer_params=None,
                        kernel_size=[3, 3], stride=1, padding="SAME",
                        outputs_collections=outputs_collections) as arg_scp:
        return arg_scp


def generator(z, is_training, y=None, gen_input=None, scope=None, num_classes=None):
    inputs = tf.concat(z, 1)
    with tf.variable_scope(scope or "generator") as scp:
        end_pts_collection = scp.name+"end_pts"
        with slim.arg_scope(gen_arg_scope(is_training, end_pts_collection)):
            net = slim.fully_connected(inputs, 4*4*512,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope="projection")
            net = slim.batch_norm(net, scope="batch_norm",
                                  **batch_norm_params(is_training))
            net = tf.nn.relu(net)
            net = tf.reshape(net, [-1, 4, 4, 512])
            net = conv2d_transpose(net, 256, scope="conv_tp0")
            net = conv2d_transpose(net, 128, scope="conv_tp1")
            net = conv2d_transpose(net, 64, scope="conv_tp2")
            net = conv2d_transpose(net, 3,
                                   activation_fn=tf.nn.tanh,
                                   normalizer_fn=None,
                                   normalizer_params=None,
                                   kernel_size=[3, 3],
                                   stride=1,
                                   scope="conv_tp3")
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
    return net, end_pts


def discriminator(inputs, is_training, y=None, gen_input=None, reuse=None, scope=None, num_classes=None):
    with tf.variable_scope(scope or "discriminator", values=[inputs], reuse=reuse) as scp:
        end_pts_collection = scp.name+"end_pts"
        if num_classes is None:
            num_classes = args.num_classes
        with slim.arg_scope(disc_arg_scope(is_training, end_pts_collection)):
            net = conv2d(inputs, 64, stride=1, kernel_size=[3, 3], scope="conv0_1")
            net = conv2d(net, 64, stride=2, kernel_size=[4, 4], scope="conv0_2")
            net = conv2d(net, 128, stride=1, kernel_size=[3, 3], scope="conv1_1")
            net = conv2d(net, 128, stride=2, kernel_size=[4, 4], scope="conv1_2")
            net = conv2d(net, 256, stride=1, kernel_size=[3, 3], scope="conv2_1")
            net = conv2d(net, 256, stride=2, kernel_size=[4, 4], scope="conv2_2")
            net = conv2d(net, 512, stride=1, kernel_size=[3, 3], scope="conv3")
            gan_logits = conv2d(net, 1,
                                activation_fn=None,
                                kernel_size=[4, 4], stride=1,
                                padding="VALID",
                                normalizer_fn=None,
                                normalizer_params=None,
                                scope="gan_conv4")
            class_logits = conv2d(net, num_classes,
                                  activation_fn=None,
                                  kernel_size=[4, 4], stride=1,
                                  padding="VALID",
                                  normalizer_fn=None,
                                  normalizer_params=None,
                                  scope="cls_conv4")
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            gan_logits = tf.squeeze(gan_logits, [1, 2], name="squeeze")
            class_logits = tf.squeeze(class_logits, [1, 2], name="squeeze")
    return gan_logits, class_logits, end_pts
