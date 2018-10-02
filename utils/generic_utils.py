import tensorflow as tf
import numpy as np
import sys
import subprocess
import os
import socket
import math

slim = tf.contrib.slim

################################################################################
#######################   AUXILLARY FUNCTIONS   ################################
################################################################################


def get_glorot_uniform_initializer(relu=True):
    scale = math.sqrt(2) if relu else 1.0
    return tf.variance_scaling_initializer(scale=scale, distribution='uniform',
                                           mode='fan_avg')


def gram_matrix(activations):
    b, h, w, c = activations.shape
    features_matrix = tf.reshape(activations, tf.stack([b, -1, c]))
    gram_matrix = tf.matmul(features_matrix, features_matrix,
                            transpose_a=True)
    gram_matrix = gram_matrix / tf.cast(h*w, tf.float32)
    return gram_matrix


def nprelu(x):
    x[x < 0] = 0
    return x


@slim.add_arg_scope
def lrelu(inputs, leak=0.2, scope="lrelu"):
    """
    https://github.com/tensorflow/tensorflow/issues/4079
    """
    with tf.variable_scope(scope):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * inputs + f2 * abs(inputs)


def split_apply_concat(arr, pyfunc, num_splits, num_outputs=1):
    output_list = []
    if isinstance(arr, tuple):
        for x in zip(*list(np.split(y, num_splits) for y in arr)):
            output_list.append(pyfunc(tuple(x)))
    else:
        for x in np.split(arr, num_splits):
            output_list.append(pyfunc(x))
    if num_outputs == 1:
        if output_list[0].ndim == 0:
            return np.array(output_list)
        else:
            return np.concatenate(output_list, 0)
    else:
        return tuple(np.concatenate(out, 0)
                     for out in zip(*output_list))


def restore_ckpt(ckpt_dir, log, ckpt_number=0, global_step=None, reset_slots=False):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        if ckpt_number == 0:
            ckpt_to_restore = ckpt.model_checkpoint_path
        else:
            ckpt_to_restore = ckpt_dir+'/model.ckpt-%i' % ckpt_number
        if reset_slots:
            variables_to_restore = slim.get_model_variables()
            if global_step is not None:
                variables_to_restore += [global_step]
        else:
            variables_to_restore = tf.global_variables()
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
            ckpt_to_restore, variables_to_restore)
        log.info("Restore weights from %s", ckpt_to_restore)
    else:
        init_assign_op = tf.no_op()
        init_feed_dict = None
        log.info("This network is trained from scratch")

    return init_assign_op, init_feed_dict


def show_startup_logs(log):
    exec_string = ' '.join(sys.argv)
    log.debug("Executing a command: %s", exec_string)
    cur_commit = subprocess.check_output("git log -n 1 --pretty=format:\"%H\"".split())
    cur_branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD".split())
    git_diff = subprocess.check_output('git diff --no-color'.split()).decode('ascii')
    log.debug("on branch %s with the following diff from HEAD (%s):" % (cur_branch, cur_commit))
    log.debug(git_diff)
    hostname = socket.gethostname()
    if 'gpuhost' in hostname:
        log.info("Current host is %s:" % (hostname, ))
        gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
        log.info("Currently used GPU is %s" % (gpu_id, ))
        nvidiasmi = subprocess.check_output('nvidia-smi').decode('ascii')
        log.info(nvidiasmi)


def count_parameters():
    vars = tf.trainable_variables()
    params = 0
    for v in vars:
        print(v.op.name, v.shape)
        params += np.prod(v.shape.as_list())

    print("In total %i parameters" % params)



