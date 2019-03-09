import argparse
import logging
import logging.config
import time
import datetime
import os
from easydict import EasyDict as edict

import tensorflow as tf

import resnet_tf_models as resnet
from data import matcher
from utils import generic_utils as utils
from logging_config import get_logging_config

from paths import CKPT_ROOT

log = logging.getLogger("gan")


class CifarModel(resnet.Model):

  def __init__(self, resnet_size, num_classes, data_format=None):
    """These are the parameters that work for CIFAR-10 data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
    """
    if resnet_size % 6 != 2:
      raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    num_blocks = (resnet_size - 2) // 6

    super(CifarModel, self).__init__(
        resnet_size=resnet_size,
        num_classes=num_classes,
        num_filters=16,
        kernel_size=3,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        second_pool_size=8,
        second_pool_stride=1,
        block_fn=resnet.building_block,
        block_sizes=[num_blocks] * 3,
        block_strides=[1, 2, 2],
        final_size=64,
        data_format=data_format)


def cifar_model_fn(features, labels, mode, params):
  """Model function for CIFAR."""
  learning_rate_fn = resnet.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=128,
      num_images=50000, boundary_epochs=[82, 123],
      decay_rates=[1, 0.1, 0.01])

  # We use a weight decay of 0.0002, which performs better
  # than the 0.0001 that was originally suggested.
  weight_decay = 2e-4

  # Empirical testing showed that including batch_normalization variables
  # in the calculation of regularized loss helped validation accuracy
  # for the CIFAR-10 dataset, perhaps because the regularization prevents
  # overfitting on the small data set. We therefore include all vars when
  # regularizing and computing loss during training.
  def loss_filter_fn(name):
    return True

  return resnet.resnet_model_fn(features, labels, mode, CifarModel,
                                num_classes=params['num_classes'],
                                resnet_size=params['resnet_size'],
                                weight_decay=weight_decay,
                                learning_rate_fn=learning_rate_fn,
                                momentum=0.9,
                                data_format=params['data_format'],
                                loss_filter_fn=loss_filter_fn)

class SmallImagenetModel(resnet.Model):

  def __init__(self, resnet_size, num_classes=1000, data_format=None):
    """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
    """

    super(SmallImagenetModel, self).__init__(
        resnet_size=resnet_size,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=5,
        conv_stride=2,
        first_pool_size=None,
        first_pool_stride=None,
        second_pool_size=4,
        second_pool_stride=1,
        block_fn=resnet.building_block,
        block_sizes=[3, 4, 6, 3],
        block_strides=[1, 2, 2, 2],
        final_size=512,
        data_format=data_format)


class BigImagenetModel(resnet.Model):

  def __init__(self, resnet_size, num_classes=1000, data_format=None):
    """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
    """

    super(BigImagenetModel, self).__init__(
        resnet_size=resnet_size,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=5,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        second_pool_size=4,
        second_pool_stride=1,
        block_fn=resnet.building_block,
        block_sizes=[3, 4, 6, 3],
        block_strides=[1, 2, 2, 2],
        final_size=512,
        data_format=data_format)


def imagenet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  learning_rate_fn = resnet.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=128,
      num_images=1280000, boundary_epochs=[10, 20, 30, 40],
      decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])
  if features.shape[2] == 128:
    imagenet_model = BigImagenetModel
  elif features.shape[2] == 64:
    imagenet_model = SmallImagenetModel
  else:
    raise NotImplementedError

  return resnet.resnet_model_fn(features, labels, mode, imagenet_model,
                                num_classes=params['num_classes'],
                                resnet_size=params['resnet_size'],
                                weight_decay=1e-4,
                                learning_rate_fn=learning_rate_fn,
                                momentum=0.9,
                                data_format=params['data_format'],
                                loss_filter_fn=None)



def train_classifier(train_dir, cfg):
    log.info("Training classifier network from the following config: %s", str(cfg))
    assert cfg.evaluation.test_set_size % cfg.evaluation.batch_size == 0
    num_test_batches = cfg.evaluation.test_set_size//cfg.evaluation.batch_size

    model_params = {
        'resnet_size': cfg.resnet_size,
        'data_format': 'channels_first',
        'batch_size': cfg.training.batch_size,
        'num_classes': cfg.dataset.num_classes,
    }

    log.info("Creating the graph...")
    images, labels, iter_fn = matcher.load_dataset(cfg.training.split, cfg.training.batch_size,
                                                   cfg.dataset.name, cfg.dataset.image_size,
                                                   augmentation=True, shuffle=True,
                                                   normalize=True, onehot=True)
    val_images, val_labels, val_iter_fn = matcher.load_dataset(cfg.evaluation.split, cfg.evaluation.batch_size,
                                                               cfg.dataset.name, cfg.dataset.image_size,
                                                               augmentation=False, shuffle=True,
                                                               normalize=True, onehot=True, dequantize=False)

    with tf.variable_scope("resnet", reuse=None):
        model = cfg.network_fn(images, labels, tf.estimator.ModeKeys.TRAIN, model_params)

    with tf.variable_scope("resnet", reuse=True):
        val_model = cfg.network_fn(val_images, val_labels, tf.estimator.ModeKeys.EVAL, model_params)

    local_init_op = tf.local_variables_initializer()
    clean_init_op = tf.group(tf.global_variables_initializer(), local_init_op)
    global_step = tf.train.get_or_create_global_step()
    init_assign_op, init_feed_dict = utils.restore_ckpt(train_dir, log)
    saver = tf.train.Saver(max_to_keep=1)

    summary_op = tf.summary.merge_all()
    tf.get_default_graph().finalize()

    log.info("Creating the session...")
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        summary_writer = tf.summary.FileWriter(train_dir)
        sess.run(clean_init_op)
        sess.run(init_assign_op, feed_dict=init_feed_dict)
        iter_fn(sess)
        val_iter_fn(sess)

        starting_step = sess.run(global_step)
        starting_time = time.time()

        log.info("Starting training from step %i..." % starting_step)
        for step in range(starting_step, cfg.training.max_iterations+1):
            start_time = time.time()
            try:
                _, train_loss = sess.run([model.train_op, model.loss])
            except (tf.errors.OutOfRangeError, tf.errors.CancelledError):
                break
            except KeyboardInterrupt:
                log.info("Killed by ^C")
                break

            if step % cfg.training.print_step == 0:
                duration = float(time.time() - start_time)
                examples_per_sec = cfg.training.batch_size / duration
                avg_speed = (time.time() - starting_time)/(step - starting_step + 1)
                time_to_finish = datetime.timedelta(
                  seconds=(avg_speed * (cfg.training.max_iterations - step)))
                end_date = datetime.datetime.now() + time_to_finish
                format_str = ('step %d, %.3f (%.1f examples/sec; %.3f sec/batch)')
                log.info(format_str % (step, train_loss, examples_per_sec, duration))
                log.info("%i iterations left expected to finish after %s, thus at %s (avg speed: %.3f sec/batch)"
                         % (cfg.training.max_iterations - step, str(time_to_finish),
                            end_date.strftime("%Y-%m-%d %H:%M:%S"), avg_speed))

            if step % cfg.training.summary_step == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % cfg.training.ckpt_step == 0 and step > 0:
                summary_writer.flush()
                log.debug("Saving checkpoint...")
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)

            if step % cfg.training.eval_step == 0:
                sess.run(local_init_op)  # reset counters in metrics
                for _ in range(num_test_batches):
                    eval_acc = sess.run(val_model.eval_metric_ops)['accuracy'][1]
                log.info("Intermediate evaluation accuracy: %.4f" % eval_acc)
                summary = tf.Summary()
                summary.value.add(tag='accuracy/test', simple_value=eval_acc)
                summary_writer.add_summary(summary, step)

        summary_writer.close()

        sess.run(local_init_op)  # reset counters in metrics
        for i in range(num_test_batches):
              eval_metrics = sess.run(val_model.eval_metric_ops)
              eval_acc = eval_metrics['accuracy'][1]
              eval_acc_top5 = eval_metrics['top5_accuracy'][1]
        log.info("Final evaluation accuracy on split %s: %.4f", cfg.evaluation.split, eval_acc)
        if cfg.dataset.name == 'imagenet':
          log.info("Final evaluation top-5 accuracy on split %s: %.4f", cfg.evaluation.split, eval_acc_top5)
        return eval_acc


def build_predictor(train_dir, cfg, images):
    log.info("Loading classifier network from the following config: %s", str(cfg))
    model_params = {
        'resnet_size': cfg.resnet_size,
        'data_format': 'channels_first',
        'batch_size': cfg.evaluation.batch_size,
        'num_classes': cfg.dataset.num_classes,
    }

    with tf.variable_scope("resnet", reuse=None):
        model = cfg.network_fn(images, None, tf.estimator.ModeKeys.PREDICT, model_params)
    init_assign_op, init_feed_dict = utils.restore_ckpt(train_dir, log)

    def init_classifier_fn(sess):
        sess.run(init_assign_op, feed_dict=init_feed_dict)

    return model, init_classifier_fn


def evaluate_classifier(train_dir, cfg):
    log.info("Loading classifier network from the following config: %s", str(cfg))
    model_params = {
        'resnet_size': cfg.resnet_size,
        'data_format': 'channels_first',
        'batch_size': cfg.evaluation.batch_size,
        'num_classes': cfg.dataset.num_classes,
    }
    images, labels, iter_fn = matcher.load_dataset(cfg.evaluation.split, cfg.training.batch_size,
                                                   cfg.dataset.name, cfg.dataset.image_size,
                                                   augmentation=False, shuffle=False,
                                                   normalize=True, onehot=True)

    with tf.variable_scope("resnet", reuse=None):
        model = cfg.network_fn(images, labels, tf.estimator.ModeKeys.EVAL, model_params)
    init_assign_op, init_feed_dict = utils.restore_ckpt(train_dir, log)
    local_init_op = tf.local_variables_initializer()
    tf.get_default_graph().finalize()

    log.info("Creating the session...")
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        sess.run(init_assign_op, feed_dict=init_feed_dict)
        iter_fn(sess)
        sess.run(local_init_op)  # reset counters in metrics
        num_test_batches = cfg.evaluation.test_set_size//cfg.evaluation.batch_size
        for i in range(num_test_batches):
            eval_acc = sess.run(model.eval_metric_ops)['accuracy'][1]
        log.info("Final evaluation accuracy on split %s: %.4f", cfg.evaluation.split, eval_acc)
    return eval_acc


cifar10_config = edict({'evaluation': {}, 'training': {}, 'dataset': {}})
cifar10_config.dataset.name = 'cifar10'
cifar10_config.dataset.num_classes = 10
cifar10_config.dataset.image_size = 32
cifar10_config.resnet_size = 32
cifar10_config.network_fn = cifar_model_fn
cifar10_config.evaluation.batch_size = 500
cifar10_config.evaluation.test_set_size = 10000
cifar10_config.evaluation.split = "test"
cifar10_config.training.batch_size = 128
cifar10_config.training.max_iterations = 64000
cifar10_config.training.ckpt_step = 1000
cifar10_config.training.eval_step = 5000
cifar10_config.training.print_step = 500
cifar10_config.training.summary_step = 500
cifar10_config.training.split = "train"

cifar100_config = edict({'evaluation': {}, 'training': {}, 'dataset': {}})
cifar100_config.dataset.name = 'cifar100'
cifar100_config.dataset.num_classes = 100
cifar100_config.dataset.image_size = 32
cifar100_config.resnet_size = 32
cifar100_config.network_fn = cifar_model_fn
cifar100_config.evaluation.batch_size = 500
cifar100_config.evaluation.test_set_size = 10000
cifar100_config.evaluation.split = "test"
cifar100_config.training.batch_size = 128
cifar100_config.training.max_iterations = 64000
cifar100_config.training.ckpt_step = 1000
cifar100_config.training.eval_step = 5000
cifar100_config.training.print_step = 500
cifar100_config.training.summary_step = 500
cifar100_config.training.split = "train"

imagenet64_config = edict({'evaluation': {}, 'training': {}, 'dataset': {}})
imagenet64_config.dataset.name = 'imagenet'
imagenet64_config.dataset.num_classes = 1000
imagenet64_config.dataset.image_size = 64
imagenet64_config.resnet_size = 34
imagenet64_config.network_fn = imagenet_model_fn
imagenet64_config.evaluation.batch_size = 100
imagenet64_config.evaluation.test_set_size = 50000
imagenet64_config.evaluation.split = "validation"
imagenet64_config.training.batch_size = 128
imagenet64_config.training.max_iterations = 400000
imagenet64_config.training.ckpt_step = 1000
imagenet64_config.training.eval_step = 5000
imagenet64_config.training.print_step = 500
imagenet64_config.training.summary_step = 500
imagenet64_config.training.split = "train"

lsun_config = edict(imagenet64_config)
lsun_config.dataset.name = 'lsun'
lsun_config.dataset.num_classes = 10
lsun_config.dataset.image_size = 128
lsun_config.evaluation.split = "val"

def get_config(dataset, size):
    if dataset == 'cifar10':
        return edict(cifar10_config)
    elif dataset == 'cifar100':
        return edict(cifar100_config)
    elif dataset == 'imagenet':
        d = edict(imagenet64_config)
        d.dataset.image_size = size
        return d
    elif dataset == 'lsun':
        return edict(lsun_config)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or eval a model trained on CIFAR-10 or CIFAR-100.')
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--test_split", type=str, default="test")

    args = parser.parse_args()

    cfg = get_config(args.dataset, args.image_size)

    cfg.run_name = args.run_name
    cfg.training.split = args.train_split
    cfg.evaluation.split = args.test_split

    logging.config.dictConfig(get_logging_config(args.run_name))

    train_classifier(CKPT_ROOT+args.run_name, cfg)
