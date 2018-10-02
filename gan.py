from config_gan import args, get_logging_config
from paths import DATASETS, CKPT_ROOT

from data import cifar
from models import cifar_gan
from models import dcgan_model
from models import gan_resnet_big
from models import gan_resnet_64
from utils import gan_losses
from data import imagenet
from evaluation.inception import Inception
from evaluation.swd import Sliced_Wasserstein_Scorer
from utils.generic_utils import split_apply_concat
import nn_search
from data import matcher
import classifier

from easydict import EasyDict as edict
import logging
import os
import time
import datetime
from utils import generic_utils as utils

import logging.config

import numpy as np
import tensorflow as tf
tfgan = tf.contrib.gan
slim = tf.contrib.slim


logging.config.dictConfig(get_logging_config(args.run_name))
log = logging.getLogger("gan")



archs = {"resnet": cifar_gan,
         "dcgan": dcgan_model,
         "resnet128": gan_resnet_big,
         "resnet64": gan_resnet_64}


def get_optimizer(name, optimizer=args.optimizer):
    if args.lr_decay:
        global_step = tf.train.get_global_step()
        decay = tf.maximum(0., 1.-(tf.maximum(0., tf.cast(global_step, tf.float32) -
                                              args.linear_decay_start))/args.max_iterations)
    else:
        decay = 1.0

    lr = args.learning_rate*decay
    tf.summary.scalar(name+'_learning_rate', lr)

    if optimizer == 'adam':
        return tf.train.AdamOptimizer(lr, beta1=args.adam_beta1, beta2=args.adam_beta2)
    elif optimizer == 'rmsprop':
        return tf.train.RMSPropOptimizer(lr, decay=0.99)
    else:
        raise NotImplementedError


def train(train_dir):
    # XXX subsampling support is dropped silently
    assert abs(args.subsampling - 1) < 0.01
    target_classes = list(range(args.num_classes))
    # XXX classes support is dropped, UPDATE: retrofitted for class splits
    classes = None
    split_training_mode = args.total_class_splits > 0
    num_classes = args.num_classes
    if split_training_mode:
        assert args.num_classes % args.total_class_splits == 0
        split_sz = args.num_classes // args.total_class_splits
        classes_a = args.active_split_num * split_sz
        classes_b = classes_a + split_sz
        classes = list(range(classes_a, classes_b))
        num_classes = split_sz
        log.info("Class split training mode is activated: "
                 "this run chooses %i split out of %i in total, "
                 "thus classes=%s", args.active_split_num,
                 args.total_class_splits, str(classes))
    images, labels, iter_fn = matcher.load_dataset(args.train_split, args.batch_size,
                                                   args.dataset, args.image_size,
                                                   augmentation=False, shuffle=True,
                                                   classes=classes, normalize=True)
    if split_training_mode:
        labels -= classes_a

    noise = tf.random_normal([args.batch_size, args.noise_dims])
    discriminator_train_steps = args.num_discriminator_steps
    generator_train_steps = 1

    def conditional_generator(x):
        return archs[args.arch].generator(x, True, num_classes=num_classes)[0]

    def conditional_discriminator(x, conditioning):
        gan_logits, class_logits, _ = archs[args.arch].discriminator(x, True,
                                                                     gen_input=conditioning,
                                                                     num_classes=num_classes)
        return gan_logits, class_logits

    def unconditional_discriminator(x, conditioning):
        gan_logits, _, _ = archs[args.arch].discriminator(x, True,
                                                          gen_input=conditioning,
                                                          num_classes=num_classes)
        return gan_logits

    one_hot_labels = tf.one_hot(labels, num_classes)
    if args.unconditional:
        gan_model = tfgan.gan_model(
            generator_fn=conditional_generator,
            discriminator_fn=unconditional_discriminator,
            real_data=images,
            generator_inputs=noise)
    elif args.projection:
        gan_model = tfgan.gan_model(
            generator_fn=conditional_generator,
            discriminator_fn=unconditional_discriminator,
            real_data=images,
            generator_inputs=(noise, one_hot_labels))
    else:
        gan_model = tfgan.acgan_model(
            generator_fn=conditional_generator,
            discriminator_fn=conditional_discriminator,
            real_data=images,
            generator_inputs=(noise, one_hot_labels),
            one_hot_labels=one_hot_labels)

    gp = None if abs(args.gradient_penalty) < 0.01 else args.gradient_penalty
    acgan_gw = None if (args.unconditional or abs(args.acgan_gw) < 0.001) else args.acgan_gw
    acgan_dw = None if (args.unconditional or abs(args.acgan_dw) < 0.001) else args.acgan_dw

    if args.gan_loss == 'hinge':
        model_gen_loss = gan_losses.hinge_generator_loss
        model_dis_loss = gan_losses.hinge_discriminator_loss
    elif args.gan_loss == 'wasserstein':
        model_gen_loss = tfgan.losses.wasserstein_generator_loss
        model_dis_loss = tfgan.losses.wasserstein_discriminator_loss
    elif args.gan_loss == 'classical':
        model_gen_loss = tfgan.losses.modified_generator_loss
        model_dis_loss = tfgan.losses.modified_discriminator_loss
    else:
        raise ValueError("Unsupported GAN loss")

    gan_loss = tfgan.gan_loss(
        gan_model,
        generator_loss_fn=model_gen_loss,
        discriminator_loss_fn=model_dis_loss,
        aux_cond_generator_weight=acgan_gw,
        aux_cond_discriminator_weight=acgan_dw,
        gradient_penalty_weight=gp,
    )

    global_step = tf.train.get_or_create_global_step()
    train_ops = tfgan.gan_train_ops(
        gan_model,
        gan_loss,
        generator_optimizer=get_optimizer("generator"),
        discriminator_optimizer=get_optimizer("discriminator"))

    if args.inception_step > 0:
        real_activations = matcher.load_inception_activations(args.dataset, args.image_size)
        inception = Inception(init_generator(args.eval_batch_size, reuse=True, denormalize=True)[0])

    tfgan.eval.add_gan_model_image_summaries(gan_model, grid_size=int(np.sqrt(args.batch_size)))

    init_assign_op, init_feed_dict = utils.restore_ckpt(train_dir, log)
    summary_op = tf.summary.merge_all()
    clean_init_op = tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)
    tf.get_default_graph().finalize()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        summary_writer = tf.summary.FileWriter(train_dir)
        sess.run(clean_init_op)
        sess.run(init_assign_op, feed_dict=init_feed_dict)
        iter_fn(sess)

        starting_step = sess.run(global_step)
        starting_time = time.time()
        log.info("Starting training from step %i..." % starting_step)
        for step in range(starting_step, args.max_iterations+1):
            start_time = time.time()
            try:
                gen_loss = 0
                for _ in range(generator_train_steps):
                    cur_gen_loss = sess.run(train_ops.generator_train_op)
                    gen_loss += cur_gen_loss

                dis_loss = 0
                for _ in range(discriminator_train_steps):
                    cur_dis_loss = sess.run(train_ops.discriminator_train_op)
                    dis_loss += cur_dis_loss

                sess.run(train_ops.global_step_inc_op)
            except (tf.errors.OutOfRangeError, tf.errors.CancelledError):
                break
            except KeyboardInterrupt:
                log.info("Killed by ^C")
                break

            if step % args.print_step == 0:
                duration = float(time.time() - start_time)
                examples_per_sec = args.batch_size / duration
                log.info("step %i: gen loss = %f, dis loss = %f (%.1f examples/sec; %.3f sec/batch)"
                         % (step, gen_loss, dis_loss, examples_per_sec, duration))
                avg_speed = (time.time() - starting_time)/(step - starting_step + 1)
                time_to_finish = avg_speed * (args.max_iterations - step)
                end_date = datetime.datetime.now() + datetime.timedelta(seconds=time_to_finish)
                log.info("%i iterations left expected to finish at %s (avg speed: %.3f sec/batch)"
                        % (args.max_iterations - step, end_date.strftime("%Y-%m-%d %H:%M:%S"), avg_speed))

            if step % args.summary_step == 0:
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                if args.inception_step != 0 and step % args.inception_step == 0 and step > 0:
                    scores, fids = inception.compute_inception_score_and_fid(real_activations, sess)
                    is_mean, is_std = scores
                    fid_mean, fid_std = fids
                    summary.value.add(tag='Inception_50K_mean', simple_value=is_mean)
                    summary.value.add(tag='Inception_50K_std', simple_value=is_std)
                    summary.value.add(tag='FID_50K_mean', simple_value=fid_mean)
                    summary.value.add(tag='FID_50K_std', simple_value=fid_std)
                    log.info("Inception score over 50K images: %f +- %f" % (is_mean, is_std))
                    log.info("Frechet Inception distance over 50K images: %f +- %f" % (fid_mean, fid_std))
                summary_writer.add_summary(summary, step)

            if step % args.ckpt_step == 0 and step >= 0:
                summary_writer.flush()
                log.debug("Saving checkpoint...")
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)

        summary_writer.close()


def generate(train_dir, suffix=""):
    log.info("Generating %i batches using suffix %s", args.num_generated_batches, suffix)
    images, labels = init_generator(args.eval_batch_size, denormalize=True)
    init_assign_op, init_feed_dict = utils.restore_ckpt(train_dir, log)
    tf.get_default_graph().finalize()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        sess.run(init_assign_op, feed_dict=init_feed_dict)

        def _generate_fake_images(x):
            return sess.run([images, labels])

        data, gt = split_apply_concat(
            np.zeros(args.num_generated_batches*args.eval_batch_size),
            _generate_fake_images, args.num_generated_batches, num_outputs=2)

    split_training_mode = args.total_class_splits > 0
    if split_training_mode:
        assert args.num_classes % args.total_class_splits == 0
        split_sz = args.num_classes // args.total_class_splits
        classes_a = args.active_split_num * split_sz
        log.info("Split training mode for generation: adding %i to all labels", classes_a)
        gt += classes_a

    assert len(data) == len(gt)
    assert len(gt) == args.num_generated_batches*args.eval_batch_size
    data = data.astype(np.uint8)

    np.save(os.path.join(DATASETS, args.dataset, "X_gan_100_%s.npy" % (args.run_name+suffix)), data)
    if not args.unconditional:
        np.save(os.path.join(DATASETS, args.dataset, "Y_gan_100_%s.npy" % (args.run_name+suffix)), gt)


def generate_imagenet(train_dir):
    max_real_batch_size = 128
    num_batches_in_shard = args.eval_batch_size//max_real_batch_size
    images, labels = init_generator(max_real_batch_size, denormalize=True)
    init_assign_op, init_feed_dict = utils.restore_ckpt(train_dir, log)
    tf.get_default_graph().finalize()
    split = 'gan_100_'+args.run_name

    tfrecord_root = os.path.join(DATASETS, args.dataset)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        sess.run(init_assign_op, feed_dict=init_feed_dict)

        num_shards = args.num_generated_batches
        for shard in range(num_shards):
            output_file = os.path.join(tfrecord_root,
                                       '%s-%.5d-of-%.5d' % (split, shard, num_shards))

            def _generate_fake_images(x):
                return sess.run([images, labels])

            batch = split_apply_concat(
                np.zeros(args.eval_batch_size),
                _generate_fake_images, num_batches_in_shard, num_outputs=2)
            imagenet.convert_to_tfrecord(batch, output_file)


def init_generator(batch_size, reuse=None, denormalize=False):
    num_classes = args.num_classes
    split_training_mode = args.total_class_splits > 0
    if split_training_mode:
        assert args.num_classes % args.total_class_splits == 0
        num_classes = args.num_classes // args.total_class_splits

    noise = tf.random_normal([batch_size, args.noise_dims])
    if args.random_labels or batch_size % num_classes != 0:
        labels = tf.multinomial(tf.log([num_classes*[10.]]), batch_size)[0]
    else:
        labels = tf.constant(np.array(list(range(num_classes))*(batch_size//num_classes), dtype='int32'))
    onehot = tf.one_hot(labels, num_classes)

    with tf.variable_scope("Generator", reuse=reuse):
        if args.unconditional:
            inpt = noise
        else:
            inpt = [noise, onehot]
        images = archs[args.arch].generator(inpt, False, num_classes=num_classes)[0]
        if denormalize:
            images = (images+1) * 127.5
        return images, labels


def swd(train_dir):
    bs = 8192

    fake_images, _ = init_generator(bs//32, denormalize=True)
    real_images, _ = cifar.load_cifar("train", bs, normalize=False,
                                      return_numpy=True, dataset=args.dataset)
    real_images = real_images[:bs]
    swd = Sliced_Wasserstein_Scorer(32, 16, 32)
    init_assign_op, init_feed_dict = utils.restore_ckpt(train_dir, log)
    tf.get_default_graph().finalize()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        sess.run(init_assign_op, feed_dict=init_feed_dict)

        def _generate_images(x):
            return sess.run(fake_images)
        fake_images_batch = split_apply_concat(np.arange(bs), _generate_images, 32)
        swd_scores = swd.calc_sliced_wasserstein_scores(real_images, fake_images_batch)
        log.info("SWD scores: %s" % (swd_scores,))
        scaled_swd = 10**3 * np.array(swd_scores)
        log.info("SWD scores * 10^3: %s", str(scaled_swd))


def inception_score(train_dir):
    split = "gan_100_%s" % args.run_name
    if args.inception_file != "":
        split = args.inception_file
    images, _, iter_fn = matcher.load_dataset(split, 100,
                                              args.dataset, args.image_size,
                                              augmentation=False, shuffle=True,
                                              classes=None, normalize=False)

    inception = Inception(images)
    init_assign_op, init_feed_dict = utils.restore_ckpt(train_dir, log)
    tf.get_default_graph().finalize()

    real_activations = matcher.load_inception_activations(args.dataset, args.image_size)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        sess.run(init_assign_op, feed_dict=init_feed_dict)
        iter_fn(sess)
        scores, fids = inception.compute_inception_score_and_fid(real_activations, sess, splits=args.inception_splits)
        is_mean, is_std = scores
        fid_mean, fid_std = fids
        log.info("Final Inception score over 50K images: %f +- %f" % (is_mean, is_std))
        log.info("Final Frechet Inception distance over 50K images of %s: %f +- %f" % (split, fid_mean, fid_std))


def generate_nn_gallery():
    train_dir = os.path.join(CKPT_ROOT, args.dataset+"_classifier_ms_decay")
    args.test_split = "gan_100_%s" % args.run_name
    args.train_split = "train"

    train_images, _ = cifar.load_cifar(args.train_split, args.batch_size,
                                       normalize=False,
                                       dataset=args.dataset,
                                       classes=list(range(args.num_classes)),
                                       return_numpy=True)

    gan_images, _ = cifar.load_cifar(args.test_split, args.batch_size,
                                     normalize=False,
                                     dataset=args.dataset,
                                     classes=list(range(args.num_classes)),
                                     return_numpy=True)

    images_ph = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    images = (images_ph - 127.5)/127.5


    cfg = classifier.get_config(args.dataset, args.image_size)
    model, init_fn = classifier.build_predictor(train_dir, cfg, images)
    features = model.predictions["activations"]
    tf.get_default_graph().finalize()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        init_fn(sess)

        def _compute_features_fn(inpt):
            return sess.run(features, feed_dict={images_ph: inpt})

        def compute_features_fn(inpt):
            num_splits = max(1, len(inpt)//1000)
            return split_apply_concat(inpt, _compute_features_fn, num_splits)

        nn_search.compose_gallery(train_images, gan_images, 5, compute_features_fn)


def compute_nn_distances():
    train_dir = os.path.join(CKPT_ROOT, args.dataset+"_classifier_ms_decay")
    args.test_split = "gan_100_%s" % args.run_name
    args.train_split = "train"

    # TODO rewrite via matcher for universality?
    train_images, _ = cifar.load_cifar(args.train_split, args.batch_size,
                                       normalize=False,
                                       dataset=args.dataset,
                                       classes=list(range(args.num_classes)),
                                       return_numpy=True)


    test_images, _ = cifar.load_cifar("test", args.batch_size,
                                       normalize=False,
                                       dataset=args.dataset,
                                       classes=list(range(args.num_classes)),
                                       return_numpy=True)

    gan_images, _ = cifar.load_cifar(args.test_split, args.batch_size,
                                     normalize=False,
                                     dataset=args.dataset,
                                     classes=list(range(args.num_classes)),
                                     return_numpy=True)

    images_ph = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    images = (images_ph/128.0 - 1.0)

    cfg = classifier.get_config(args.dataset, args.image_size)
    model, init_fn = classifier.build_predictor(train_dir, cfg, images)
    features = model.predictions["activations"]

    import matplotlib.pyplot as plt

    tf.get_default_graph().finalize()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        init_fn(sess)

        def _compute_features_fn(inpt):
            return sess.run(features, feed_dict={images_ph: inpt})

        def compute_features_fn(inpt):
            # num_splits = max(1, len(inpt)//1000)
            num_splits = max(1, len(inpt)//200)
            return split_apply_concat(inpt, _compute_features_fn, num_splits)

        train_gan_dist = nn_search.compute_nn_distance(train_images, gan_images, compute_features_fn, n_neighbors=1)
        train_test_dist = nn_search.compute_nn_distance(train_images, test_images, compute_features_fn, n_neighbors=1)
        gan_test_dist = nn_search.compute_nn_distance(gan_images, test_images, compute_features_fn, n_neighbors=1)
        log.info("GAN-train avg 1-NN = %f; GAN-test avg 1-NN = %f; train-test 1-NN = %f", train_gan_dist.mean(), gan_test_dist.mean(), train_test_dist.mean())

if __name__ == '__main__':
    utils.show_startup_logs(log)

    train_dir = CKPT_ROOT + args.run_name
    for action in args.action.split(','):
        tf.reset_default_graph()
        if action == 'train_gan':
            train(train_dir)
        elif action == 'generate':
            if args.dataset == 'imagenet':
                generate_imagenet(train_dir)
            else:
                generate(train_dir)
        elif action == 'train_all_classifiers':
            cfg = classifier.get_config(args.dataset, args.image_size)
            cfg.training.split = 'gan_100_' + args.run_name
            cfg.evaluation.split = args.test_split
            train_acc = classifier.train_classifier(train_dir+"_resnet_classifier", cfg)
            log.info("Accuracy (GAN-train) = %.4f", train_acc)
            tf.reset_default_graph()

            cls_train_dir = CKPT_ROOT + args.dataset + '_classifier_ms_decay'
            if args.train_split == 'train_shuffled':
                cls_train_dir = CKPT_ROOT + args.dataset + '_shuffled_classifier_ms_decay'
            cfg.training.split = args.train_split
            cfg.evaluation.split = 'gan_100_' + args.run_name
            rev_acc = classifier.evaluate_classifier(cls_train_dir, cfg)
            log.info("Accuracy (GAN-test) = %.4f", rev_acc)

            # Final summary
            log.info("Summary for classification experiments on %s:"
                     "\n | Acc (GAN-train) | Acc (GAN-test) |"
                     "\n | %.4f | %.4f |", args.run_name, train_acc, rev_acc)
        elif action == 'train_resnet_classifier':
            cfg = classifier.get_config(args.dataset, args.image_size)
            cfg.training.split = 'gan_100_' + args.run_name
            cfg.evaluation.split = args.test_split
            classifier.train_classifier(train_dir+"_resnet_classifier", cfg)
        elif action == 'train_reverse_classifier':
            cfg = classifier.get_config(args.dataset, args.image_size)
            cls_train_dir = CKPT_ROOT + args.dataset + '_classifier_ms_decay'
            cfg.evaluation.split = 'gan_100_' + args.run_name
            classifier.evaluate_classifier(cls_train_dir, cfg)
        elif action == 'inception_score':
            inception_score(train_dir)
        elif action == 'swd':
            swd(train_dir)
        elif action == 'nn_search':
            generate_nn_gallery()
        elif action == 'nn_dist':
            compute_nn_distances()
        else:
            print("Action is unknown")
            quit(1)
