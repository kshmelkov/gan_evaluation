import tensorflow as tf
import numpy as np
import os
import sys
import urllib
import tarfile
from utils.generic_utils import split_apply_concat

tfgan = tf.contrib.gan

MODEL_DIR = './inception/'
INCEPTION_GRAPH_NAME = 'inceptionv1_for_inception_score.pb'
INCEPTION_INPUT = 'Mul:0'
INCEPTION_OUTPUT = 'logits:0'
INCEPTION_FINAL_POOL = 'pool_3:0'
INCEPTION_DEFAULT_IMAGE_SIZE = 299
INCEPTION_SHAPE = [INCEPTION_DEFAULT_IMAGE_SIZE, INCEPTION_DEFAULT_IMAGE_SIZE]

INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz' 


#### some parts are taken from https://github.com/igul222/improved_wgan_training
class Inception:
    def __init__(self, images):
        _download_inception_if_needed()
        self.images = images
        self.images_ph = tf.placeholder(tf.float32, shape=(None, None, None, None))
        self._preprocess_images()
        self._import_inception_graph()
        self.fake_activations_ph = tf.placeholder(tf.float32, shape=(None, None))
        self.real_activations_ph = tf.placeholder(tf.float32, shape=(None, None))
        self.inception_scores = tfgan.eval.classifier_score_from_logits(self.fake_activations_ph)
        self.frechet_distance = tfgan.eval.frechet_classifier_distance_from_activations(
            self.real_activations_ph, self.fake_activations_ph)

    def _preprocess_images(self):
        resized_images = tf.image.resize_bilinear(self.images_ph,
                                                  INCEPTION_SHAPE)
        self.inception_input = (resized_images - 128.0) / 128.0

    def _import_inception_graph(self):
        with tf.gfile.FastGFile(os.path.join(MODEL_DIR, INCEPTION_GRAPH_NAME),
                                'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            input_map = {INCEPTION_INPUT: self.inception_input}
            output_tensors = [INCEPTION_FINAL_POOL, INCEPTION_OUTPUT]
            inception_output = tf.import_graph_def(
                graph_def, input_map, output_tensors, name='inception')
            self.inception_features = tf.squeeze(inception_output[0], [1, 2])
            # https://github.com/openai/improved-gan/issues/29
            # Fix this for the future. In practice it doesn't matter much.
            self.logits = inception_output[1][:, :1001]

    def _generate_fake_images(self, n, sess):
        def _generate_images(x):
            return sess.run(self.images)
        return split_apply_concat(np.zeros(n), _generate_images, n//100)

    def compute_inception_score_and_fid(self, real_activations, sess, splits=10):
        assert real_activations.ndim == 2
        n = len(real_activations)
        samples = self._generate_fake_images(n, sess)

        def _compute_inception_features_and_logits(x):
            return sess.run([self.inception_features, self.logits],
                            {self.images_ph: x})
        activations, logits = split_apply_concat(
            samples, _compute_inception_features_and_logits, n//500, num_outputs=2)
        scores = self._compute_inception_score_from_logits(logits, sess, splits=splits)
        fids = self._compute_fid_from_activations(real_activations, activations,
                                                  sess, splits=splits)
        return scores, fids

    def compute_inception_score(self, n, sess, splits=10):
        images = self._generate_fake_images(n, sess)

        def _compute_logits(x):
            return sess.run(self.logits, {self.images_ph: x})
        logits = split_apply_concat(images, _compute_logits, splits)
        return self._compute_inception_score_from_logits(logits, sess, splits)

    def _compute_inception_activations(self, data, sess, splits=10):
        def _compute_inception_features(x):
            return sess.run(self.inception_features, {self.images_ph: x})
        return split_apply_concat(data, _compute_inception_features, splits)

    def compute_fid(self, real_activations, sess):
        assert real_activations.ndim == 2
        n = len(real_activations)
        fake_samples = self._generate_fake_images(n, sess)
        fake_activations = self._compute_inception_activations(fake_samples,
                                                               sess, n//100)
        return self._compute_fid_from_activations(real_activations,
                                                  fake_activations, sess)

    def _compute_fid_from_activations(self, real_activations, fake_activations,
                                      sess, splits=10):
        def _compute_fid(x):
            real_act, fake_act = x
            return sess.run(self.frechet_distance,
                            feed_dict={self.real_activations_ph: real_act,
                                       self.fake_activations_ph: fake_act})
        fids = split_apply_concat((real_activations, fake_activations),
                                  _compute_fid, splits)
        return np.mean(fids), np.std(fids)

    def _compute_inception_score_from_logits(self, logits, sess, splits=10):
        def _compute_scores(x):
            return (sess.run(self.inception_scores,
                             feed_dict={self.fake_activations_ph: x}))
        scores = split_apply_concat(logits, _compute_scores, splits)
        return np.mean(scores), np.std(scores)


def _download_inception_if_needed():
    filepath = os.path.join(MODEL_DIR, INCEPTION_GRAPH_NAME)
    if os.path.exists(filepath):
        return
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = INCEPTION_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(INCEPTION_URL, filepath, _progress)
      print()
      statinfo = os.stat(filepath)
      print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
      tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)

if __name__ == '__main__':
    from paths import DATASETS
    import argparse
    from data import matcher


    parser = argparse.ArgumentParser(description='Generate and cache Inception activations')
    parser.add_argument("--dataset", default='cifar10', choices=['imagenet', 'cifar10', 'cifar100'])
    parser.add_argument("--image_size", default=32, type=int)
    args = parser.parse_args()

    inception = Inception(None)
    images, _, iter_fn = matcher.load_dataset('train', 500, normalize=False,
                                     dataset=args.dataset, size=args.image_size)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        iter_fn(sess)

        def _sample_images(x):
            return sess.run(images)
        print("Loading images...")
        dset = split_apply_concat(np.zeros(50000), _sample_images, 100)
        print("Images are loaded: ", dset.shape)
        print("Computing inception activations...")
        activations = inception._compute_inception_activations(dset, sess, splits=100)
        np.save(os.path.join(DATASETS, args.dataset, "inception_train_%i.npy" %
                             args.image_size), activations)
