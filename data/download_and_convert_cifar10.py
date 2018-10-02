# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts cifar* data to TFRecords of TF-Example protos.

This module downloads the cifar* data, uncompresses it, reads the files
that make up the cifar* data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take several minutes to run.

"""
import pickle
import os
import sys

import numpy as np
import tensorflow as tf

from data import dataset_utils
from data.cifar import datasets
from paths import DATASETS


def _extract(filename):
    """Loads data from the cifar10 pickle files and writes files to a TFRecord.

    Args:
      filename: The filename of the cifar10 pickle file.
      tfrecord_writer: The TFRecord writer to use for writing.
      offset: An offset into the absolute number of images previously written.

    Returns:
      The new offset.
    """
    with tf.gfile.Open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    images = data['data']
    num_images = images.shape[0]

    images = images.reshape((num_images, 3, 32, 32))
    images = images.transpose((0, 2, 3, 1))
    if 'labels' in data:
        labels = data['labels']
    else:
        labels = data['fine_labels']

    return images, labels

def _add_to_tfrecord(filename, tfrecord_writer, offset=0):
    """Loads data from the cifar10 pickle files and writes files to a TFRecord.

    Args:
      filename: The filename of the cifar10 pickle file.
      tfrecord_writer: The TFRecord writer to use for writing.
      offset: An offset into the absolute number of images previously written.

    Returns:
      The new offset.
    """
    with tf.gfile.Open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    images = data['data']
    num_images = images.shape[0]

    images = images.reshape((num_images, 3, 32, 32))
    if 'labels' in data:
        labels = data['labels']
    else:
        labels = data['fine_labels']

    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_image = tf.image.encode_png(image_placeholder)

        with tf.Session('') as sess:
            for j in range(num_images):
                sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (
                    filename, offset + j + 1, offset + num_images))
                sys.stdout.flush()

                image = np.squeeze(images[j]).transpose((1, 2, 0))
                label = labels[j]

                png_string = sess.run(encoded_image,
                                      feed_dict={image_placeholder: image})

                example = dataset_utils.image_to_tfexample(
                    png_string, 'png', 32, 32, label)
                tfrecord_writer.write(example.SerializeToString())

    return offset + num_images


def _get_output_filename(dataset_dir, split_name, dataset):
  """Creates the output filename.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
    split_name: The name of the train/test split.
    dataset: The dict containing dataset-specific constants.

  Returns:
    An absolute file path.
  """
  return '%s/%s_%s.tfrecord' % (dataset_dir, dataset, split_name)


def _clean_up_temporary_files(dataset_dir, dataset):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
    dataset: The dict containing dataset-specific constants.
  """
  filename = dataset['url'].split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, dataset['folder'])
  tf.gfile.DeleteRecursively(tmp_dir)


def run(dataset_dir, dataset_name):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
    dataset_name: The dataset name (cifar10 or cifar100)
  """
  dataset = datasets[dataset_name]

  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  training_filename = _get_output_filename(dataset_dir, 'train', dataset_name)
  testing_filename = _get_output_filename(dataset_dir, 'test', dataset_name)

  if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  dataset_utils.download_and_uncompress_tarball(dataset['url'], dataset_dir)

  # First, process the training data:
  with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
    offset = 0
    for i in range(dataset['num_train_files']):
      filename = os.path.join(dataset_dir,
                              dataset['folder'],
                              dataset['train_filename_fn'](i))
      offset = _add_to_tfrecord(filename, tfrecord_writer, offset)

  # Next, process the testing data:
  with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
    filename = os.path.join(dataset_dir, dataset['folder'], dataset['test_filename'])
    _add_to_tfrecord(filename, tfrecord_writer)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(dataset['labels'])), dataset['labels']))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  _clean_up_temporary_files(dataset_dir, dataset)
  print('\nFinished converting the %s dataset!' % dataset_name)


def convert_to_numpy(dataset_dir, dataset_name):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
    dataset_name: The dataset name (cifar10 or cifar100)
  """
  dataset = datasets[dataset_name]

  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  dataset_utils.download_and_uncompress_tarball(dataset['url'], dataset_dir)
  images = []
  labels = []

  for i in range(dataset['num_train_files']):
    filename = os.path.join(dataset_dir,
                            dataset['folder'],
                            dataset['train_filename_fn'](i))
    im, lbl = _extract(filename)
    images.append(im)
    labels.append(lbl)

  images = np.concatenate(images, 0)
  labels = np.concatenate(labels, 0)
  np.save(os.path.join(dataset_dir, "X_train.npy"), images)
  np.save(os.path.join(dataset_dir, "Y_train.npy"), labels)

  filename = os.path.join(dataset_dir, dataset['folder'], dataset['test_filename'])
  images, labels = _extract(filename)
  np.save(os.path.join(dataset_dir, "X_test.npy"), images)
  np.save(os.path.join(dataset_dir, "Y_test.npy"), labels)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(dataset['labels'])), dataset['labels']))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  _clean_up_temporary_files(dataset_dir, dataset)
  print('\nFinished converting the %s dataset!' % dataset_name)


if __name__ == '__main__':
    convert_to_numpy(DATASETS+'cifar10', 'cifar10')
    convert_to_numpy(DATASETS+'cifar100', 'cifar100')
    # run(DATASETS+'cifar10', 'cifar10')
    # run(DATASETS+'cifar100', 'cifar100')
