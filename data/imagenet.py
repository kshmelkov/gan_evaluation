import tensorflow as tf
import numpy as np
import os
import math
import sys
import glob

from paths import DATASETS
from data.dataset_utils import bytes_feature, int64_feature

TFRECORD_ROOT = os.path.join(DATASETS, 'imagenet')


def load_dataset(batch_size, split='train', size=128, augmentation=False,
                 shuffle=True, prefetch_batches=10, classes=None, normalize=True,
                 dequantize=True, reader_threads=12):

    # TODO split
    root = os.path.join(DATASETS, 'imagenet', 'raw_images')
    images_dataset = tf.data.TextLineDataset(os.path.join(root, 'images_list.txt'))
    labels_dataset = tf.data.TextLineDataset(os.path.join(root, 'labels_list.txt'))
    dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))

    if classes is not None:
        classes = tf.convert_to_tensor(classes, dtype=tf.int64)
        dataset = dataset.filter(lambda x, y:
                                 tf.reduce_any(tf.equal(tf.to_int64(tf.string_to_number(y)), classes)))

    if shuffle:
        dataset = dataset.shuffle(1300000)

    def process_single_line(filename, label):
        image_string = tf.read_file(tf.string_join([root, filename], separator='/'))
        image = tf.image.decode_image(image_string, channels=3)
        im_sh = tf.shape(image)
        h, w = im_sh[0], im_sh[1]
        if augmentation:
            crop_size = tf.minimum(h, w)
            image = tf.random_crop(image, [crop_size, crop_size, 3])
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.central_crop(image, 1.0)
            image.set_shape((None, None, 3))
        image = tf.image.resize_images(image, [size, size])
        if normalize:
            image = (image/128.0 - 1.0 +\
                    tf.random_uniform(shape=[size, size, 3], minval=0.0, maxval=1./128))
            if dequantize:
                image = image + tf.random_uniform(
                    shape=[size, size, 3],
                    minval=0.0, maxval=1./128)
        return image, tf.string_to_number(label, out_type=tf.int64)

    dataset = dataset.map(process_single_line, num_parallel_calls=reader_threads)

    dataset = dataset.repeat()
    images, labels = (dataset.batch(batch_size).prefetch(prefetch_batches)
                      .make_one_shot_iterator().get_next())

    return images, labels


def generate_text_files(root_path):
    dirname_to_label = {}
    with open(os.path.join(DATASETS, 'imagenet',
                           'dirname_to_label.txt'), 'r') as f:
        for line in f:
            dirname, label = line.strip('\n').split(' ')
            dirname_to_label[dirname] = label

    count = 0
    n_images_list = []
    n_labels_list = []
    filenames = glob.glob(root_path + '/*/*.JPEG')
    for filename in filenames:
        filename = filename.split('/')
        dirname = filename[-2]
        label = int(dirname_to_label[dirname])
        n_images_list.append(os.path.join(filename[-2], filename[-1]))
        n_labels_list.append(int(label))
        count += 1
        if count % 10000 == 0:
            print(count)
    print("Num of examples:{}".format(count))
    n_image_list = np.array(n_images_list, np.str)
    n_labels_list = np.array(n_labels_list, np.int64)
    np.savetxt(os.path.join(root_path, 'images_list.txt'), n_images_list, fmt="%s")
    np.savetxt(os.path.join(root_path, 'labels_list.txt'), n_labels_list, fmt="%i")


def load_inception_activations(size):
    return np.load(os.path.join(DATASETS, 'imagenet', 'inception_train_%i.npy' % size))


def load_tfrecord_dataset(batch_size, split='train', size=128, augmentation=False,
                          shuffle=True, classes=None, normalize=True,
                          dequantize=True, reader_threads=32):
    """Read the images and labels from 'filenames'."""
    filenames = tf.matching_files(os.path.join(TFRECORD_ROOT, "%s-00*-of-00*" % split))
    dataset = tf.data.TFRecordDataset(filenames).repeat()

    if 'imagenet64' in split:
        disk_sz = 64
    elif 'sngan_128' in split or 'imagenet128' in split:
        disk_sz = 128
    else:
        disk_sz = 256

    def parse_single_example(serialized_example, height=disk_sz, width=disk_sz, depth=3):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([height * width * depth])
        # Reshape from [height * width * depth] to [height, width, depth].
        image = tf.reshape(image, [height, width, depth])
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_images(image, [size, size])
        if augmentation:
            image = tf.image.random_flip_left_right(image)
        label = features['label']
        if normalize:
            image = (image/128.0 - 1.0)
            if dequantize:
                image = image + tf.random_uniform(
                    shape=[size, size, 3],
                    minval=0.0, maxval=1./128)

        return image, label

    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.map(parse_single_example, num_parallel_calls=reader_threads)

    if classes is not None:
        classes = tf.convert_to_tensor(classes, dtype=tf.int64)
        dataset = dataset.filter(lambda x, y: tf.reduce_any(tf.equal(y, classes)))

    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    images, labels = dataset.batch(batch_size).make_one_shot_iterator().get_next()
    return images, labels

def write_tfrecord_dataset(root, split='train', num_shards=256, size=256,
                           shuffle=True, classes=None):
    dataset_sizes = {'train': 1281167, 'validation': 50000}
    dset_sz = dataset_sizes[split]
    print("Starting to convert ImageNet split %s to TFRecord (%i shards)" % (split, num_shards))
    images_per_shard = int(math.ceil(float(dset_sz)/num_shards))
    # root = os.path.join(DATASETS, 'imagenet', '4gans')
    images_dataset = tf.data.TextLineDataset(os.path.join(root, 'images_list.txt'))
    labels_dataset = tf.data.TextLineDataset(os.path.join(root, 'labels_list.txt'))
    dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=dset_sz)

    if classes is not None:
        classes = tf.convert_to_tensor(classes, dtype=tf.int64)
        dataset = dataset.filter(lambda x, y:
                                 tf.reduce_any(tf.equal(tf.to_int64(tf.string_to_number(y)), classes)))

    def process_single_line(filename, label):
        image_string = tf.read_file(tf.string_join([root, filename], separator='/'))
        image = tf.image.decode_image(image_string, channels=3)
        image = tf.image.central_crop(image, 1.0)
        image.set_shape((None, None, 3))
        image = tf.image.resize_images(image, [size, size])
        return tf.cast(image, dtype=tf.uint8), tf.string_to_number(label, out_type=tf.int64)

    dataset = dataset.map(process_single_line, num_parallel_calls=8)
    images, labels = (dataset.batch(images_per_shard).make_one_shot_iterator().get_next())

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        try:
            shard = 0
            while True:
                output_file = os.path.join(TFRECORD_ROOT, '%s-%.5d-of-%.5d' % (split, shard, num_shards))
                batch = sess.run([images, labels])
                convert_to_tfrecord(batch, output_file)
                shard += 1
        except tf.errors.OutOfRangeError:
            print("Looks like processing is finished")


def convert_to_tfrecord(batch, output_file):
  images, labels = batch
  print('Generating %s' % output_file)
  images = images.astype(np.uint8)
  labels = labels.astype(np.int64)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
      for i in range(len(images)):
          example = tf.train.Example(features=tf.train.Features(
              feature={
                  'image': bytes_feature(images[i].tobytes()),
                  'label': int64_feature(labels[i])
              }))
          record_writer.write(example.SerializeToString())


def convert_numpy_to_tfrecord(split):
    folder = "/home/lear/kshmelko/gpu_scratch/sngan_weights/gen_dataset_850k"
    images = []
    labels = []
    for i in range(1000):
        batch = np.load(os.path.join(folder, "%i.npy" % i))
        images.append(batch)
        labels.append([i]*(batch.shape[0]))
        print("%i class is read from disk" % i)

    images = np.concatenate(images, 0)
    labels = np.concatenate(labels, 0)
    num_images = len(images)

    perm = np.random.permutation(num_images)
    num_shards = 32

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        try:
            for shard, batch_indices in enumerate(np.split(perm, num_shards)):
                output_file = os.path.join(TFRECORD_ROOT, '%s-%.5d-of-%.5d' % (split, shard, num_shards))
                batch = (images[batch_indices], labels[batch_indices])
                convert_to_tfrecord(batch, output_file)
        except tf.errors.OutOfRangeError:
            print("Looks like processing is finished")


if __name__ == "__main__":
    convert_numpy_to_tfrecord("sngan_128_850k")
    quit(0)
    val_root = os.path.join(DATASETS, 'imagenet_validation_images')
    generate_text_files(val_root)
    write_tfrecord_dataset(val_root, split='validation', num_shards=16)
    # quit(0)
    images, labels = load_dataset(64, augmentation=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        img, lbl = sess.run([images, labels])
        print(img.shape, lbl.shape)

