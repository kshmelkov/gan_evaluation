import tensorflow as tf
import numpy as np
from paths import DATASETS
import os

slim = tf.contrib.slim

datasets = {
    'cifar100': {
        'mean': np.array([129.3, 124.1, 112.4]),
        'std': np.array([68.2, 65.4, 70.4]),
        'url': 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
        'labels': [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
            'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
            'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
            'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
            'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
            'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
            'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
            'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
            'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
            'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
            'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
            'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
            'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm',
        ],
        'folder': 'cifar-100-python',
        'num_train_files': 1,
        'train_filename_fn': lambda i: 'train',
        'test_filename': 'test',
    },
    'cifar10': {
        'mean': np.array([125.3, 123.0, 113.9]),
        'std': np.array([63.0, 62.0, 66.7]),
        'url': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
        'labels': [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck',
        ],
        'folder': 'cifar-10-batches-py',
        'num_train_files': 5,
        'train_filename_fn': lambda i: 'data_batch_%d' % (i+1),
        'test_filename': 'test_batch',
    }
}


def distort_image(im):
    im = tf.image.random_flip_left_right(im)
    im = tf.pad(im, [[4, 4], [4, 4], [0, 0]])
    im = tf.random_crop(im, [32, 32, 3])
    return im


def data_augmentation(im, label):
    im = distort_image(im)
    return im, label


def get_fixed_permutation(num_classes):
    # obtained via np.random.permutation(100)
    # should be 100% random
    if num_classes == 100:
        return np.array([83, 53, 70, 45, 44, 39, 22, 80, 10, 0, 18, 30, 73, 33, 90, 4, 76,
                77, 12, 31, 55, 88, 26, 42, 69, 15, 40, 96, 9, 72, 11, 47, 85, 28, 93,
                5, 66, 65, 35, 16, 49, 34, 7, 95, 27, 19, 81, 25, 62, 13, 24, 3, 17, 38,
                8, 78, 6, 64, 36, 89, 56, 99, 54, 43, 50, 67, 46, 68, 61, 97, 79, 41,
                58, 48, 98, 57, 75, 32, 94, 59, 63, 84, 37, 29, 1, 52, 21, 2, 23, 87,
                91, 74, 86, 82, 20, 60, 71, 14, 92, 51])
    else:
        raise NotImplementedError


def choose_split(images, labels, num_classes, num_split, split_size):
    assert num_classes % split_size == 0
    assert num_split < num_classes // split_size
    a = num_split*split_size
    b = (num_split+1)*split_size
    perm = get_fixed_permutation(num_classes)
    labels = perm[labels]
    idx = np.logical_and(labels >= a, labels < b)
    return images[idx], labels[idx] - a


def reshuffle_cifar100(split, name):
    images, labels = load_cifar(split, 0, normalize=False, dataset='cifar100',
                                return_numpy=True, shuffle=False,
                                augmentation=False)
    perm = get_fixed_permutation(100)
    new_labels = perm[labels]
    images = images.astype(np.uint8)
    dataset_folder = os.path.join(DATASETS, 'cifar100')
    np.save(os.path.join(dataset_folder, "X_%s.npy" % name), images)
    np.save(os.path.join(dataset_folder, "Y_%s.npy" % name), labels)


def load_cifar(split, batch_size, normalize=True, dataset='cifar100',
               return_numpy=False, augmentation=False,
               shuffle=True, prefetch_batches=10, classes=None,
               force_argmax=False, subsample_factor=1.0):
    dataset_folder = os.path.join(DATASETS, dataset)
    images = np.load(os.path.join(dataset_folder, "X_%s.npy" % split))
    labels = np.load(os.path.join(dataset_folder, "Y_%s.npy" % split))
    images = images.astype(np.float32)

    if normalize:
        mean, std = datasets[dataset]['mean'], datasets[dataset]['std']
        mean = mean.reshape((1, 1, 1, 3))
        std = std.reshape((1, 1, 1, 3))
        images = (images - mean)/std
    images = images.astype(np.float32)
    if force_argmax and len(labels.shape) == 2:
        labels = np.argmax(labels, 1)
    if classes is not None and len(labels.shape) == 1:
        assert len(labels.shape) == 1
        # TODO support for logits here
        indices = np.in1d(labels, classes)
        images = images[indices]
        labels = labels[indices]
        assert len(images) > 0
    if abs(subsample_factor - 1) > 0.05:
        u_labels = np.unique(labels)
        samples_per_label = int(len(images)*subsample_factor/len(u_labels))
        print("samples per label = ", samples_per_label)
        images_list = []
        labels_list = []
        for label in u_labels:
            split = images[labels == label]
            indices = np.random.choice(len(split), size=samples_per_label, replace=False)
            images_list.append(split[indices])
            labels_list.append(samples_per_label*[label])

        images = np.concatenate(images_list)
        labels = np.concatenate(labels_list)
        print("Subsampling is done with factor %f: %i images left" %
              (subsample_factor, len(images)))
    if return_numpy:
        return images, labels

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if augmentation:
        dataset = dataset.map(data_augmentation, num_parallel_calls=4)
    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(len(images))
    images, labels = (dataset.batch(batch_size).prefetch(prefetch_batches)
                      .make_one_shot_iterator().get_next())

    images.set_shape((batch_size, 32, 32, 3))
    labels.set_shape((batch_size, ))
    return (images, labels)


def load_cifar_iterator(split, batch_size, normalize=True, dataset='cifar100',
                        augmentation=False, shuffle=True, prefetch_batches=10,
                        classes=None, force_argmax=False):
    images, labels = load_cifar(split, batch_size, return_numpy=True,
                                normalize=normalize,
                                dataset=dataset, classes=classes)

    images_ph = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    labels_ph = tf.placeholder(tf.int64, shape=(None, ))

    dataset = tf.data.Dataset.from_tensor_slices((images_ph, labels_ph))
    if augmentation:
        dataset = dataset.map(data_augmentation, num_parallel_calls=4)
    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(len(images))
    iterator = (dataset.batch(batch_size).prefetch(prefetch_batches)
                .make_initializable_iterator())
    images, labels = iterator

    images.set_shape((batch_size, 32, 32, 3))
    labels.set_shape((batch_size, ))

    def init_iterator(sess):
        sess.run(iterator.initializer, feed_dict={images_ph: images,
                                                  labels_ph: labels})
    return (images, labels), init_iterator


def load_dataset(split, batch_size, normalize=True, dataset='cifar100',
                 augmentation=False, shuffle=True, prefetch_batches=10,
                 dequantize=True, classes=None):
    dataset_folder = os.path.join(DATASETS, dataset)
    # ugly hack to allow support of multiple splits
    # this pattern though looks very generic
    # if only I had enough time to abstract it away...
    if not isinstance(split, list):
        split = [split]
    X = []
    Y = []
    for s in split:
        images = np.load(os.path.join(dataset_folder, "X_%s.npy" % s))
        labels = np.load(os.path.join(dataset_folder, "Y_%s.npy" % s))
        X.append(images)
        Y.append(labels)
    X = np.concatenate(X, 0)
    Y = np.concatenate(Y, 0)
    X = X.astype(np.float32)

    if classes is not None and len(labels.shape) == 1:
        indices = np.in1d(Y, classes)
        X = X[indices]
        X = Y[indices]
        assert len(X) > 0

    images_ph = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    labels_ph = tf.placeholder(tf.int64, shape=(None, ))
    dataset = tf.data.Dataset.from_tensor_slices((images_ph, labels_ph))
    if augmentation:
        dataset = dataset.map(data_augmentation, num_parallel_calls=4)
    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(len(images))

    iterator = (dataset.batch(batch_size).prefetch(prefetch_batches)
                .make_initializable_iterator())
    images, labels = iterator.get_next()

    if normalize:
        images = (images/128.0 - 1.0)
        if dequantize:
            images = images + tf.random_uniform(shape=[batch_size, 32, 32, 3],
                                                minval=0.0, maxval=1./128)

    images.set_shape((batch_size, 32, 32, 3))
    labels.set_shape((batch_size, ))

    def init_iterator_fn(sess):
        sess.run(iterator.initializer, feed_dict={images_ph: X,
                                                  labels_ph: Y})

    return images, labels, init_iterator_fn


def load_inception_activations(dataset):
    return np.load(os.path.join(DATASETS, dataset, 'inception_train.npy'))


if __name__ == '__main__':
    reshuffle_cifar100('train', "train_shuffled")
    reshuffle_cifar100('test', "test_shuffled")
