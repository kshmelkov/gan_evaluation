from data import cifar
from data import imagenet
import tensorflow as tf

NUM_CLASSES = {'cifar10': 10,
               'cifar100': 100,
               'imagenet': 1000}


def load_dataset(split, batch_size, dataset, size, onehot=False, **kwargs):
    if dataset == 'imagenet':
        if size is None:
            raise ValueError("size should be defined for ImageNet")
        images, labels = imagenet.load_tfrecord_dataset(batch_size, split=split, size=size, **kwargs)
        init_fn = lambda x: None
    if dataset in ['cifar10', 'cifar100']:
        if size != 32:
            raise ValueError("CIFAR is only available in size 32")
        images, labels, init_fn = cifar.load_dataset(split, batch_size, dataset=dataset, **kwargs)
    images.set_shape((batch_size, size, size, 3))
    labels.set_shape((batch_size,))
    if onehot:
        labels = tf.one_hot(labels, NUM_CLASSES[dataset])
    return images, labels, init_fn


def load_inception_activations(dataset, size):
    if dataset == 'imagenet':
        return imagenet.load_inception_activations(size)
    elif dataset in ['cifar10', 'cifar100']:
        return cifar.load_inception_activations(dataset)
    else:
        raise ValueError("Unsupported dataset")


def load_network(model, dataset):
    pass


if __name__ == '__main__':
    load_dataset('train', 128, 'cifar10', 32)
