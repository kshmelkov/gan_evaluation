from config_gan import args, get_logging_config
from paths import DATASETS, CKPT_ROOT

import logging
import logging.config
import numpy as np
from sklearn.neighbors import NearestNeighbors

from utils import plot_utils


logging.config.dictConfig(get_logging_config(args.run_name))
log = logging.getLogger("nn_search")


def nearest_neighbor_search(real_features, target_features, n_neighbors=5):
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(real_features)
    distances, indices = neigh.kneighbors(target_features)
    return indices, distances


def compose_gallery(train_images, gan_images, num_samples,
                    compute_features_fn):
    np.random.shuffle(gan_images)
    n_neighbors = 5
    target_images = gan_images[:num_samples]

    train_features = compute_features_fn(train_images)
    target_features = compute_features_fn(target_images)

    idx, _ = nearest_neighbor_search(train_features, target_features,
                                     n_neighbors=n_neighbors)

    selected_images = train_images[idx].astype(np.uint8)
    target_images = target_images.astype(np.uint8)
    selected_images = [[x] + list(y) for y, x in zip(list(selected_images),
                                                     list(target_images))]
    text = num_samples*[["target"] + ["#%i" % i for i in range(n_neighbors)]]
    img = plot_utils.image_grid(selected_images, margin_after_first_col=20)
    # img = plot_utils.image_grid(selected_images, titles=text, font_size=10)
    # TODO figure out how to include text and whatnot
    img.save("/home/lear/kshmelko/gallery_sample_%s.png" % args.run_name)


def compute_nn_distance(train_images, gan_images, compute_features_fn, n_neighbors=1):
    train_features = compute_features_fn(train_images)
    target_features = compute_features_fn(gan_images)

    _, dist = nearest_neighbor_search(train_features, target_features,
                                      n_neighbors=n_neighbors)
    return dist
