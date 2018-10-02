import argparse
import os
import numpy as np

from paths import GPU_DATASETS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or eval a model trained on CIFAR-10 or CIFAR-100.')
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--new_name", type=str, required=True)
    parser.add_argument("--subsample", type=float, default=0.5)

    args = parser.parse_args()

    dataset_folder = os.path.join(GPU_DATASETS, args.dataset)
    images = np.load(os.path.join(dataset_folder, "X_%s.npy" % args.split))
    labels = np.load(os.path.join(dataset_folder, "Y_%s.npy" % args.split))

    new_split_img = []
    new_split_lbl = []

    for l in np.unique(labels):
        idx = (labels == l)
        cur_imgs = images[idx]
        num = int(args.subsample*len(cur_imgs))
        idx = np.random.choice(len(cur_imgs), size=num, replace=False)
        subsampled_img = cur_imgs[idx]
        new_split_img.append(subsampled_img)
        new_split_lbl.append([l]*num)
    new_split_img = np.concatenate(new_split_img, 0)
    new_split_lbl = np.concatenate(new_split_lbl, 0)
    print(new_split_img.shape, new_split_lbl.shape)

    np.save(os.path.join(dataset_folder, "X_%s.npy" % args.new_name), new_split_img)
    np.save(os.path.join(dataset_folder, "Y_%s.npy" % args.new_name), new_split_lbl)
