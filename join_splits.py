import numpy as np
import argparse
import os
import glob
from paths import GPU_DATASETS

def np_flat_map(src, func):
    return np.concatenate([func(s) for s in src], 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--total_splits", type=int, required=True)

    args = parser.parse_args()

    X_mask = os.path.join(GPU_DATASETS, args.dataset,
                          "X_gan_100_%s_split_?_out_of_%i_setA.npy" % (args.run_name, args.total_splits))
    Y_mask = os.path.join(GPU_DATASETS, args.dataset,
                          "Y_gan_100_%s_split_?_out_of_%i_setA.npy" % (args.run_name, args.total_splits))

    x_list = sorted(glob.glob(X_mask))
    y_list = sorted(glob.glob(Y_mask))

    x = np_flat_map(x_list, np.load)
    x = x.astype(np.uint8)
    y = np_flat_map(y_list, np.load)

    np.save(os.path.join(GPU_DATASETS, args.dataset, "X_gan_100_%s_%i_splits" % (args.run_name, args.total_splits)), x)
    np.save(os.path.join(GPU_DATASETS, args.dataset, "Y_gan_100_%s_%i_splits" % (args.run_name, args.total_splits,)), y)

