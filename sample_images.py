import numpy as np
import argparse
from PIL import Image

from data import cifar

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and cache Inception activations')
    parser.add_argument("--dataset", default='cifar10', choices=['imagenet', 'cifar10', 'cifar100'])
    parser.add_argument("--split", default='train', type=str)
    parser.add_argument("--num_images", default=16, type=int)
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--grid_h", default=4, type=int)
    parser.add_argument("--grid_w", default=4, type=int)
    args = parser.parse_args()

    split = args.split
    if 'gan' in args.split:
        split = "gan_100_%s" % args.split

    images, labels = cifar.load_cifar(split, 100, dataset=args.dataset,
                                      return_numpy=True,
                                      augmentation=False,
                                      shuffle=False, normalize=False)

    selected_images = []
    for n in range(args.num_images):
        cls_img = images[labels == (n % args.num_classes)]
        idx = np.random.randint(len(cls_img))
        selected_images.append(cls_img[idx])
    images = np.stack(selected_images, 0)
    images = images.astype(np.uint8)
    im_sz = 32

    assert args.num_images == args.grid_h * args.grid_w
    arr = np.zeros(dtype=np.uint8, shape=(args.grid_h*im_sz, args.grid_w*im_sz, 3))
    for k in range(args.num_images):
        i = k // args.grid_h
        j = k % args.grid_h
        arr[i*im_sz:(i+1)*im_sz, j*im_sz:(j+1)*im_sz] = images[k]

    img = Image.fromarray(arr)
    img.save("%s_%s.png" % (args.dataset, args.split))

