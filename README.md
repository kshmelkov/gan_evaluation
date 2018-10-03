# How good is my GAN?

This is code release for our paper ["How good is my GAN?"](https://arxiv.org/abs/1807.09499) published on ECCV 2018.

## Requirements

Code is written for Python 3.5 and TensorFlow 1.8 (might require minor modifications for more recent versions). You are also expected to have normal scientific stack installed: NumPy, SciPy, Matplotlib. Among auxillary packages install easydict

## Datasets

To launch experiments on CIFAR10/100 you should download and convert it first. Run the following command (it will download both CIFAR10 and CIFAR100):
```
python data/download_and_convert_cifar10.py
```

Then you need to download Inception and compute Inception statistics:
```
python evaluation/inception.py --dataset=cifar10
```

With ImageNet it is a bit more complicated, you have to register to download images and then reconvert them to tfrecord first if you want fast loading (related functions are collected in `data/imagenet.py`), but I don't remember all details now. Alternatively, you can use a function `data/imagenet.py:load_dataset()` instead of converting to tfrecord (it reads individual files directly), but it will still require minor modifications of input pipeline.

## Training

First you need to train a vanilla classifier (replace cifar10 by cifar100 if needed):

```
python classifier.py --run_name=cifar10_classifier_ms_decay --dataset=cifar10 --image_size=32
```

To train SNGAN and evaluate it, run the following command:

```
python gan.py --action=train_gan,generate,train_all_classifiers --dataset=cifar10 --run_name=sngan --sum_pooling --gen_depth=256 --gen_linear_dim=1024 --gan_loss=hinge --spectral_normalization --conditional_bn --projection --acgan_dw=0 --acgan_gw=0 --gradient_penalty=0 --learning_rate=2e-4 --adam_beta1=0.0 --adam_beta2=0.9 --num_discriminator_steps=5 --arch=resnet --lr_decay --bn_decay=0.9 --batch_size=64
```

To train WGAN-GP and evaluate it, run the following command:
```
python gan.py --action=train_gan,generate,train_all_classifiers --dataset=cifar10 --run_name=wgangp --gan_loss=wasserstein --acgan_dw=1.0 --acgan_gw=0.1 --gradient_penalty=10 --learning_rate=2e-4 --adam_beta1=0.0 --adam_beta2=0.9 --num_discriminator_steps=5 --arch=resnet --lr_decay
```

To train DCGAN and evaluate it, run the following command:
```
python gan.py --action=train_gan,generate,train_all_classifiers --dataset=cifar10 --run_name=dcgan ---gan_loss=classical --acgan_dw=1.0 --acgan_gw=0.1 --gradient_penalty=0 --learning_rate=2e-4 --adam_beta1=0.5 --adam_beta2=0.999 --num_discriminator_steps=1 --arch=dcgan --batch_size=100 --max_iterations=50000 --noise_dim=100
```
