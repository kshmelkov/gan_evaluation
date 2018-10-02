from paths import LOGS
import argparse

parser = argparse.ArgumentParser(description='Train or eval a model train on CIFAR-100.')
parser.add_argument("--run_name", type=str, required=True)

# optimization parameters:
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--pretrained_net", default='', type=str)
parser.add_argument("--ckpt", default=0, type=int)
parser.add_argument("--action", required=True, type=str)
# parser.add_argument("--data_format", default='NHWC', choices=['NHWC', 'NCHW'])
parser.add_argument("--num_classes", default=10, type=int)
parser.add_argument("--dataset", default='cifar10', choices=['imagenet', 'cifar10', 'cifar100'])
parser.add_argument("--image_size", default=32, type=int)
parser.add_argument("--train_split", default='train', type=str)
parser.add_argument("--test_split", default='test', type=str)

parser.add_argument("--lr_decay_steps", default=[], nargs='+', type=int)
parser.add_argument("--linear_decay_start", default=0, type=int)

parser.add_argument("--print_step", default=100, type=int)
parser.add_argument("--ckpt_step", default=1000, type=int)
parser.add_argument("--summary_step", default=1000, type=int)
parser.add_argument("--inception_step", default=5000, type=int)
parser.add_argument("--inception_splits", default=10, type=int)
parser.add_argument("--inception_file", default="", type=str)

parser.add_argument("--eval_batch_size", default=100, type=int)
parser.add_argument("--num_generated_batches", default=500, type=int)
parser.add_argument("--random_labels", default=False, action='store_true')
parser.add_argument("--subsampling", default=1.0, type=float)
parser.add_argument("--generator_coef", default=1, type=float)

# general optimization setup
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--optimizer", default='adam', choices=['adam', 'rmsprop', 'sgd'])
parser.add_argument("--bn_decay", default=0.9, type=float)
parser.add_argument("--learning_rate", default=2e-4, type=float)
parser.add_argument("--adam_beta1", default=0.0, type=float)
parser.add_argument("--adam_beta2", default=0.9, type=float)
parser.add_argument("--max_iterations", default=100000, type=int)
parser.add_argument("--lr_decay", default=False, action='store_true')

# GAN general parameters
parser.add_argument("--unconditional", default=False, action='store_true')
parser.add_argument("--noise_dims", default=128, type=int)
parser.add_argument("--num_discriminator_steps", default=5, type=int)
parser.add_argument("--arch", default='resnet', choices=['resnet', 'dcgan', 'dcgan_acgan', 'dcgan_sn', 'dcgan_wgan', 'resnet128', 'resnet64'])
parser.add_argument("--gan_loss", default='wasserstein', choices=['wasserstein', 'hinge', 'classical'])

# SNGAN
parser.add_argument("--spectral_normalization", default=False, action='store_true')
parser.add_argument("--conditional_bn", default=False, action='store_true')
parser.add_argument("--projection", default=False, action='store_true')
parser.add_argument("--gen_depth", default=128, type=int)
parser.add_argument("--gen_linear_dim", default=128, type=int)
parser.add_argument("--sum_pooling", default=False, action='store_true')

# WGAN
parser.add_argument("--acgan_gw", default=0.1, type=float)
parser.add_argument("--acgan_dw", default=1.0, type=float)
parser.add_argument("--gradient_penalty", default=10.0, type=float)

# SNGAN hyperparameters:
# --sum_pooling --gen_depth=256 --gen_linear_dim=1024 --gan_loss=hinge --spectral_normalization --conditional_bn --projection --acgan_dw=0 --acgan_gw=0 --gradient_penalty=0 --learning_rate=2e-4 --adam_beta1=0.0 --adam_beta2=0.9 --num_discriminator_steps=5 --arch=resnet --lr_decay --bn_decay=0.9 --batch_size=64
# ACGAN/WGAN hyperparameters
# --gan_loss=wasserstein --acgan_dw=1.0 --acgan_gw=0.1 --gradient_penalty=10 --learning_rate=2e-4 --adam_beta1=0.0 --adam_beta2=0.9 --num_discriminator_steps=5 --arch=resnet --lr_decay
# ACGAN/DCGAN hyperparameters:
# --gan_loss=classical --acgan_dw=1.0 --acgan_gw=0.1 --gradient_penalty=0 --learning_rate=2e-4 --adam_beta1=0.5 --adam_beta2=0.999 --num_discriminator_steps=1 --arch=dcgan_acgan --batch_size=100 --max_iterations=50000 --noise_dim=100
# there is also activation noise std?

parser.add_argument("--crossentropy_loss_coef", default=1.0, type=float)
parser.add_argument("--crossentropy", default=False, action='store_true')

parser.add_argument("--num_dataset_readers", default=2, type=int)
parser.add_argument("--num_prep_threads", default=4, type=int)

parser.add_argument("--total_class_splits", default=0, type=int)
parser.add_argument("--active_split_num", default=0, type=int)

args = parser.parse_args()


from logging_config import get_logging_config
