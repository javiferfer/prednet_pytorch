import argparse
import torch
import os
import sys

from distutils.util import strtobool


sys.path.append('.')
from src.test.test import test

os.environ['KMP_DUPLICATE_LIB_OK']='True'


parser = argparse.ArgumentParser(description='PredNet')
parser.add_argument('--images', '-i', default='data/train_list.txt', help='Path to image list file')
parser.add_argument('--sequences', '-seq', default='', help='Path to sequence list file')
parser.add_argument('--device', '-d', default="", type=str,
                    help='Computational device')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of sequence and image files')
parser.add_argument('--initmodel', default='',
                    help='Initialize the model from given file')
parser.add_argument('--size', '-s', default='160,120',
                    help='Size of target images. width,height (pixels)')
parser.add_argument('--channels', '-c', default='3,48,96,192',
                    help='Number of channels on each layers')
parser.add_argument('--offset', '-o', default='0,0',
                    help='Center offset of clipping input image (pixels)')
parser.add_argument('--input_len', '-l', default=20, type=int,
                    help='Input frame length fo extended prediction on test (frames)')
parser.add_argument('--ext', '-e', default=10, type=int,
                    help='Extended prediction on test (frames)')
parser.add_argument('--bprop', default=20, type=int,
                    help='Back propagation length (frames)')
parser.add_argument('--save', default=10000, type=int,
                    help='Period of save model and state (frames)')
parser.add_argument('--period', default=1000000, type=int,
                    help='Period of training (frames)')
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--saveimg', dest='saveimg', action='store_true')
parser.add_argument('--useamp', dest='useamp', action='store_true', help='Flag for using AMP')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate')
parser.add_argument('--lr_rate', default=1.0, type=float,
                    help='Reduction rate for Step lr scheduler')
parser.add_argument('--min_lr', default=0.0001, type=float,
                    help='Lower bound learning rate for Step lr scheduler')
parser.add_argument('--batchsize', default=1, type=int, help='Input batch size')
parser.add_argument('--shuffle', default=False, type=strtobool, help=' True is enable to sampl data randomly (default: False)')
parser.add_argument('--num_workers', default=0, type=int, help='Num. of dataloader process. (default: num of cpu cores')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='True is enable to log for Tensorboard')
parser.add_argument('--up_down_up', action='store_true', help='True is enable to cycle up-down-up in order')
parser.add_argument('--color_space', default='RGB', type=str, help='Image color space(RGB, HSV, LAB, CMYK, YcbCr) - the dimension of this color space and 1st channel must be same.')
parser.add_argument('--loss', type=str, default='mse', help='Loss name for training. Please select loss from "mse", "corr_wise", and "ensemble" (default: mse).')
parser.add_argument('--amp', default=0.0, type=float, help='Amplitude for sine function')
parser.add_argument('--omg', default=1.0, type=float, help='Angular velocity for sine function')
parser.set_defaults(test=False)
args = parser.parse_args()


if __name__ == '__main__':

    args.size = args.size.split(',')
    for i in range(len(args.size)):
        args.size[i] = int(args.size[i])
    args.channels = args.channels.split(',')
    for i in range(len(args.channels)):
        args.channels[i] = int(args.channels[i])

    if args.device == '':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    test(images=args.images,
        sequences=args.sequences,
        root=args.root,
        tensorboard=args.tensorboard,
        channels=args.channels,
        initmodel=args.initmodel,
        useamp=args.useamp,
        size=args.size,
        color_space=args.color_space,
        batchsize=args.batchsize,
        num_workers=args.num_workers,
        input_len=args.input_len,
        ext=args.ext,
        device=device)

