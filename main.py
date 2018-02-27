#coding:utf-8
import argparse

import train
from vlib.load_data import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='nogan', help='DCGAN or WGAN_GP or ct-gan')
parser.add_argument('--trainable', type=bool, default=True, help='True for train and False for test')
parser.add_argument('--load_model', type=bool, default=False, help='True for load ckpt model and False for otherwise')
parser.add_argument('--label_num', type=int, default=2, help='the num of labled images we use， 2*100=200，batchsize:100')
args = parser.parse_args()

def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    sess = tf.InteractiveSession(config=config)
    model = train.Train(sess, args)
    if args.trainable:
        model.train()
    else:
        print(model.test())

if __name__ == '__main__':
    main()
