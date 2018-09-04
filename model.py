# coding=utf-8

import tensorflow as tf
import vgg_img_process
import voc2007
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--m', type=int, default=8, help='the maps for per class')
parser.add_argument('--kmax', type=int, default=0.2,help='the num of positive regions')
parser.add_argument('--kmin', type=int, default=1, help='the num of negative regions')
parser.add_argument('--alpha', type=float, default=0.6, help='the num of negative regions')

parser.add_argument('--class_num', type=int, default=20, help='the num of total class ')
parser.add_argument('--batch_size', type=int, default=16, help='batch size  ')
parser.add_argument('--num_gpus', type=int, default=4, help='the num of gpus uesed ')
parser.add_argument('--img_size', type=int, default=448, help='the num of imgs size ')

parser.add_argument('--is_training', type=bool, default=False, help='the num of imgs size ')
parser.add_argument('--train_record_path', type=str,
                    default='/home/liuweiwei02/Projects/tf_resnet/trainval.tfrecord', help='train_record_path')
parser.add_argument('--test_record_path', type=str,
                    default='/home/liuweiwei02/Projects/tf_resnet/trainval.tfrecord', help='test_record_path')

args = parser.parse_args()

slim = tf.contrib.slim


def data_load(args):

    if args.is_training:
        print('mode train')
        img, label = voc2007.read_and_decode('/home/liuweiwei02/Projects/tf_resnet/trainval.tfrecord')
        img = vgg_img_process.preprocess_image(img, args.img_size, args.img_size, True)
        img_batch, label_batch = tf.train.batch([img, label],
                                                batch_size=args.batch_size, capacity=args.batch_size * 3)
        label_batch = tf.cast(label_batch, tf.float32)
        print('traing mode: data load done')

    else:
        print('mode test')
        img, label = voc2007.read_and_decode('/home/liuweiwei02/Projects/tf_resnet/trainval.tfrecord')
        img = vgg_img_process.preprocess_image(img, args.img_size, args.img_size, False)
        img_batch, label_batch = tf.train.batch([img, label],
                                                batch_size=args.batch_size, capacity=args.batch_size * 3)
        label_batch = tf.cast(label_batch, tf.float32)
        print('testing mode: data load done')

    return img_batch, label_batch




