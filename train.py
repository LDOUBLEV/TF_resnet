# coding=utf-8

import tensorflow as tf
import numpy as np

import resnet_utils
import resnet_v1_101
import vgg_img_process
import matplotlib.pyplot as plt
import os
import time
import model as resnet_wildcat

from tensorflow.python.ops import control_flow_ops

try:
    import urllib2
except ImportError:
    import urllib.request as urllib

import multi_gpu

args = resnet_wildcat.args

slim = tf.contrib.slim

gpus = [0, 1, 2]  # Use GPU 0, 1, 2
os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(i) for i in gpus])

num_gpus = len(gpus)  # number of GPUs to use

class Train(object):
    def __init__(self, sess, args, training):
        self.sess = sess

        # super(wildcat.WildcatPooling, self).__init__(args)
        # wildcat.WildcatPooling.__init__(self, args)

        self.m = args.m
        self.is_img_process = training
        self.class_num = args.class_num
        self.alpha =args.alpha

        self.img_size = 448
        self.batch_size = 16*num_gpus

        self.max_step = int(5011/self.batch_size)*20
        self.cross_entropy = 0.
        self.iters_per_epoch = int(5011/(args.batch_size*num_gpus))

        self.build_model()

    def multi_label_cross_entropy_loss(self, predictions, labels):

        epsilon = 1e-5
        labels = tf.cast(labels, tf.float32)

        loss = labels*tf.log(predictions+epsilon) + (1-labels)*tf.log(1-predictions + epsilon)

        return -tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    def build_model(self):

        self.is_training = tf.placeholder(tf.bool, [])

        self.step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0.0), trainable=False)
        lr = tf.train.exponential_decay(learning_rate=1e-2, global_step=self.step, decay_steps=10000, decay_rate=0.1,
                                        staircase=True)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        # opt_init = tf.train.GradientDescentOptimizer(learning_rate=lr)
        labels_all = []

        tower_grads = []
        eval_logits = []

        with tf.variable_scope(tf.get_variable_scope()):

            for i in range(num_gpus):
                print('\n num gpu:{}\n'.format(i))
                with tf.device('/gpu:%d' % i), tf.name_scope('%s_%d' % ("classification", i)) as scope:
                    imgs_batch, label_batch = resnet_wildcat.data_load(args)
                    labels_all.append(label_batch)

                    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
                        logits, end_points, net_conv5 = resnet_v1_101.resnet_v1_101(imgs_batch,
                                                                                 num_classes=args.class_num,
                                                                                 is_training=args.is_training,
                                                                                 global_pool=True,
                                                                                 output_stride=None,
                                                                                 spatial_squeeze=True,
                                                                                 store_non_strided_activations=False,
                                                                                 reuse=None,
                                                                                 scope='resnet_v1_101')

                    tf.losses.sigmoid_cross_entropy(label_batch, logits)

                    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                    # updates_op = tf.group(*update_ops)
                    # with tf.control_dependencies([updates_op]):
                    #     cross_entropy = tf.identity(cross_entropy, name='train_op')

                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                    updates_op = tf.group(*update_ops)
                    with tf.control_dependencies([updates_op]):
                        losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
                        total_loss = tf.add_n(losses, name='total_loss')

                    # if update_ops:
                    #     updates = tf.group(*update_ops)
                    #     cross_entropy = control_flow_ops.with_dependencies([updates], cross_entropy)

                    # reuse var
                    tf.get_variable_scope().reuse_variables()
                    # just an assertion!
                    assert tf.get_variable_scope().reuse == True

                    # grad compute
                    # if args.is_training:
                    grads = optimizer.compute_gradients(total_loss)
                    # important!!!  logits/biases is None but not tensor, no gradient in it
                    new_grads = []
                    for gv in grads:
                        if gv[0] is not None:
                            new_grads.append(gv)

                    tower_grads.append(new_grads)
                    eval_logits.append(tf.nn.sigmoid(logits))

        # We must calculate the mean of each gradient
        # if training:
        grads = multi_gpu.average_gradients(tower_grads)
        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=self.step)
        # Group all updates to into a single train op.
        self.train_op = tf.group(apply_gradient_op)

        self.prediction = tf.concat(eval_logits, axis=0)

        self.cross_entropy = total_loss
        self.label_batch = tf.concat(labels_all, axis=0)

        merged_summary_op = tf.summary.merge_all()

        # load weights// frist to initializer all vars
        init = tf.global_variables_initializer()  # tf.variables_initializer(var_list=initvars)
        self.sess.run(init)

        all_variables = tf.global_variables()
        # for var in all_variables:
        #     print(var)

        # init vars
        load_vars = [v for v in all_variables if 'step' not in v.name]
        self.saver = tf.train.Saver(var_list=load_vars)

        frist_load_model = False
        if frist_load_model:
            resnet101_model_path = '/home/liuweiwei02/Projects/resnet_v1_101.ckpt'
            exclude = ['resnet_v1_101/logits']
            resnet_vars = slim.get_variables_to_restore(include=['resnet_v1_101'], exclude=exclude)

            init_fn = slim.assign_from_checkpoint_fn(resnet101_model_path, resnet_vars)
            init_fn(sess)
            print('resnet_model load done. \n')

    def train(self):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print('starting to train ')

        if not os.path.exists('model_saved_new'):
            os.mkdir('model_saved_new')

        pre_list = []
        label_list = []
        try:
            while not coord.should_stop():

                _, step, wsl_loss = self.sess.run([self.train_op, self.step,
                                                                 self.cross_entropy])

                wsl_pre, real_label = self.sess.run([self.prediction, self.label_batch])

                pre_list.append(np.around(wsl_pre))
                label_list.append(np.around(real_label))

                print('[%d/%d]  wsl_loss: %.4f' %
                      (int(step), self.max_step,  wsl_loss))

                # print the prediction results and the real_label
                if step % 100 == 0 :
                    print('step is big than 100')

                if step%1000==0:
                    print('model saving !!!!!')
                    self.saver.save(self.sess,
                                    os.getcwd() + '/model_saved_new/voc2007model_{}.ckpt'.format(int(step//1000)))
                    print('model save done...')

                if step %self.iters_per_epoch==0 and step!=0:
                    pre = np.concatenate(pre_list, axis=0)
                    label_ = np.concatenate(label_list, axis=0)

                    print('MAP is:', map, '\n')
                    pre_list = []
                    label_list = []

                if step>=self.max_step:
                    print('traing done')
                    break

        finally:
            # save_path = self.saver.save(sess, os.getcwd()+'/model_saved_new/final2007_model.ckpt')
            # print("Model saved in file: %s" % save_path)
            print('\n this epoch end \n')
            coord.request_stop()

        coord.join(threads)
        print('train run done')

    def test(self, test_iters):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print('starting to test ')

        arf = np.zeros(shape=(3))

        AP = np.zeros(shape=(self.class_num))

        pre_list = []
        label_list = []

        for step in range(test_iters):
            wsl_pre, real_label = self.sess.run(
                [self.prediction, self.label_batch], feed_dict={self.is_training:False})

            pre_list.append( np.around(wsl_pre))
            label_list.append(np.around(real_label))

        pre_list = np.concatenate(pre_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)
        # calculate the acc
        #
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    training = False

    if training:
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            Train_model = Train(sess, args, True)
            Train_model.train()

    else:
        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.allow_growth = True

        with tf.Session() as sess:

            Train_model = Train(sess, args, training)

            Train_model.test(4952//Train_model.batch_size)


