#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File Inception_v4_slim.py
# @ Description :
# @ Author alexchung
# @ Time 12/11/2019 PM 19:09


import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

class InceptionV4():
    """
    Inception v1
    """
    def __init__(self, input_shape, num_classes, batch_size, decay_steps, decay_rate, learning_rate,
                 keep_prob=0.8, global_pool=False, spacial_squeeze=True, reuse=tf.AUTO_REUSE):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.global_pool = global_pool
        self.spacial_squeeze = spacial_squeeze
        self.reuse = reuse

        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        # add placeholder (X,label)
        self.raw_input_data = tf.compat.v1.placeholder (tf.float32, shape=[None, input_shape[0], input_shape[1], input_shape[2]], name="input_images")
        # y [None,num_classes]
        self.raw_input_label = tf.compat.v1.placeholder (tf.float32, shape=[None, self.num_classes], name="class_label")
        self.is_training = tf.compat.v1.placeholder_with_default(input=False, shape=(), name='is_training')

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")

        # logits
        self.logits =  self.inference(self.raw_input_data, scope='InceptionV3')
        # # computer loss value
        self.loss = self.losses(labels=self.raw_input_label, logits=self.logits, scope='Loss')
        # train operation
        self.train = self.training(self.learning_rate, self.global_step, loss=self.loss)
        self.evaluate_accuracy = self.evaluate_batch(self.logits, self.raw_input_label) / batch_size


    def inference(self, inputs, scope='InceptionV4'):
        """
        Inception V4 net structure
        :param inputs:
        :param scope:
        :return:
        """
        self.prameter = []
        prop = self.inception_v4(inputs=inputs,
                                 num_classes=self.num_classes,
                                 keep_prob=self.keep_prob,
                                 global_pool=self.global_pool,
                                 spatial_squeeze=self.spacial_squeeze,
                                 reuse = self.reuse,
                                 scope=scope,
                                 is_training = self.is_training
                                 )
        return prop

    def inception_v4(self, inputs, scope='InceptionV4', num_classes=10, keep_prob=0.8, global_pool=False,
                     spatial_squeeze=True, reuse=None, is_training=False):
        """
        inception v4
        :return:
        """
        batch_norm_params = {
            'is_training': is_training,
            'decay': 0.9,
            'epsilon': 0.01,
            'scale': True
        }
        with tf.compat.v1.variable_scope(scope, default_name='InceptionV4', values=[inputs], reuse=reuse) as scope:
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                with slim.arg_scope([slim.conv2d],
                                    weights_regularizer=slim.l2_regularizer(2e-4),
                                    weights_initializer=slim.xavier_initializer(),
                                    normalizer_fn = slim.batch_norm,
                                    normalizer_params = batch_norm_params
                                    ):
                    pass


    def reduce_kernel_size(self, input_op, kernel_size):
        """
        automatically reduce kernel size
        :param input_op:
        :param kernel_size:
        :return:
        """
        shape = input_op.get_shape().as_list()
        if shape[1] is None or shape[2] is None:
            kernel_size_out = kernel_size
        else:
            kernel_size_out = [min(shape[1], kernel_size[0],),
                               min(shape[2], kernel_size[1])]
        return kernel_size_out

    def inception_v4_base(self, inputs, scope='InceptionV4'):
        """
        inception V3 base
        :param inputs:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope, default_name='InceptionV3', values=[inputs]):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
                # 299 x 299 x 3
                net = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=[3, 3], stride=2, scope='Conv2d_1a_3x3')

                # 149 x 149 x 32
                net = slim.conv2d(inputs=net, num_outputs=32, kernel_size=[3, 3], stride=1, scope='Conv2d_2a_3x3')
                # 147 x 147 x 32
                net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[3, 3], stride=1, scope='Conv2d_2b_3x3',
                                  padding='SAME')
                # 147 x 147 x 64
                with tf.variable_scope('Mixed_3a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_0a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 96, [3, 3], stride=2, padding='VALID', scope='Conv2d_0a_3x3')
                    net = tf.concat(axis=3, values=[branch_0, branch_1])
                # 73 x 73 x 160
                with tf.variable_scope('Mixed_4a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_0 = slim.conv2d(branch_0, 96, [3, 3], padding='VALID', cope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 64, [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 64, [7, 1], scope='Conv2d_0c_7x1')
                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], padding='VALID', scope='Conv2d_1a_3x3')
                    net = tf.concat(axis=3, values=[branch_0, branch_1])
                # 71 x 71 x 192
                with tf.variable_scope('Mixed_5a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                    net = tf.concat(axis=3, values=[branch_0, branch_1])
                # 35 x 35 x 384


    def inception_v4_block_a(self, inputs, scope, reuse=None):
        """
        inception v4 block_a
        :param inputs:
        :param scope:
        :param reuse:
        :return:
        """
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope(scope, 'BlockInceptionA', [inputs], reuse=reuse):
                # branch 0
                with tf.compat.v1.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs=inputs, num_outputs=96, kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                # branch 1
                with tf.compat.v1.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs=inputs, num_outputs=64, kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(inputs=branch_1, num_outputs=96, kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0b_5x5')
                # branch 2
                with tf.compat.v1.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(inputs=inputs, num_outputs=64, kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=96, kernel_size=[3, 3], stride=1,
                                           scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=96, kernel_size=[3, 3], stride=1,
                                           scope='Conv2d_0c_3x3')
                # branch 3
                with tf.compat.v1.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(inputs=inputs, kernel_size=[3, 3], stride=1, scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(inputs=branch_3, num_outputs=96, kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3)
                return net

    def inception_v4_block_b(self, inputs, scope, reuse=None):
        """
        inception v4 block_b
        :param inputs:
        :param scope:
        :param reuse:
        :return:
        """
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope(scope, 'BlockInceptionB', [inputs], reuse=reuse):
                # branch 0
                with tf.compat.v1.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs=inputs, num_outputs=384, kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                # branch 1
                with tf.compat.v1.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs=inputs, num_outputs=192, kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(inputs=branch_1, num_outputs=224, kernel_size=[1, 7], stride=1,
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(inputs=branch_1, num_outputs=256, kernel_size=[7, 1], stride=1,
                                           scope='Conv2d_0c_7x1')
                # branch 2
                with tf.compat.v1.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(inputs=inputs, num_outputs=192, kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=192, kernel_size=[7, 1], stride=1,
                                           scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=224, kernel_size=[1, 7], stride=1,
                                           scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=224, kernel_size=[7, 1], stride=1,
                                           scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=256, kernel_size=[1, 7], stride=1,
                                           scope='Conv2d_0e_1x7')
                # branch c
                with tf.compat.v1.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(inputs=inputs, kernel_size=[3, 3], stride=1, scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(inputs=branch_3, num_outputs=128, kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3)
                return net

    def inception_v4_block_c(self, inputs, scope, reuse=None):
        """
        inception v4 block_c
        :param inputs:
        :param scope:
        :param reuse:
        :return:
        """
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope(scope, 'BlockInceptionC', [inputs], reuse=reuse):
                with tf.compat.v1.variable_scope('Branch_0'):
                    # branch_0
                    branch_0 = slim.conv2d(inputs=inputs, num_outputs=256, kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                with tf.compat.v1.variable_scope('Branch_1'):
                    # branch_1
                    branch_1 = slim.conv2d(inputs=inputs, num_outputs=384, kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                    branch_1_1 = slim.conv2d(inputs=branch_1, num_outputs=256, kernel_size=[1, 3], stride=1,
                                             scope='Conv2d_0b_1x3')
                    branch_1_2 = slim.conv2d(inputs=branch_1, num_outputs=256, kernel_size=[3, 1], stride=1,
                                             scope='Conv2d_0c_3x1')
                    branch_1 = tf.concat(values=[branch_1_1, branch_1_2], axis=3)

                with tf.compat.v1.variable_scope('Branch_2'):
                    # branch_1
                    branch_2 = slim.conv2d(inputs=inputs, num_outputs=384, kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=448, kernel_size=[3, 1], stride=1,
                                           scope='Conv2d_0b_3x1')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=512, kernel_size=[1, 3], stride=1,
                                           scope='Conv2d_0c_1x3')
                    branch_2_1 = slim.conv2d(inputs=branch_2, num_outputs=256, kernel_size=[1, 3], stride=1,
                                             scope='Conv2d_0d_1x3')
                    branch_2_2 = slim.conv2d(inputs=branch_2, num_outputs=256, kernel_size=[3, 1], stride=1,
                                             scope='Conv2d_0e_3x1')
                    branch_2 = tf.concat(values=[branch_2_1, branch_2_2], axis=3)
                with tf.compat.v1.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(inputs=inputs, kernel_size=[3, 3], stride=1, scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(inputs=branch_3, num_outputs=256, kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3)
                return net

    def training(self, learnRate, globalStep, loss):
        """
        train operation
        :param learnRate:
        :param globalStep:
        :param args:
        :return:
        """
        learning_rate = tf.train.exponential_decay(learning_rate=learnRate, global_step=globalStep,
                                                   decay_steps=self.decay_steps, decay_rate=self.decay_rate,
                                                   staircase=False)
        # update moving_mean and moving_variance when training
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op =  tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=globalStep)
        return train_op

    def losses(self, logits, labels, scope='Loss'):
        """
        loss function
        :param logits:
        :param labels:
        :return:
        """
        with tf.name_scope(scope) as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='Entropy')
            return tf.reduce_mean(input_tensor=cross_entropy, name='Entropy_Mean')

    def evaluate_batch(self, logits, labels, scope='Evaluate_Batch'):
        """
        evaluate one batch correct num
        :param logits:
        :param label:
        :return:
        """
        with tf.name_scope(scope):
            correct_predict = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            return tf.reduce_sum(tf.cast(correct_predict, dtype=tf.float32))

    def fill_feed_dict(self, image_feed, label_feed, is_training):
        feed_dict = {
            self.raw_input_data: image_feed,
            self.raw_input_label: label_feed,
            self.is_training: is_training
        }
        return feed_dict