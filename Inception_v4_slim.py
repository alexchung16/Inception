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

    def __init__(self, input_shape, num_classes, batch_size, num_samples_per_epoch, num_epoch_per_decay,
                 decay_rate, learning_rate, keep_prob=0.8, regular_weight_decay=0.00004, batch_norm_decay=0.9997,
                 batch_norm_epsilon=0.001, batch_norm_scale=False,batch_norm_fused=True, is_pretrain=False,
                 reuse=tf.AUTO_REUSE):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.decay_steps = int(num_samples_per_epoch / batch_size * num_epoch_per_decay)
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.is_pretrain = is_pretrain
        self.reuse = reuse
        self.regular_weight_decay = regular_weight_decay
        self.batch_norm_decay = batch_norm_decay
        self.batch_norm_epsilon = batch_norm_epsilon
        self.batch_norm_scale = batch_norm_scale
        self.batch_norm_fused = batch_norm_fused
        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        # add placeholder (X,label)
        self.raw_input_data = tf.compat.v1.placeholder(tf.float32,
                                                       shape=[None, input_shape[0], input_shape[1], input_shape[2]],
                                                       name="input_images")
        # y [None,num_classes]
        self.raw_input_label = tf.compat.v1.placeholder(tf.float32, shape=[None, self.num_classes], name="class_label")
        self.is_training = tf.compat.v1.placeholder_with_default(input=False, shape=(), name='is_training')

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")

        # logits
        self.logits = self.inference(self.raw_input_data, scope='InceptionV4')
        # # computer loss value
        self.loss = self.losses(labels=self.raw_input_label, logits=self.logits, scope='Loss')
        # train operation
        self.train = self.training(self.learning_rate, self.global_step, loss=self.loss)
        self.train_accuracy = self.evaluate_batch(self.logits, self.raw_input_label) / batch_size


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
                                 reuse = self.reuse,
                                 scope=scope,
                                 is_training = self.is_training
                                 )
        return prop

    def inception_v4(self, inputs, scope='InceptionV4', num_classes=10, keep_prob=0.8,
                     reuse=None, is_training=False):
        """
        inception v4
        :return:
        """

        batch_norm_params = {
            'is_training': is_training,
            # Decay for the moving averages.
            'decay': self.batch_norm_decay,
            # epsilon to prevent 0s in variance.
            'epsilon': self.batch_norm_epsilon,
            # collection containing update_ops.
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            # use gamma for update
            'scale': self.batch_norm_scale,
            # use fused batch norm if possible.
            'fused': self.batch_norm_fused,
        }
        with tf.compat.v1.variable_scope(scope, default_name='InceptionV4', values=[inputs], reuse=reuse) as scope:
            # Set weight_decay for weights in Conv and FC layers.
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(self.regular_weight_decay)):
                with slim.arg_scope(
                        [slim.conv2d],
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as sc:
                    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
                        net = self.inception_v4_base(inputs=inputs, scope=scope)
                        with tf.variable_scope('Logits'):
                            # 8 x 8 x 1536
                            kernel_size = net.get_shape()[1:3] # 8 x 8
                            net = slim.avg_pool2d(net, kernel_size, padding='VALID', scope='AvgPool_1a')
                            # 1 x 1 x 1536
                            net = slim.dropout(inputs=net, keep_prob=keep_prob, scope='Dropout_1b')
                            net = slim.flatten(inputs=net, scope='PreLogitsFlatten')
                            # (, 1536)
                            net = slim.fully_connected(inputs=net, num_outputs=512, scope='fcn_1c')
                            # (, 512)
                            logits = slim.fully_connected(inputs=net, num_outputs=num_classes, scope='Logits')
                            # (, num_classes)
                            prop = slim.softmax(logits=logits, scope='Softmax')
                            return prop

    def inception_v4_base(self, inputs, scope='InceptionV4'):
        """
        inception V4 base
        :param inputs:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope, default_name='InceptionV4', values=[inputs]):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
                # 299 x 299 x 3
                net = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=[3, 3], stride=2,
                                  scope='Conv2d_1a_3x3')

                # 149 x 149 x 32
                net = slim.conv2d(inputs=net, num_outputs=32, kernel_size=[3, 3], stride=1,
                                  scope='Conv2d_2a_3x3')
                # 147 x 147 x 32
                net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[3, 3], stride=1,
                                  scope='Conv2d_2b_3x3',
                                  padding='SAME')
                # 147 x 147 x 64
                with tf.variable_scope('Mixed_3a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                                   scope='MaxPool_0a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 96, [3, 3], stride=2, padding='VALID',
                                               scope='Conv2d_0a_3x3')
                    net = tf.concat(axis=3, values=[branch_0, branch_1])
                # 73 x 73 x 160
                with tf.variable_scope('Mixed_4a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_0 = slim.conv2d(branch_0, 96, [3, 3], padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1', padding='SAME')
                        branch_1 = slim.conv2d(branch_1, 64, [1, 7], scope='Conv2d_0b_1x7', padding='SAME')
                        branch_1 = slim.conv2d(branch_1, 64, [7, 1], scope='Conv2d_0c_7x1', padding='SAME')
                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_1a_3x3', padding='VALID')
                    net = tf.concat(axis=3, values=[branch_0, branch_1])
                # 71 x 71 x 192
                with tf.variable_scope('Mixed_5a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [3, 3], stride=2, padding='VALID',
                                               scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                                   scope='MaxPool_1a_3x3')
                    net = tf.concat(axis=3, values=[branch_0, branch_1])
                # inception_v4_block_a x 4
                # 35 x 35 x 384
                net = self.inception_v4_block_a(inputs=net, scope='Mixed_5b')
                net = self.inception_v4_block_a(inputs=net, scope='Mixed_5c')
                net = self.inception_v4_block_a(inputs=net, scope='Mixed_5d')
                net = self.inception_v4_block_a(inputs=net, scope='Mixed_5e')
                
                # inception_v4_block_reduce_a x 1
                # 35 x 35 x 384
                net = self.inception_v4_block_reduce_a(inputs=net, scope='Mixed_6a')
                # inception_v4_block_b x 7
                # 17 x 17 x 1024
                net  = self.inception_v4_block_b(inputs=net, scope='Mixed_6b')
                net = self.inception_v4_block_b(inputs=net, scope='Mixed_6c')
                net = self.inception_v4_block_b(inputs=net, scope='Mixed_6d')
                net = self.inception_v4_block_b(inputs=net, scope='Mixed_6e')
                net = self.inception_v4_block_b(inputs=net, scope='Mixed_6f')
                net = self.inception_v4_block_b(inputs=net, scope='Mixed_6g')
                net = self.inception_v4_block_b(inputs=net, scope='Mixed_6h')

                # inception_v4_block_reduce_b x 1
                # 17 x 17 x 1024
                net = self.inception_v4_block_reduce_b(inputs=net, scope='Mixed_7a')
                # inception_v4_block_c x 3
                # 8 x 8 x 1526
                net = self.inception_v4_block_c(inputs=net, scope='Mixed_7b')
                net = self.inception_v4_block_c(inputs=net, scope='Mixed_7c')
                net = self.inception_v4_block_c(inputs=net, scope='Mixed_7d')
                
                return net

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

    def inception_v4_arg_scope(weight_decay=0.00004,
                               use_batch_norm = True,
                               batch_norm_decay=0.9997,
                               batch_norm_epsilon=0.001,
                               activation_fn = tf.nn.relu,
                               batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
                               batch_norm_scale = False
                               ):
        """Defines the default InceptionV4 arg scope.
        """
        batch_norm_params = {
            # Decay for the moving averages.
            'decay': batch_norm_decay,
            # epsilon to prevent 0s in variance.
            'epsilon': batch_norm_epsilon,
            # collection containing update_ops.
            'updates_collections': batch_norm_updates_collections,
            # use fused batch norm if possible.
            'fused': None,
            # use gamma for update
            'scale': batch_norm_scale,
        }
        if use_batch_norm:
            normalizer_fn = slim.batch_norm
            normalizer_params = batch_norm_params
        else:
            normalizer_fn = None
            normalizer_params = {}
        # Set weight_decay for weights in Conv and FC layers.
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope(
                    [slim.conv2d],
                    weights_initializer=slim.variance_scaling_initializer(),
                    activation_fn=activation_fn,
                    normalizer_fn=normalizer_fn,
                    normalizer_params=normalizer_params) as sc:
                return sc

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
                    branch_1 = slim.conv2d(inputs=branch_1, num_outputs=96, kernel_size=[3, 3], stride=1,
                                           scope='Conv2d_0b_3x3')
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

    def inception_v4_block_reduce_a(self, inputs, scope=None, reuse=None):
        """
        inception v4 block_reduce_a

        :param inputs:
        :param scope:
        :param reuse:
        :return:
        """
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],stride=1, padding='SAME'):
            with tf.variable_scope(scope, 'BlockReductionA', [inputs], reuse=reuse):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs, num_outputs=384, kernel_size=[3, 3], stride=2, padding='VALID',
                                           scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs, num_outputs=192, kernel_size=[1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, num_outputs=224, kernel_size=[3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, num_outputs=256, kernel_size=[3, 3], stride=2, padding='VALID', 
                                           scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(inputs, kernel_size=[3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])

    def inception_v4_block_reduce_b(self, inputs, scope=None, reuse=None):
        """
        inception v4 block_reduce_b

        :param inputs:
        :param scope:
        :param reuse:
        :return:
        """
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope(scope, 'BlockReductionB', [inputs], reuse=reuse):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs, num_outputs=192, kernel_size=[1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, num_outputs=192, kernel_size=[3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs, num_outputs=256, kernel_size=[1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, num_outputs=256, kernel_size=[1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, num_outputs=320, kernel_size=[7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, num_outputs=320, kernel_size=[3, 3], stride=2, padding='VALID', 
                                           scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(inputs, kernel_size=[3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
            
    def training(self, learnRate, globalStep, loss):
        """
        train operation
        :param learnRate:
        :param globalStep:
        :param args:
        :return:
        """
        # define trainable variable
        trainable_variable = None
        # trainable_scope = self.trainable_scope
        trainable_scope = ['InceptionV4/Logits/fcn_1c', 'InceptionV4/Logits/fcn_1c']
        if self.is_pretrain:
            trainable_variable = []
            if trainable_scope is not None:
                for scope in trainable_scope:
                    variables = tf.model_variables(scope=scope)
                    [trainable_variable.append(var) for var in variables]
            else:
                trainable_variable = None
        learning_rate = tf.train.exponential_decay(learning_rate=learnRate, global_step=globalStep,
                                                   decay_steps=self.decay_steps, decay_rate=self.decay_rate,
                                                   staircase=False)
        # update moving_mean and moving_variance when training
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op =  tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=globalStep,
                                                                       var_list=trainable_variable)
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