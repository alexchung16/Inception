#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File Inception_v3_slim.py
# @ Description :
# @ Author alexchung
# @ Time 11/11/2019 PM 15:41

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

class InceptionV3():
    """
    Inception v1
    """
    def __init__(self, input_shape, num_classes, batch_size, num_samples_per_epoch, num_epoch_per_decay,
                 decay_rate, learning_rate, keep_prob=0.8, global_pool=False, spacial_squeeze=True,
                 reuse=tf.AUTO_REUSE):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.decay_steps = int(num_samples_per_epoch / batch_size * num_epoch_per_decay)
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


    def inference(self, inputs, scope='InceptionV3'):
        """
        Inception V3 net structure
        :param inputs:
        :param scope:
        :return:
        """
        self.prameter = []
        prop = self.inception_v3(inputs=inputs,
                                 num_classes=self.num_classes,
                                 keep_prob=self.keep_prob,
                                 global_pool=self.global_pool,
                                 spatial_squeeze=self.spacial_squeeze,
                                 reuse = self.reuse,
                                 scope=scope,
                                 is_training = self.is_training
                                 )
        return prop

    def inception_v3(self, inputs, scope='InceptionV3', num_classes=10, keep_prob=0.8, global_pool=False,
                     spatial_squeeze=True, reuse=None, is_training=False):
        """
        inception v3
        :return:
        """
        weight_decay = 0.00004
        batch_norm_var_collection = 'moving_vars'
        batch_norm_decay = 0.9997
        batch_norm_epsilon = 0.01
        updates_collections = ops.GraphKeys.UPDATE_OPS
        use_fused_batchnorm = True

        batch_norm_params = {
            'is_training': is_training,
            # Decay for the moving averages.
            'decay': batch_norm_decay,
            # epsilon to prevent 0s in variance.
            'epsilon': batch_norm_epsilon,
            # collection containing update_ops.
            'updates_collections': updates_collections,
            # Use fused batch norm if possible.
            'fused': use_fused_batchnorm,
            # collection containing the moving mean and moving variance.
            'variables_collections': {
                'beta': None,
                'gamma': None,
                'moving_mean': [batch_norm_var_collection],
                'moving_variance': [batch_norm_var_collection],
            }
        }
        with tf.compat.v1.variable_scope(scope, default_name='InceptionV3', values=[inputs], reuse=reuse) as scope:
            # Set weight_decay for weights in Conv and FC layers.
            with arg_scope(
                    [layers.conv2d, layers_lib.fully_connected],
                    weights_regularizer=regularizers.l2_regularizer(weight_decay)):
                with arg_scope(
                        [layers.conv2d],
                        weights_initializer=initializers.variance_scaling_initializer(),
                        activation_fn=nn_ops.relu,
                        normalizer_fn=layers_lib.batch_norm,
                        normalizer_params=batch_norm_params) as sc:
                    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):

                            # 8 x 8 x 2048 17 x 17 x 768
                            net, aux_net = self.inception_v3_base(inputs=inputs, scope=scope)

                            with tf.compat.v1.variable_scope('Logits'):

                                kernel_size = self.reduce_kernel_size(net, kernel_size=[8, 8])
                                net = slim.avg_pool2d(inputs=net,kernel_size=kernel_size, stride=1,
                                                      scope='AvgPool_1a_{}x{}'.format(*kernel_size), padding='VALID')
                                # 1 x 1 x 2048
                                # dropout
                                net = slim.dropout(inputs=net, keep_prob=keep_prob, scope='Dropout_1b')
                                # conv layer 1 * 1 * num_class
                                logits = slim.conv2d(inputs=net, num_outputs=num_classes, kernel_size=[1, 1], stride=1,
                                                     scope='Conv2d_1c_1x1')

                                logits = tf.squeeze(input=logits, axis=[1, 2], name='SpatialSqueeze')
                                prop = slim.softmax(logits=logits, scope='Softmax')

                                return prop


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

    def inception_v3_arg_scope(weight_decay=0.00004,
                               batch_norm_var_collection='moving_vars',
                               batch_norm_decay=0.9997,
                               batch_norm_epsilon=0.001,
                               updates_collections=ops.GraphKeys.UPDATE_OPS,
                               use_fused_batchnorm=True):
        """Defines the default InceptionV3 arg scope.

        Args:
          weight_decay: The weight decay to use for regularizing the model.
          batch_norm_var_collection: The name of the collection for the batch norm
            variables.
          batch_norm_decay: Decay for batch norm moving average
          batch_norm_epsilon: Small float added to variance to avoid division by zero
          updates_collections: Collections for the update ops of the layer
          use_fused_batchnorm: Enable fused batchnorm.

        Returns:
          An `arg_scope` to use for the inception v3 model.
        """
        batch_norm_params = {
            # Decay for the moving averages.
            'decay': batch_norm_decay,
            # epsilon to prevent 0s in variance.
            'epsilon': batch_norm_epsilon,
            # collection containing update_ops.
            'updates_collections': updates_collections,
            # Use fused batch norm if possible.
            'fused': use_fused_batchnorm,
            # collection containing the moving mean and moving variance.
            'variables_collections': {
                'beta': None,
                'gamma': None,
                'moving_mean': [batch_norm_var_collection],
                'moving_variance': [batch_norm_var_collection],
            }
        }

        # Set weight_decay for weights in Conv and FC layers.
        with arg_scope(
                [layers.conv2d, layers_lib.fully_connected],
                weights_regularizer=regularizers.l2_regularizer(weight_decay)):
            with arg_scope(
                    [layers.conv2d],
                    weights_initializer=initializers.variance_scaling_initializer(),
                    activation_fn=nn_ops.relu,
                    normalizer_fn=layers_lib.batch_norm,
                    normalizer_params=batch_norm_params) as sc:
                return sc

    def inception_v3_base(self, inputs, scope='InceptionV3'):
        """
        inception V3 base
        :param inputs:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope, default_name='InceptionV3', values=[inputs]):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
                # 299 x 299 x 3
                net = slim.conv2d(inputs=inputs, num_outputs=32,kernel_size=[3, 3], stride=2, scope='Conv2d_1a_3x3')

                # 149 x 149 x 32
                net = slim.conv2d(inputs=net, num_outputs=32, kernel_size=[3, 3], stride=1, scope='Conv2d_2a_3x3')
                # 147 x 147 x 32
                net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[3, 3], stride=1, scope='Conv2d_2b_3x3',
                                  padding='SAME')
                # 147 x 147 x 64
                net = slim.max_pool2d(inputs=net,kernel_size=[3, 3], stride=2, scope='MaxPool_3a_3x3')
                # 73 X 73 X 64
                net = slim.conv2d(inputs=net, num_outputs=80, kernel_size=[1, 1], stride=1, scope='Conv2d_3b_1x1')

                # 73 x 73 x 80
                net = slim.conv2d(inputs=net, num_outputs=192, kernel_size=[3, 3], stride=1, scope='Conv2d_4a_3x3')

                # 71 x 71 x 192
                net = slim.max_pool2d(inputs=net, kernel_size=[3, 3], stride=2, scope='MaxPool_5a_3x3')
                # 3 x inception_module_v3_a
                # 35 x 35 x 192
                net = self.inception_module_v3_a_1(inputs=net, outputs_list=[64, 48, 64, 64, 96, 96, 32], scope='Mixed_5b')
                # 35 x 35 x 256
                net = self.inception_module_v3_a_2(inputs=net, outputs_list=[64, 48, 64, 64, 96, 96, 64], scope='Mixed_5c')
                # 35 x 35 x 288
                net = self.inception_module_v3_a_1(inputs=net, outputs_list=[64, 48, 64, 64, 96, 96, 64], scope='Mixed_5d')

                # 35 x 35 x 288
                net = self.inception_module_v3_reduce_a(inputs=net, outputs_list=[384, 64, 96, 96], scope='Mixed_6a')
                # 5 inception_module_v3_b
                # 17 x 17 x 768
                net = self.inception_module_v3_b(inputs=net,
                                                 outputs_list=[192, 128, 128, 192, 128, 128, 128, 128, 192, 192],
                                                 scope='Mixed_6b')
                # 17 x 17 x 768
                net = self.inception_module_v3_b(inputs=net,
                                                 outputs_list=[192, 160, 160, 192, 160, 160, 160, 160, 192, 192],
                                                 scope='Mixed_6c')
                # 17 x 17 x 768
                net = self.inception_module_v3_b(inputs=net,
                                                 outputs_list=[192, 160, 160, 192, 160, 160, 160, 160, 192, 192],
                                                 scope='Mixed_6d')
                # 17 x 17 x 768
                net = self.inception_module_v3_b(inputs=net,
                                                 outputs_list=[192, 192, 192, 192, 192, 192, 192, 192, 192, 192],
                                                 scope='Mixed_6e')
                # auxiliary net
                aux_net = net

                # 17 x 17 x 768
                net = self.inception_module_v3_reduce_b(inputs=net,
                                                        outputs_list=[192, 320, 192, 192, 192, 192],
                                                        scope='Mixed_7a')
                # 2 x inception_module_v3_c
                # 8 x 8 x 1280
                net = self.inception_module_v3_c_1(inputs=net,
                                                 outputs_list=[320, 384, 384, 384, 448, 384, 384, 384, 192],
                                                 scope='Mixed_7b')
                # 8 x 8 x 2048
                net = self.inception_module_v3_c_2(inputs=net,
                                                 outputs_list=[320, 384, 384, 384, 448, 384, 384, 384, 192],
                                                 scope='Mixed_7c')
                # 8 x 8 x 2048
                return net, aux_net

    def inception_module_v3_a_1(self, inputs, outputs_list, scope):
        """
        inception v3 module a(paper figure 5)
        :param inputs:
        :param outputs_list:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                # branch 0
                with tf.compat.v1.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[0],kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                # branch 1
                with tf.compat.v1.variable_scope('Branch_1'):

                    branch_1 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[1], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(inputs=branch_1, num_outputs=outputs_list[2], kernel_size=[5, 5], stride=1,
                                           scope='Conv2d_0b_5x5')
                # branch 2
                with tf.compat.v1.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[3], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=outputs_list[4], kernel_size=[3, 3], stride=1,
                                           scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=outputs_list[5], kernel_size=[3, 3], stride=1,
                                           scope='Conv2d_0c_3x3')
                # branch 3
                with tf.compat.v1.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(inputs=inputs, kernel_size=[3, 3], stride=1, scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(inputs=branch_3, num_outputs=outputs_list[6], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3)
                return net

    def inception_module_v3_a_2(self, inputs, outputs_list, scope):
        """
        inception v3 module a(paper figure 5)
        :param inputs:
        :param outputs_list:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                # branch 0
                with tf.compat.v1.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[0],kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                # branch 1
                with tf.compat.v1.variable_scope('Branch_1'):

                    branch_1 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[1], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(inputs=branch_1, num_outputs=outputs_list[2], kernel_size=[5, 5], stride=1,
                                           scope='Conv_1_0c_5x5')
                # branch 2
                with tf.compat.v1.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[3], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=outputs_list[4], kernel_size=[3, 3], stride=1,
                                           scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=outputs_list[5], kernel_size=[3, 3], stride=1,
                                           scope='Conv2d_0c_3x3')
                # branch 3
                with tf.compat.v1.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(inputs=inputs, kernel_size=[3, 3], stride=1, scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(inputs=branch_3, num_outputs=outputs_list[6], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3)
                return net

    def inception_module_v3_b(self, inputs, outputs_list, scope):
        """
        inception v3 module b(paper figure 6)
        :param input_op:
        :param outputs_list:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                # branch 0
                with tf.compat.v1.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[0], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                # branch 1
                with tf.compat.v1.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[1], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(inputs=branch_1, num_outputs=outputs_list[2], kernel_size=[1, 7], stride=1,
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(inputs=branch_1, num_outputs=outputs_list[3], kernel_size=[7, 1], stride=1,
                                           scope='Conv2d_0c_7x1')
                # branch 2
                with tf.compat.v1.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[4], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=outputs_list[5], kernel_size=[7, 1], stride=1,
                                           scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=outputs_list[6], kernel_size=[1, 7], stride=1,
                                           scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=outputs_list[7], kernel_size=[7, 1], stride=1,
                                           scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=outputs_list[8], kernel_size=[1, 7], stride=1,
                                           scope='Conv2d_0e_1x7')
                # branch c
                with tf.compat.v1.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(inputs=inputs, kernel_size=[3, 3], stride=1, scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(inputs=branch_3, num_outputs=outputs_list[9], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3)
                return net

    def inception_module_v3_c_1(self, inputs, outputs_list, scope):
        """
        inception v3 module c (paper figure 7)
        :param input_op:
        :param filters_list:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
                with tf.compat.v1.variable_scope('Branch_0'):
                    # branch_0
                    branch_0 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[0], kernel_size=[1, 1], stride=1,
                                               scope='Conv2d_0a_1x1')
                with tf.compat.v1.variable_scope('Branch_1'):
                    # branch_1
                    branch_1 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[1], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                    branch_1_1 = slim.conv2d(inputs=branch_1, num_outputs=outputs_list[2], kernel_size=[1, 3], stride=1,
                                           scope='Conv2d_0b_1x3')
                    # branch_1_2 = slim.conv2d(inputs=branch_1, num_outputs=outputs_list[3], kernel_size=[3, 3], stride=1,
                                           # scope='Conv2d_0b_3x1')
                    branch_1_2 = slim.conv2d(inputs=branch_1, num_outputs=outputs_list[3], kernel_size=[3, 1], stride=1,
                                             scope='Conv2d_0b_3x1')
                    branch_1 = tf.concat(values=[branch_1_1, branch_1_2], axis=3)

                with tf.compat.v1.variable_scope('Branch_2'):
                    # branch_1
                    branch_2 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[4], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=outputs_list[5], kernel_size=[3, 3], stride=1,
                                           scope='Conv2d_0b_3x3')
                    branch_2_1 = slim.conv2d(inputs=branch_2, num_outputs=outputs_list[6], kernel_size=[1, 3], stride=1,
                                           scope='Conv2d_0c_1x3')
                    branch_2_2 = slim.conv2d(inputs=branch_2, num_outputs=outputs_list[7], kernel_size=[3, 1], stride=1,
                                           scope='Conv2d_0d_3x1')
                    branch_2 = tf.concat(values=[branch_2_1, branch_2_2], axis=3)
                with tf.compat.v1.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(inputs=inputs, kernel_size=[3, 3], stride=1, scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(inputs=branch_3, num_outputs=outputs_list[8], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3)
                return net

    def inception_module_v3_c_2(self, inputs, outputs_list, scope):
        """
        inception v3 module c (paper figure 7)
        :param input_op:
        :param filters_list:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
                with tf.compat.v1.variable_scope('Branch_0'):
                    # branch_0
                    branch_0 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[0], kernel_size=[1, 1], stride=1,
                                               scope='Conv2d_0a_1x1')
                with tf.compat.v1.variable_scope('Branch_1'):
                    # branch_1
                    branch_1 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[1], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                    branch_1_1 = slim.conv2d(inputs=branch_1, num_outputs=outputs_list[2], kernel_size=[1, 3], stride=1,
                                           scope='Conv2d_0b_1x3')
                    # branch_1_2 = slim.conv2d(inputs=branch_1, num_outputs=outputs_list[3], kernel_size=[3, 3], stride=1,
                                           # scope='Conv2d_0b_3x1')
                    branch_1_2 = slim.conv2d(inputs=branch_1, num_outputs=outputs_list[3], kernel_size=[3, 1], stride=1,
                                             scope='Conv2d_0c_3x1')
                    branch_1 = tf.concat(values=[branch_1_1, branch_1_2], axis=3)

                with tf.compat.v1.variable_scope('Branch_2'):
                    # branch_1
                    branch_2 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[4], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(inputs=branch_2, num_outputs=outputs_list[5], kernel_size=[3, 3], stride=1,
                                           scope='Conv2d_0b_3x3')
                    branch_2_1 = slim.conv2d(inputs=branch_2, num_outputs=outputs_list[6], kernel_size=[1, 3], stride=1,
                                           scope='Conv2d_0c_1x3')
                    branch_2_2 = slim.conv2d(inputs=branch_2, num_outputs=outputs_list[7], kernel_size=[3, 1], stride=1,
                                           scope='Conv2d_0d_3x1')
                    branch_2 = tf.concat(values=[branch_2_1, branch_2_2], axis=3)
                with tf.compat.v1.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(inputs=inputs, kernel_size=[3, 3], stride=1, scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(inputs=branch_3, num_outputs=outputs_list[8], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3)
                return net


    def inception_module_v3_reduce_a(self, inputs, outputs_list, scope):
        """
         inception v3 module reduce_a(figure 10)
        :param input_op:
        :param filters_list:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                # branch 0
                with tf.compat.v1.variable_scope('Branch_0'):
                    # branch_0 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[0], kernel_size=[3, 3], stride=2,
                    #                        scope='Conv2d_1a_1x1')
                    branch_0 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[0], kernel_size=[3, 3], stride=2,
                                           scope='Conv2d_1a_1x1', padding='VALID')
                # branch 1
                with tf.compat.v1.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[1], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1', padding='SAME')
                    branch_1 = slim.conv2d(inputs=branch_1, num_outputs=outputs_list[2], kernel_size=[3, 3], stride=1,
                                           scope='Conv2d_0b_3x3', padding='SAME')
                    # branch_1 = slim.conv2d(inputs=branch_1, num_outputs=outputs_list[3], kernel_size=[3, 3], stride=2,
                    #                        scope='Conv2d_1a_3x3', padding='VALID')
                    branch_1 = slim.conv2d(inputs=branch_1, num_outputs=outputs_list[3], kernel_size=[3, 3], stride=2,
                                           scope='Conv2d_1a_1x1', padding='VALID')

                # branch 2
                with tf.compat.v1.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(inputs=inputs, kernel_size=[3, 3], stride=2, scope='MaxPool_1a_3x3',
                                               padding='VALID')
                net = tf.concat(values=[branch_0, branch_1, branch_2], axis=3)
                return net

    def inception_module_v3_reduce_b(self, inputs, outputs_list, scope):
        """
         inception v3 module reduce_b
        :param input_op:
        :param filters_list:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                # branch 0
                with tf.compat.v1.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[0], kernel_size=[1, 1], stride=1,
                                scope='Conv2d_0a_1x1', padding='SAME')
                    branch_0 = slim.conv2d(inputs=branch_0, num_outputs=outputs_list[1], kernel_size=[3, 3], stride=2,
                                           scope='Conv2d_1a_3x3', padding='VALID')
                # branch 1
                with tf.compat.v1.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs=inputs, num_outputs=outputs_list[2], kernel_size=[1, 1], stride=1,
                                           scope='Conv2d_0a_1x1', padding='SAME')
                    branch_1 = slim.conv2d(inputs=branch_1, num_outputs=outputs_list[3], kernel_size=[1, 7], stride=1,
                                           scope='Conv2d_0b_1x7', padding='SAME')
                    branch_1 = slim.conv2d(inputs=branch_1, num_outputs=outputs_list[4], kernel_size=[7, 1], stride=1,
                                           scope='Conv2d_0c_7x1', padding='SAME')
                    branch_1 = slim.conv2d(inputs=branch_1, num_outputs=outputs_list[5], kernel_size=[3, 3], stride=2,
                                           scope='Conv2d_1a_3x3', padding='VALID')
                # branch 2
                with tf.compat.v1.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(inputs=inputs, kernel_size=[3, 3], stride=2, scope='MaxPool_1a_3x3',
                                               padding='VALID')
                net = tf.concat(values=[branch_0, branch_1, branch_2], axis=3)
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