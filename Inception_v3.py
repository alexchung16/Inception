#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File Inception_v3.py
# @ Description :
# @ Author alexchung
# @ Time 1/11/2019 PM 15:27
"""
inception V3 network structure
Conv2d_1a_3x3
Conv2d_2a_3x3
Conv2d_2b_3x3
MaxPool_3a_3x3
Conv2d_3b_1x1
Conv2d_4a_3x3
MaxPool_5a_3x3
Mixed_5b
Mixed_5c
Mixed_5d
Mixed_6a
Mixed_6b
Mixed_6c
Mixed_6d
Mixed_6e
Mixed_7a
Mixed_7b
Mixed_7c

scope description:
Conv2d_0a_3x3
{operation}_{type}_{kernel_size}
operation: Conv2d, MaxPool...
type: 0->'SAME' 1->'VALID'
kernel_size: 1x1, 3x3, 5x5, 7x7...
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

class InceptionV3():
    """
    Inception v1
    """
    def __init__(self, input_shape, num_classes, batch_size, decay_steps, decay_rate, learning_rate,
                 keep_prob=0.8, global_pool=False, spacial_squeeze=True):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.global_pool = global_pool
        self.spacial_squeeze = spacial_squeeze
        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        # add placeholder (X,label)
        self.raw_input_data = tf.compat.v1.placeholder (tf.float32, shape=[None, input_shape[0], input_shape[1], input_shape[2]], name="input_images")
        # y [None,num_classes]
        self.raw_input_label = tf.compat.v1.placeholder (tf.float32, shape=[None, self.num_classes], name="class_label")

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
        pass

    def inception_v3(self, inputs, scope='InceptionV3', num_classes=10, keep_prob=0.8, global_pool=False,
                     spatial_squeeze=True, reuse=None, create_aux_logits=True):
        """
        inception v3
        :return:
        """
        with tf.compat.v1.variable_scope(scope, default_name='InceptionV3', values=[inputs]) as scope:
            # 8 x 8 x 2048 17 x 17 x 768
            net, aux_net = self.inception_v3_base(inputs=inputs, scope=scope)
        if create_aux_logits and num_classes:
            with tf.compat.v1.variable_scope('AuxLogits'):
                # 17 x 17 x 768
                aux_logits = avgPool2dLayer(input_op=aux_net, scope='AvgPool_1a_5x5', kernel_size=[5 ,5],
                                            strides=3, padding='VALID')
                # 5 x 5 x 768
                aux_logits = conv2dLayer(input_op=aux_logits, scope='Conv2d_1b_1x1', kernel_size=[1, 1],
                                         filters=128, strides=1, padding='SAME')
                # 1 x 1 x 128
                kernel_size = self.reduce_kernel_size(input_op=aux_logits, kernel_size=[5, 5])
                aux_logits = conv2dLayer(input_op=aux_logits, scope='Conv2d_2b_{}x{}'.format(*kernel_size),
                                         kernel_size=kernel_size, weight_mean=1e-2, filters=768, strides=1,
                                         padding='VALID')
                # 1 X 1 X 768
                aux_logits = conv2dLayer(input_op=aux_logits, scope='Conv2d_2b_1x1', kernel_size=kernel_size,
                                         weight_mean=1e-3, filters=num_classes, strides=1,
                                         padding='VALID')
                if spatial_squeeze:
                    aux_logits = tf.squeeze(input=aux_logits, axis=[1, 2], name='SpatialSqueeze')

        with tf.compat.v1.variable_scope('Logits'):
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='Global_Pool')
            else:
                # 8 x 8 x 2048
                kernel_size = self.reduce_kernel_size(net, kernel_size=[8, 8])
                net = avgPool2dLayer(input_op=net, scope='AvgPool_1a_{}x{}'.format(*kernel_size),
                                     kernel_size=kernel_size, strides=1, padding='VALID')
            # 1 x 1 x 2048
            # dropout
            net = dropoutLayer(input_op=net, scope='Dropout_0b', keep_prob=keep_prob)
            # conv layer 1*1*num_class
            logits = conv2dLayer(input_op=net, scope='Conv2d_0c_1x1', filters=num_classes, kernel_size=[1, 1],
                                 strides=1)

            if spatial_squeeze:
                logits = tf.squeeze(input=logits, axis=[1, 2], name='SpatialSqueeze')
            prop = softmaxLayer(input_op=logits, scope='Softmax')

        return prop, aux_logits




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

    def inception_v3_base(self, inputs, scope='InceptionV3'):
        """
        inception V3 base
        :param inputs:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope, default_name='InceptionV1', values=[inputs]):
            # 229 x 229 x 3
            net = conv2dLayer(input_op=inputs, scope='Conv2d_1a_3x3', kernel_size=[3, 3],
                              filters=32, strides=2, padding='VALID')

            # 149 x 149 x 32
            net = conv2dLayer(input_op=net, scope='Conv2d_2a_3x3', kernel_size=[3, 3],
                              filters=32, strides=1, padding='VALID')
            # 147 x 147 x 32
            net = conv2dLayer(input_op=net, scope='Conv2d_2b_3x3', kernel_size=[3, 3],
                              filters=64, strides=1, padding='SAME')

            # 147 x 147 x 64
            net = maxPool2dLayer(input_op=net, scope='MaxPool_3a_3x3', kernel_size=[3, 3],
                                 strides=2, padding='VALID')
            # 73 X 73 X 64
            net = conv2dLayer(input_op=net, scope='Conv2d_3b_1x1', kernel_size=[1, 1],
                              filters=80, strides=1, padding='VALID')

            # 73 x 73 x 80
            net = conv2dLayer(input_op=net, scope='Conv2d_4a_3x3', kernel_size=[3, 3],
                              filters=192, strides=1, padding='VALID')

            # 71 x 71 x 192
            net = maxPool2dLayer(input_op=net, scope='MaxPool_5a_3x3', kernel_size=[3, 3],
                                 strides=2, padding='VALID')
            # 3 x inception_module_v3_a
            # 35 x 35 x 192
            net = self.inception_module_v3_a(input_op=net, filters_list=[64, 48, 64, 64, 96, 96, 32], scope='Mixed_5b')
            # 35 x 35 x 256
            net = self.inception_module_v3_a(input_op=net, filters_list=[64, 48, 64, 64, 96, 96, 64], scope='Mixed_5c')
            # 35 x 35 x 288
            net = self.inception_module_v3_a(input_op=net, filters_list=[64, 48, 64, 64, 96, 96, 64], scope='Mixed_5d')
            # 35 x 35 x 288
            net = self.inception_module_v3_reduce_a(input_op=net, filters_list=[384, 64, 96, 96], scope='Mixed_6a')
            # 5 inception_module_v3
            # 17 x 17 x 768
            net = self.inception_module_v3_b(input_op=net,
                                             filters_list=[192, 128, 128, 192, 128, 128, 128, 128, 192, 192],
                                             scope='Mixed_6b')
            # 17 x 17 x 768
            net = self.inception_module_v3_b(input_op=net,
                                             filters_list=[192, 160, 160, 192, 160, 160, 160, 160, 192, 192],
                                             scope='Mixed_6c')
            # 17 x 17 x 768
            net = self.inception_module_v3_b(input_op=net,
                                             filters_list=[192, 160, 160, 192, 160, 160, 160, 160, 192, 192],
                                             scope='Mixed_6d')
            # 17 x 17 x 768
            net = self.inception_module_v3_b(input_op=net,
                                             filters_list=[192, 192, 192, 192, 192, 192, 192, 192, 192, 192],
                                             scope='Mixed_6e')
            # auxiliary net
            aux_net = net

            # 17 x 17 x 768
            net = self.inception_module_v3_reduce_b(input_op=net, filters_list=[192, 320, 192, 192, 192, 192],
                                                    scope='Mixed_7a')
            # 2 x inception_module_v3_c
            # 17 x 17 x 1280
            net = self.inception_module_v3_c(input_op=net, filters_list=[320, 384, 384, 384, 448, 384, 384, 384, 192],
                                             scope='Mixed_7b')
            # 8 x 8 x 2048
            net = self.inception_module_v3_c(input_op=net, filters_list=[320, 384, 384, 384, 448, 384, 384, 384, 192],
                                             scope='Mixed_7c')
            # 8 x 8 x 2048
            return net, aux_net


    def inception_module_v3_a(self, input_op, filters_list, scope):
        """
        inception v3 module a(paper figure 5)
        :param input_op:
        :param filters_list:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope):
            # branch 0
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[0], strides=1)
            # branch 1
            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[1], strides=1)
                branch_1 = conv2dLayer(input_op=branch_1, scope='Conv2d_0b_3x3', kernel_size=[5, 5],
                                       filters=filters_list[2], strides=1)
                # branch_1 = conv2dLayer(input_op=branch_1, scope='Conv2d_0b_3x3', kernel_size=[3, 3],
                #                        filters=filters_list[2], strides=1)
                # branch_1 = conv2dLayer(input_op=branch_1, scope='Conv2d_0c_3x3', kernel_size=[3, 3],
                #                        filters=filters_list[3], strides=1)
            # branch 2
            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[3], strides=1)
                branch_2 = conv2dLayer(input_op=branch_2, scope='Conv2d_0b_3x3', kernel_size=[3, 3],
                                       filters=filters_list[4], strides=1)
                branch_2 = conv2dLayer(input_op=branch_2, scope='Conv2d_0c_3x3', kernel_size=[3, 3],
                                       filters=filters_list[5], strides=1)
            # branch 3
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = avgPool2dLayer(input_op=input_op, scope='AvgPool_3x3', kernel_size=[3, 3], strides=1)
                branch_3 = conv2dLayer(input_op=branch_1, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[6], strides=1)
            net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3)
        return net


    def inception_module_v3_b(self, input_op, filters_list, scope):
        """
        inception v3 module b(paper figure 6)
        :param input_op:
        :param filters_list:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope):
            # branch 0
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[0], strides=1)
            # branch 1
            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[1], strides=1)
                branch_1 = conv2dLayer(input_op=branch_1, scope='Conv2d_0b_1x7', kernel_size=[1, 7],
                                       filters=filters_list[2], strides=1)
                branch_1 = conv2dLayer(input_op=branch_1, scope='Conv2d_0c_7x1', kernel_size=[7, 1],
                                       filters=filters_list[3], strides=1)
            # branch 2
            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[4], strides=1)
                branch_2 = conv2dLayer(input_op=branch_2, scope='Conv2d_0b_7x1', kernel_size=[7, 1],
                                       filters=filters_list[5], strides=1)
                branch_2 = conv2dLayer(input_op=branch_2, scope='Conv2d_0c_1x7', kernel_size=[1, 7],
                                       filters=filters_list[6], strides=1)
                branch_2 = conv2dLayer(input_op=branch_2, scope='Conv2d_0d_7x1', kernel_size=[7, 1],
                                       filters=filters_list[7], strides=1)
                branch_2 = conv2dLayer(input_op=branch_2, scope='Conv2d_0e_1x7', kernel_size=[1, 7],
                                       filters=filters_list[8], strides=1)
            # branch c
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = avgPool2dLayer(input_op=input_op, scope='AvgPool_0a_3x3', kernel_size=[3, 3])
                branch_3 = conv2dLayer(input_op=branch_3, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[9], strides=1)
            net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3)
            return net

    def inception_module_v3_c(self, input_op, filters_list, scope):
        """
        inception v3 module c (paper figure 7)
        :param input_op:
        :param filters_list:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope):
            with tf.compat.v1.variable_scope('Branch_0'):
                # branch_0
                branch_0 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[0], strides=1)
            with tf.compat.v1.variable_scope('Branch_1'):
                # branch_1
                branch_1 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[1], strides=1)
                branch_1_1 = conv2dLayer(input_op=branch_1, scope='Conv2d_0b_1x3', kernel_size=[1, 3],
                                         filters=filters_list[2], strides=1)
                branch_1_2 = conv2dLayer(input_op=branch_1, scope='Conv2d_0b_3x1', kernel_size=[3, 1],
                                         filters=filters_list[3], strides=1)
                branch_1 = tf.concat(values=[branch_1_1, branch_1_2], axis=3)

            with tf.compat.v1.variable_scope('Branch_2'):
                # branch_1
                branch_2 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[4], strides=1)
                branch_2 = conv2dLayer(input_op=branch_2, scope='Conv2d_0b_3x3', kernel_size=[3, 3],
                                         filters=filters_list[5], strides=1)
                branch_2_1 = conv2dLayer(input_op=branch_2, scope='Conv2d_0c_1x3', kernel_size=[1, 3],
                                         filters=filters_list[6], strides=1)
                branch_2_2 = conv2dLayer(input_op=branch_2, scope='Conv2d_0c_3x1', kernel_size=[3, 1],
                                         filters=filters_list[7], strides=1)
                branch_2 = tf.concat(values=[branch_2_1, branch_2_2], axis=3)
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = avgPool2dLayer(input_op=input_op, scope='AvgPool_0a_3x3', kernel_size=[3, 3])
                branch_3 = conv2dLayer(input_op=branch_3, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[8], strides=1)
            net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3)
            return net

    def inception_module_v3_reduce_a(self, input_op, filters_list, scope):
        """
         inception v3 module reduce_a(figure 10)
        :param input_op:
        :param filters_list:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope):
            # branch 0
            with tf.compat.v1.variable_scope('Branch_0'):
                # branch_0 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                #                        filters=filters_list[0], strides=1)
                branch_0 = conv2dLayer(input_op=input_op, scope='Conv2d_1a_3x3', kernel_size=[3, 3],
                                       filters=filters_list[0], strides=2, padding='VALID')
            # branch 1
            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[1], strides=1)
                branch_1 = conv2dLayer(input_op=branch_1, scope='Conv2d_0b_3x3', kernel_size=[3, 3],
                                       filters=filters_list[2], strides=1)
                branch_1 = conv2dLayer(input_op=branch_1, scope='Con2d_0c_3x3', kernel_size=[3, 3],
                                       filters=filters_list[3], strides=2, padding='VALID')
            # branch 2
            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = maxPool2dLayer(input_op=input_op, scope='MaxPool_3x3', kernel_size=[3, 3],
                                          strides=2, padding='VALID')
            net = tf.concat(values=[branch_0, branch_1, branch_2], axis=3)
            return net

    def inception_module_v3_reduce_b(self, input_op, filters_list, scope):
        """
         inception v3 module reduce_b
        :param input_op:
        :param filters_list:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope):
            # branch 0
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[0], strides=1)
                branch_0 = conv2dLayer(input_op=branch_0, scope='Conv2d_1a_3x3', kernel_size=[3, 3],
                                       filters=filters_list[1], strides=2, padding='VALID')
            # branch 1
            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', kernel_size=[1, 1],
                                       filters=filters_list[2], strides=1)
                branch_1 = conv2dLayer(input_op=branch_1, scope='Conv2d_0b_1x7', kernel_size=[1, 7],
                                       filters=filters_list[3], strides=1)
                branch_1 = conv2dLayer(input_op=branch_1, scope='Conv2d_0c_7x1', kernel_size=[7, 1],
                                       filters=filters_list[4], strides=1)
                branch_1 = conv2dLayer(input_op=branch_1, scope='Con2d_1a_3x3', kernel_size=[3, 3],
                                       filters=filters_list[5], strides=2, padding='VALID')
            # branch 2
            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = maxPool2dLayer(input_op=input_op, scope='MaxPool_1a_3x3', kernel_size=[3, 3],
                                          strides=2, padding='VALID')
            net = tf.concat(values=[branch_0, branch_1, branch_2], axis=3)
            return net



def conv2dLayer(input_op, scope, filters, kernel_size=None, strides=None, use_bias=False, weight_mean=None,
                weight_stddev=None, bias_value=None, padding='SAME', parameter=None):
    """
    convolution operation
    :param input_op:
    :param scope:
    :param filters:
    :param kernel_size:
    :param strides:
    :param use_bias:
    :param padding:
    :param parameter:
    :return:
    """
    if kernel_size is None:
        kernel_size = [3, 3]
    if strides is None:
        strides = 1
    if weight_mean is None:
        weight_mean = 0.0
    if weight_stddev is None:
        weight_mean = 1e-1
    if bias_value is None:
        bias_value = 0.0
    # get feature num
    features = input_op.get_shape()[-1].value
    with tf.name_scope(scope) as scope:
        filter = getConvFilter(filter_shape=[kernel_size[0], kernel_size[1], features, filters],
                               mean=weight_mean, stddev=weight_stddev)

        outputs = tf.nn.conv2d(input=input_op, filter=filter, strides=[1, strides, strides, 1], padding=padding)

        if use_bias:
            biases = getBias(bias_shape=[filters], value=bias_value)
            outputs = tf.nn.bias_add(value=outputs, bias=biases)
        #     parameter += [filter, biases]
        # else:
        #     parameter += [filters]
        outputs = batchNormalize2d(input_op=outputs)
        return tf.nn.relu(outputs)


def fcLayer(input_op, scope, num_outputs, parameter=None):
    """
     full connect operation
    :param input_op:
    :param scope:
    :param num_outputs:
    :param parameter:
    :return:
    """
    # get feature num
    features = input_op.get_shape()[-1].value
    with tf.name_scope(scope) as scope:
        weights = getFCWeight(weight_shape=[features, num_outputs])
        biases = getBias(bias_shape=[num_outputs])
        # parameter += [weights, biases]
        return tf.nn.relu_layer(x=input_op, weights=weights, biases=biases)


def maxPool2dLayer(input_op, scope, kernel_size=None, strides=None, padding='SAME'):
    """
     max pooling layer
    :param input_op:
    :param scope:
    :param kernel_size:
    :param strides_size:
    :param padding:
    :return:
    """
    with tf.compat.v1.variable_scope(scope) as scope:
        if kernel_size is None:
            ksize = [1, 2, 2, 1]
        else:
            ksize = [1, kernel_size[0], kernel_size[1], 1]
        if strides is None:
            strides = [1, 2, 2, 1]
        else:
            strides = [1, strides, strides, 1]
        return tf.nn.max_pool2d(input=input_op, ksize=ksize, strides=strides, padding=padding, name='MaxPool')


def avgPool2dLayer(input_op, scope, kernel_size=None, strides=None, padding='SAME'):
    """
    average_pool pooling layer
    :param input_op:
    :return:
    """
    with tf.compat.v1.variable_scope(scope) as scope:
        if kernel_size is None:
            ksize = [1, 2, 2, 1]
        else:
            ksize = [1, kernel_size[0], kernel_size[1], 1]
        if strides is None:
            strides = [1, 2, 2, 1]
        else:
            strides = [1, strides, strides, 1]
        return tf.nn.avg_pool2d(value=input_op, ksize=ksize, strides=strides, padding=padding, name='AvgPool')


def flatten(input_op, scope):
    """
    flatten layer
    :param input_op:
    :return:
    """
    with tf.compat.v1.variable_scope(scope) as scope:
        shape = input_op.get_shape().as_list()
        out_dim = 1
        for d in shape[1:]:
            out_dim *= d
        return tf.reshape(tensor=input_op, shape=[-1, out_dim], name='Flatten')


def dropoutLayer(input_op, scope, keep_prob):
    """
    dropout regularization layer
    :param inpu_op:
    :param name:
    :param keep_prob:
    :return:
    """
    with tf.compat.v1.variable_scope(scope) as scope:
        return tf.nn.dropout(input_op, rate=1-keep_prob, name='Dropout')


def softmaxLayer(input_op, scope):
    """
    softmax layer
    :param logits:
    :param name:
    :param n_class:
    :return:
    """
    with tf.compat.v1.variable_scope(scope):
        return tf.nn.softmax(logits=input_op, name='Softmax')


def getConvFilter(filter_shape, mean=0.0, stddev=1e-1):
    """
    convolution layer filter
    :param filter_shape:
    :param mean:
    :param stddev:
    :return:
    """
    return tf.Variable(initial_value=tf.random.truncated_normal(shape=filter_shape, mean=mean, stddev=stddev, dtype=tf.float32),
                       trainable=True, name='Filter')


def getFCWeight(weight_shape,  mean=0.0, stddev=1e-1):
    """
    full connect layer weight
    :param weight_shape:
    :return:
    """
    return tf.Variable(initial_value=tf.random.truncated_normal(shape=weight_shape, mean=mean, stddev=stddev, dtype=tf.float32),
                       trainable=True, name='Weight')


def getBias(bias_shape, value=0.0):
    """
    get bias
    :param bias_shape:
    :return:
    """
    return tf.Variable(initial_value=tf.constant(value=0.0, shape=bias_shape, dtype=tf.float32),
                       trainable=True, name='Bias')


def batchNormalize2d(input_op, istring_True=True, eps=1e-5, affine=True, decay=0.9, name=None):
    """

    :param input_op:
    :param eps:
    :param decay:
    :param name:
    :return:
    """
    with tf.compat.v1.variable_scope(name, default_name='Batch_Normalize'):
        axis = list(range(len(input_op.get_shape())-1))
        mean, variance = tf.nn.moments(x=input_op, axes=axis, name='moment')

        params_shape = tf.shape(input_op)[-1]
        moving_mean = getVariable('moving_mean', params_shape, initializer=tf.zeros_initializer, trainable=False)
        moving_variance = getVariable('moving_variance', params_shape, initializer=tf.zeros_initializer, trainable=False)

        def mean_and_var_update():
            with tf.control_dependencies(control_inputs=[
                moving_averages.assign_moving_average(variable=moving_mean, value=mean, decay=decay),
                moving_averages.assign_moving_average(variable=moving_variance, value=variance, decay=decay)
            ]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(pred=istring_True, true_fn=mean_and_var_update,
                                 false_fn=lambda:(moving_mean, moving_variance))
        if affine:
            beta = getVariable(name='beta', shape=params_shape, initializer=tf.zeros_initializer)
            gamma = getVariable(name='gamma', shape=params_shape, initializer=tf.ones_initializer)
        else:
            beta = None
            gamma = None

        x = tf.nn.batch_normalization(x=input_op, mean=mean, variance=variance, offset=beta, scale=gamma,
                                      variance_epsilon=eps)
        return x

def getVariable(name, shape, initializer, weight_decay=0.0, dtype=tf.float32, trainable=True):
    """
    add weight to tf.get_variable
    :param name:
    :param shape:
    :param initializer:
    :param weight_decay:
    :param dtype:
    :param trainable:
    :return:
    """
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    return tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=initializer, use_resource=regularizer,
                           trainable=trainable)
