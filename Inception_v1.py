#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File Inception_v1.py
# @ Description :
# @ Author alexchung
# @ Time 29/10/2019 AM 10:29


import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


# # generates a truncated normal distribution
# trunc_normal = lambda stddev: tf.random.truncated_normal _initializer(mean=0.0, stddev=stddev)

class InceptionV1():
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
        self.logits =  self.inference(self.raw_input_data, scope='InceptionV1')
        # # computer loss value
        self.loss = self.losses(labels=self.raw_input_label, logits=self.logits, scope='Loss')
        # train operation
        self.train = self.training(self.learning_rate, self.global_step, loss=self.loss)
        self.evaluate_accuracy = self.evaluate_batch(self.logits, self.raw_input_label) / batch_size


    def inference(self, inputs, scope='InceptionV1'):
       """
       Inception V1 net structure
       :param input_op:
       :param name:
       :return:
       """
       self.prameter = []
       prop =  self.inception_v1(inputs = inputs,
                                 num_classes = self.num_classes,
                                 keep_prob = self.keep_prob,
                                 global_pool = self.global_pool,
                                 spatial_squeeze = self.spacial_squeeze,
                                 scope=scope
                                 )
       return prop


    def inception_v1(self, inputs, scope='InceptionV1', num_classes=10, keep_prob=0.8, global_pool=False,
                     spatial_squeeze=True):
        """
        Inception V1 architecture
        :param inputs:
        :param scope:
        :param num_class:
        :param keep_prob:
        :param global_pool:
        :param spatial_squeeze:
        :return:
        """
        with tf.compat.v1.variable_scope(scope, default_name='InceptionV1', values=[inputs]) as scope:
            net = self.inception_v1_base(inputs=inputs, scope=scope)
            with tf.compat.v1.variable_scope('logits'):
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='Global_Pool')
                else:
                    net = avgPool2dLayer(input_op=net, scope='AvgPool_0a_7x7', kernel_size=[7, 7], strides=1, padding='VALID')
                # dropout
                net = dropoutLayer(input_op=net, scope='Dropout_0b', keep_prob=keep_prob)
                # conv layer 1*1*num_class
                logits = conv2dLayer(input_op=net, scope='Conv2d_0c_1x1', filters=num_classes, kernel_size=[1, 1],
                                     strides=1)

                if spatial_squeeze:
                    logits = tf.squeeze(input=logits, axis=[1, 2], name='SpatialSqueeze')
                prop = softmaxLayer(input_op=logits, scope='Softmax')
        return prop


    def inception_v1_base(self, inputs, scope='InceptionV1'):
        """
        Inception V1 base architecture
        :return:
        """
        with tf.compat.v1.variable_scope(scope, default_name='InceptionV1', values=[inputs]):
            # conv_7*7*64
            net = conv2dLayer(input_op=inputs, scope='Conv2d_1a_7x7', filters=64, kernel_size=[7, 7], strides=2)

            # max pool 3*3
            net = maxPool2dLayer(input_op=net, scope='MaxPool_2a_3x3', kernel_size=[3, 3], strides=2)
            # conv 1*1*64
            net = conv2dLayer(input_op=net, scope='Conv2d_2b_1x1', filters=64, kernel_size=[1, 1], strides=1)
            # conv 3*3*192
            net = conv2dLayer(input_op=net, scope='Conv2d_2c_3x3', filters=192, kernel_size=[3, 3], strides=1)

            # max pool 3*3
            net = maxPool2dLayer(input_op=net, scope='MaxPool_3a_3x3', kernel_size=[3, 3], strides=2)
            # inception module
            net = inception_module_v1(input_op=net, scope='Mixed_3b', filters_list=[64, 96, 128, 16, 32, 32])
            # inception module
            net = inception_module_v1(input_op=net, scope='Mixed_3c', filters_list=[128, 128, 192, 32, 96, 64])
            # inception module

            # max pool 3*3
            net = maxPool2dLayer(input_op=net, scope='MaxPool_4a_3x3', kernel_size=[3, 3], strides=2)
            # inception module
            net = inception_module_v1(input_op=net, scope='Mixed_4b', filters_list=[192, 92, 208, 16, 48, 64])
            # inception module
            net = inception_module_v1(input_op=net, scope='Mixed_4c', filters_list=[160, 112, 224, 24, 64, 64])
            # inception module
            net = inception_module_v1(input_op=net, scope='Mixed_4b', filters_list=[128, 128, 256, 24, 64, 64])
            # inception module
            net = inception_module_v1(input_op=net, scope='Mixed_4e', filters_list=[112, 144, 288, 32, 64, 64])
            # inception module
            net = inception_module_v1(input_op=net, scope='Mixed_4f', filters_list=[256, 160, 320, 32, 128, 128])

            # max pool 2*2
            net = maxPool2dLayer(input_op=net, scope='MaxPool_5a_2x2', kernel_size=[2, 2], strides=2)
            # inception module
            net = inception_module_v1(input_op=net, scope='Mixed_5b', filters_list=[256, 160, 320, 32, 128, 128])
            # inception module
            net = inception_module_v1(input_op=net, scope='Mixed_5c', filters_list=[384, 192, 384, 48, 128, 128])

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
        return tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=globalStep)

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

    def fill_feed_dict(self, image_feed, label_feed):
        feed_dict = {
            self.raw_input_data: image_feed,
            self.raw_input_label: label_feed
        }
        return feed_dict


def inception_module_v1(input_op, filters_list, scope):
    """
    inception module
    :param net:
    :param filters_list:[branch_0_filter,
                    branch_1_filter_0, branch_1_filter_1,
                    branch_2_filter_0, branch_2_filter_1,
                    branch_3_filter_0]
    :param scope:
    :return:
    """
    with tf.compat.v1.variable_scope(scope):
        # branch0
        with tf.compat.v1.variable_scope('Branch_0'):
            branch_0 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', filters=filters_list[0],
                                 kernel_size=[1, 1], strides=1)
        # branch 1
        with tf.compat.v1.variable_scope('Branch_1'):
            branch_1 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', filters=filters_list[1],
                                 kernel_size=[1, 1], strides=1)
            branch_1 = conv2dLayer(input_op=branch_1, scope='Conv2d_0b_3x3', filters=filters_list[2],
                                 kernel_size=[3, 3], strides=1)
        # branch 2
        with tf.compat.v1.variable_scope('Branch_2'):
            branch_2 = conv2dLayer(input_op=input_op, scope='Conv2d_0a_1x1', filters=filters_list[3],
                                 kernel_size=[1, 1], strides=1)
            branch_2 = conv2dLayer(input_op=branch_2, scope='Conv2d_0b_3x3', filters=filters_list[4],
                                 kernel_size=[3, 3], strides=1)
        # branch 3
        with tf.compat.v1.variable_scope('Branch_3'):
            branch_3 = maxPool2dLayer(input_op=input_op, scope='MaxPool_0a_3x3', kernel_size=[3, 3], strides=1)
            branch_3 = conv2dLayer(input_op=branch_3, scope='Conv2d_0b_3x3', filters=filters_list[5],
                                 kernel_size=[3, 3], strides=1)
        # concat branch
        output_op = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3, name='Concat')

    return output_op


def conv2dLayer(input_op, scope, filters, kernel_size, strides, use_bias=False, padding='SAME', parameter=None):
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
        strides = 2
    # get feature num
    features = input_op.get_shape()[-1].value
    with tf.name_scope(scope) as scope:
        filter = getConvFilter(filter_shape=[kernel_size[0], kernel_size[1], features, filters])

        outputs = tf.nn.conv2d(input=input_op, filter=filter, strides=[1, strides, strides, 1], padding=padding)

        if use_bias:
            biases = getBias(bias_shape=[filters])
            outputs = tf.nn.bias_add(value=outputs, bias=biases)
        #     parameter += [filter, biases]
        # else:
        #     parameter += [filters]

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


def getConvFilter(filter_shape):
    """
    convolution layer filter
    :param filter_shape:
    :return:
    """
    return tf.Variable(initial_value=tf.random.truncated_normal(shape=filter_shape, mean=0.0, stddev=1e-1, dtype=tf.float32),
                       trainable=True, name='Filter')


def getFCWeight(weight_shape):
    """
    full connect layer weight
    :param weight_shape:
    :return:
    """
    return tf.Variable(initial_value=tf.random.truncated_normal(shape=weight_shape, mean=0.0, stddev=1e-1, dtype=tf.float32),
                       trainable=True, name='Weight')


def getBias(bias_shape):
    """
    get bias
    :param bias_shape:
    :return:
    """
    return tf.Variable(initial_value=tf.constant(value=0.0, shape=bias_shape, dtype=tf.float32),
                       trainable=True, name='Bias')



