#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inception_v3_demo.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/5/13 上午10:01
# @ Software   : PyCharm
#-------------------------------------------------------


import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import inception_v3

pretrain_model_dir = '/home/alex/Documents/pretrain_model/Inception/inception_v3/inception_v3.ckpt'


images = tf.Variable(initial_value=tf.random_uniform(shape=(5, 299, 299, 3), minval=0, maxval=3), dtype=tf.float32)
class_num = tf.constant(value=5, dtype=tf.int32)
# is_training = True


# read net
with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits, end_points = inception_v3.inception_v3(images)

if __name__ == "__main__":

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    with tf.Session() as sess:
        # images, class_num = sess.run([images, class_num])
        sess.run(init)
        for var in tf.model_variables():
            print(var.name)

        # exclusion scope
        exclude_variable = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']

        variable = slim.get_variables_to_restore(exclude=exclude_variable)
        # assign specific variables from a checkpoint
        assign_op, feed_dict = slim.assign_from_checkpoint(model_path=pretrain_model_dir, var_list=variable,
                                                           ignore_missing_vars=True)

        for tensor, slice_value in feed_dict.items():
            print(tensor)
            # print(slice_value)

        # get an function witch can assigns specific variables from checkpoint
        assign_fn =slim.assign_from_checkpoint_fn(model_path=pretrain_model_dir, var_list=variable,
                                                  ignore_missing_vars=True)
        # restore variable to sess
        assign_fn(sess)
        print(sess.run('InceptionV3/Conv2d_1a_3x3/weights:0'))