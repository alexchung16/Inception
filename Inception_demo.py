#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File Inception_demo.py
# @ Description :
# @ Author alexchung
# @ Time 14/11/2019 AM 09:46

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import inception_v3

images = tf.Variable(initial_value=tf.random_uniform(shape=(5, 299, 299, 3), minval=0, maxval=3), dtype=tf.float32)
class_num = tf.constant(value=5, dtype=tf.int32)
# is_training = True


# read net
with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits, end_points = inception_v3.inception_v3(images)

if __name__ == "__main__":


    with tf.Session() as sess:
        # images, class_num = sess.run([images, class_num])

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        sess.run(init)
        for var in tf.model_variables():
            print(var.name)

