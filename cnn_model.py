# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:32:58 2017

@author: Xiangyong Cao
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim
from utils import patch_size

num_band = 220    
num_classes = 16


def conv_net(x):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu):
        net = tf.reshape(x, [-1, patch_size, patch_size, num_band])
        net = slim.conv2d(net, 300, 3, padding='VALID',
                      weights_initializer=tf.contrib.layers.xavier_initializer())
        net = slim.max_pool2d(net,2,padding='SAME')
        net = slim.conv2d(net, 200, 3, padding='VALID',
                      weights_initializer=tf.contrib.layers.xavier_initializer())
        net = slim.max_pool2d(net,2,padding='SAME')
        net = slim.flatten(net)
        net = slim.fully_connected(net,200)
        net = slim.fully_connected(net,100)
        logits = slim.fully_connected(net, num_classes, activation_fn=None)
    return logits
  
  