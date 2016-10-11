# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 23:58:29 2016

@author: bong
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 20:01:14 2016

@author: bong
"""

import tensorflow as tf
import numpy as np

LABEL_NUM = 10

def _variable_on_cpu(shape, init, name):
    with tf.device("/cpu:0"):
        return tf.get_variable(name, shape, dtype=tf.float32, initializer=init)
        
def weight_variable(shape, stdev=0.05, wd=0.0):
    init = tf.truncated_normal_initializer(stddev=stdev, dtype=tf.float32)
    var = _variable_on_cpu(shape=shape, init=init, name='weights')
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def bias_variable(shape, const=0.1):
    init = tf.constant_initializer(const)
    return _variable_on_cpu(shape=shape, init=init, name='biases')

def conv2d_layer(x, shape, stride=1, name='conv2d'):
    kernel = weight_variable(shape=shape)
    biases = bias_variable(shape=shape[3])
    conv = tf.nn.conv2d(x, kernel, [1,stride,stride,1], padding='SAME')
    bias = tf.nn.bias_add(conv,biases)
    return tf.nn.relu(bias, name=name)

def max_pool_layer(x, stride=2, name='max_pool'):
    return tf.nn.max_pool(x, name=name, ksize=[1,3,3,1],
                          strides=[1,stride,stride,1],padding='SAME')

def residual_layer(x, shape, name='residual'):
    assert shape[2]==shape[3]    
    
    with tf.variable_scope('conv1') as scope:
        conv1 = conv2d_layer(x, shape, name=scope.name)
    
    with tf.variable_scope('conv2') as scope:
        kernel = weight_variable(shape=shape)
        biases = bias_variable(shape=shape[3])
        temp = tf.nn.conv2d(conv1, kernel, [1,1,1,1], padding='SAME')
        conv2 = tf.add(temp, biases, name=scope.name)        
    
    return tf.add(conv2, x)
    
def inference(images, keep_prop):
    with tf.variable_scope('conv1') as scope:
        conv1 = conv2d_layer(images, [5, 5, 3, 64], name=scope.name)
    
    with tf.variable_scope('res1') as scope:
        res1 = residual_layer(conv1, [3, 3, 64, 64], name=scope.name)

    with tf.variable_scope('conv2') as scope:
        conv2 = conv2d_layer(res1, [3, 3, 64, 128], stride=2, name=scope.name)

    with tf.variable_scope('res2') as scope:
        res2 = residual_layer(conv2, [3, 3, 128, 128], name=scope.name)

    with tf.variable_scope('conv3') as scope:
        conv3 = conv2d_layer(res2, [3,3,128,256], stride=2, name=scope.name)
    
    with tf.variable_scope('res3') as scope:
        res3 = residual_layer(conv3, [3, 3, 256, 256], name=scope.name)

    with tf.variable_scope('avg_pool') as scope:
        avg_pool = tf.nn.avg_pool(res3, ksize=[1,8,8,1], strides=[1,1,1,1],
                                  padding='VALID', name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        reshaped = tf.reshape(avg_pool, [-1, 256])
        weights = weight_variable([256, LABEL_NUM], wd=0.0)
        biases = bias_variable([LABEL_NUM])
        softmax_linear = tf.add(tf.matmul(reshaped, weights), biases, name=scope.name)

    return softmax_linear

def loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
    
def train(total_loss, lr):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads)
    return apply_gradient_op