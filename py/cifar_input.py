# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 21:07:49 2016

@author: bong
"""

import cPickle
import tensorflow as tf
import numpy as np
from tensorflow.python.training import queue_runner


FILE_DIR_10 = "../../images/cifar-10-batches-py"
FILE_DIR_100 = "../../images/cifar-100-python"

IMAGE_DEPTH = 3
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

def label_to_vec(label, LABEL_NUM):
    v = np.zeros([LABEL_NUM])
    v[label]=1.0
    return v

class DataSet(object):
    def __init__(self, images, labels, label_num):
        self.images = images
        self.labels = labels
        self.length = images.shape[0]
        self.label_num = label_num
        self.queue = tf.FIFOQueue(self.length, dtypes = [self.images.dtype,
                                                        self.labels.dtype],
                                shapes=[self.images.shape[1:],self.labels.shape[1:]])
        self.enq = self.queue.enqueue_many([self.images, self.labels])
        queue_runner.add_queue_runner(queue_runner.QueueRunner(self.queue,[self.enq])) 
    
    index=0
    
    def out_tensor(self):
        images, labels = self.queue.dequeue()
        images = tf.reshape(images,[IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH])        
        return tf.transpose(images,[1,2,0]), labels
    
    def dist_out(self):
        intimage, labels = self.out_tensor()
        reshaped_image = tf.cast(intimage, tf.float32)

        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(reshaped_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                   max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                 lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_whitening(distorted_image)
        return float_image, labels

    def batch(self, batch_size):
        image, label = self.out_tensor()
        images_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size)
        return images_batch, label_batch        
    
    def dist_batch(self, batch_size, min_queue_examples):
        image, label = self.dist_out()
        images_batch, label_batch = tf.train.shuffle_batch(
                        [image, label], batch_size=batch_size, num_threads=16,
                        capacity=min_queue_examples + 3 * batch_size,
                        min_after_dequeue=min_queue_examples)
        return images_batch, label_batch

def cifar10_input():
    with open(FILE_DIR_10+"/data_batch_"+str(1), 'rb') as f:
        train_data = cPickle.load(f)
        train_x = train_data['data']
        train_y = train_data['labels']
    for i in range(4):
        with open(FILE_DIR_10+"/data_batch_"+str(i+2), 'rb') as f:
            train_data = cPickle.load(f)
            temp_x = train_data['data']
            temp_y = train_data['labels']
            train_x = np.concatenate((train_x,temp_x))
            train_y = np.concatenate((train_y,temp_y))
        
    with open(FILE_DIR_10+"/test_batch", 'rb') as f:
        test_data = cPickle.load(f)
        test_x = test_data['data']
        test_y = np.reshape(test_data['labels'],[-1])
    
    return DataSet(train_x, train_y, 10), DataSet(test_x, test_y, 10)
    
def cifar100_input():
    with open(FILE_DIR_100+"/train", 'rb') as f:
        train_data = cPickle.load(f)
        train_x = train_data['data']
        train_y = np.reshape(train_data['labels'],[-1])
        
    with open(FILE_DIR_100+"/test", 'rb') as f:
        test_data = cPickle.load(f)
        test_x = test_data['data']
        test_y = np.reshape(test_data['labels'],[-1])
    
    return DataSet(train_x, train_y, 100), DataSet(test_x, test_y, 100)
