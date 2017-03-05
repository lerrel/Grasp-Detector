#!/usr/bin/env python
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from IPython import embed

from reader import grasp_shake_data
from graspNet import model as grasp_net

class grasp_obj:
    def __init__(self, checkpoint_path='./models/shake/checkpoint.ckpt-2000'):
        self.grasp_shake_data = '/home/baxter/ros_ws/src/adversarial_grasping/grasper/data'
        self.LOGDIR = "./logs/reinforce/grasp"
        self.checkpoint = checkpoint_path
        self.dev_name = "/cpu:0"
        self.WORK_DIRECTORY = 'data'
        self.IMAGE_SIZE = 224
        self.NUM_CHANNELS = 3
        self.SHAKE_FACTOR = 0.25
        self.GRASP_ACTION_SIZE = 18
        self.SEED = 66478  # Set to None for random seed.
        self.BATCH_SIZE = 128
        self.MAX_STEPS = 10000

        #CONFIG PARAMS
        self.INTRA_OP_THREADS = 1
        self.INTER_OP_THREADS = 1
        self.SOFT_PLACEMENT = False

        tf.set_random_seed(self.SEED)
        
        self.self_test = False

        self.config = tf.ConfigProto(allow_soft_placement=self.SOFT_PLACEMENT,
                intra_op_parallelism_threads=self.INTRA_OP_THREADS,
                inter_op_parallelism_threads=self.INTER_OP_THREADS)

    def sigmoid_array(self,x):
        return 1 / (1 + np.exp(-x))
    
    def test_init(self):
        with tf.device(self.dev_name):
            with tf.name_scope('Grasp_training_data'):
                self.Grasp_patches = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE,self.IMAGE_SIZE,self.IMAGE_SIZE,self.NUM_CHANNELS])
            with tf.name_scope('Grasp'):
                self.M = grasp_net()
                self.M.initial_weights(weight_file=None)
                self.grasp_pred = self.M.gen_model(self.Grasp_patches)
        with tf.device("/cpu:0"):
            grasp_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Grasp')
            grasp_saver = tf.train.Saver(grasp_variables, max_to_keep=100)
        with tf.device(self.dev_name):
            self.sess = tf.Session(config = self.config)
            grasp_saver.restore(self.sess, self.checkpoint)

    def test_one_batch(self,Is):
        with tf.device(self.dev_name):
            grasp_feed_dict = {self.Grasp_patches : Is, self.M.dropfc6 : 1.0, self.M.dropfc7 : 1.0}
            g_pred = self.sess.run(self.grasp_pred, feed_dict=grasp_feed_dict)
        return g_pred

    def test_close(self):
        self.sess.close()

    def train_init(self):
        with tf.device(self.dev_name):
            with tf.name_scope('Grasp_training_data'):
                self.Grasp_patches = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE,self.IMAGE_SIZE,self.IMAGE_SIZE,self.NUM_CHANNELS])
                self.Grasp_angle = tf.placeholder(tf.int32, shape=[self.BATCH_SIZE])
                self.Grasp_success = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE])

            with tf.name_scope('Grasp'):
                self.M = grasp_net()
                self.M.initial_weights(weight_file=None)
                self.grasp_pred = self.M.gen_model(self.Grasp_patches)
            self.grasp_loss, self.grasp_accuracy = self.M.gen_loss(self.grasp_pred, self.Grasp_angle, self.Grasp_success)
        with tf.device("/cpu:0"):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            # Step Decay
            grasp_learning_rate = tf.train.exponential_decay(
                0.000001,                # Base learning rate.
                self.global_step,  # Current index into the dataset.
                100,          # Decay step.
                0.1,                # Decay rate.
                staircase=True,name='grasp_learning_rate')
            grasp_loss_sum = tf.scalar_summary('grasp_loss', self.grasp_loss)
            grasp_accuracy_sum = tf.scalar_summary('grasp_accuracy', self.grasp_accuracy)
            grasp_lr_sum = tf.scalar_summary('grasp_lr', grasp_learning_rate)
            grasp_step_sum = tf.scalar_summary('grasp_step', self.global_step)
            self.grasp_summary_op = tf.merge_summary([grasp_loss_sum, grasp_accuracy_sum, grasp_lr_sum, grasp_step_sum])
        with tf.device(self.dev_name):
            #Using RMSProp Optimizer
            with tf.name_scope('Optimizer'):
                grasp_optimizer = tf.train.RMSPropOptimizer(grasp_learning_rate, decay = 0.9, momentum = 0.9)
                self.minimize_grasp_net = grasp_optimizer.minimize(self.grasp_loss, global_step=self.global_step,name='grasp_optimizer')
        with tf.device("/cpu:0"):
            grasp_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Grasp')
            optimizer_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Optimizer')
            variables_to_intialize = optimizer_variables + [self.global_step]
            self.grasp_saver = tf.train.Saver(grasp_variables, max_to_keep=100)
            init_op = tf.initialize_variables(variables_to_intialize)
        with tf.device(self.dev_name):
            self.sess = tf.Session(config = self.config)
            self.grasp_saver.restore(self.sess, self.checkpoint)
            self.sess.run(init_op)
            self.train_writer = tf.train.SummaryWriter(self.LOGDIR,self.sess.graph)
            self.step = 0

    def train_one_batch(self,i_grasp, theta_labels, success_labels):
          start_time = time.time() 
          grasp_feed_dict = {self.Grasp_patches : i_grasp, self.Grasp_angle : theta_labels, self.Grasp_success : success_labels, self.M.dropfc6 : 0.5, self.M.dropfc7 : 0.5}
          _, g_l, g_a, g_p, grasp_summary, self.step, g_sig = self.sess.run([self.minimize_grasp_net, self.grasp_loss,self.grasp_accuracy, self.grasp_pred, self.grasp_summary_op, self.global_step, self.M.sig_op], feed_dict=grasp_feed_dict)
          self.train_writer.add_summary(grasp_summary,self.step)
          print('TRAINING Grasp: Iter{} Loss = {} Accuracy = {} step_time = {}'.format(self.step,g_l, g_a, time.time()-start_time))
          return g_l,g_a,time.time()-start_time


    def train_save(self, fname):
        self.grasp_saver.save(self.sess, "{}".format(fname))

    def train_close():
        self.sess.close()
        
