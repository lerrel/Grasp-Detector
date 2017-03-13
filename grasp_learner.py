#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from graspNet import model as grasp_net

class grasp_obj:
    def __init__(self, checkpoint_path='./models/shake/checkpoint.ckpt-2000', gpu_id=-1):
        self.checkpoint = checkpoint_path
        if gpu_id==-1:
            self.dev_name = "/cpu:0"
        else:
            self.dev_name = "/gpu:{}".format(gpu_id)

        self.IMAGE_SIZE = 224
        self.NUM_CHANNELS = 3
        self.GRASP_ACTION_SIZE = 18
        self.SEED = 66478  # Set to None for random seed.
        self.BATCH_SIZE = 128

        #CONFIG PARAMS
        self.INTRA_OP_THREADS = 1
        self.INTER_OP_THREADS = 1
        self.SOFT_PLACEMENT = False

        tf.set_random_seed(self.SEED)

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
            grasp_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Grasp')
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
