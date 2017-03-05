#!/usr/bin/env python

import numpy as np
import cv2
import copy
from sys import stdout
# Given image, returns image point and theta to grasp
class Predictors:
    def __init__(self, I, learner=None):
        self.I = I
        self.I_h, self.I_w, self.I_c = self.I.shape
        self.learner = learner
    def sigmoid_array(self, x):
        return 1 / (1 + np.exp(-x))
    def to_prob_map(self, fc8_vals):
        sig_scale = 1
        no_keep = 20
        fc8_sig = self.sigmoid_array(sig_scale*fc8_vals)
        r = np.sort(fc8_sig, axis = None)
        r_no_keep = r[-no_keep]
        fc8_sig[fc8_sig<r_no_keep] = 0.0
        fc8_prob_map = fc8_sig/fc8_sig.sum()
        return fc8_prob_map
    def sample_from_map(self, prob_map):
        prob_map_contig = np.ravel(prob_map)
        arg_map_contig = np.array(range(prob_map_contig.size))
        smp = np.random.choice(arg_map_contig, p=prob_map_contig)
        return np.unravel_index(smp,prob_map.shape)
    def random_grasp(self, num_angle=18):
        h_g = np.random.randint(self.I_h)
        w_g = np.random.randint(self.I_w)
        t_g = np.random.randint(num_angle)
        return h_g, w_g, t_g
    def center_grasp(self, num_angle=18):
        h_g = np.int(self.I_h/2)
        w_g = np.int(self.I_w/2)
        t_g = np.int(num_angle/2)
        return h_g, w_g, t_g
    def graspNet_grasp(self, num_angle=18, patch_size=300, num_samples=128):
        self.patch_size = patch_size
        half_patch_size = np.int(patch_size/2) + 1
        h_range = self.I_h - patch_size -2
        w_range = self.I_w - patch_size -2

        #Initialize random patch points
        patch_hs = np.random.randint(h_range,size=num_samples) + half_patch_size
        patch_ws = np.random.randint(w_range,size=num_samples) + half_patch_size

        patch_Is = np.zeros((num_samples,patch_size,patch_size,self.I_c))
        patch_Is_resized = np.zeros((num_samples,self.learner.IMAGE_SIZE,self.learner.IMAGE_SIZE,self.I_c))
        for looper in xrange(num_samples):
            isWhiteFlag = 1
            while isWhiteFlag==1:
                patch_hs[looper] = np.random.randint(h_range) + half_patch_size
                patch_ws[looper] = np.random.randint(w_range) + half_patch_size
                patch_Is[looper] = self.I[patch_hs[looper]-half_patch_size:patch_hs[looper]-half_patch_size+patch_size, patch_ws[looper]-half_patch_size:patch_ws[looper]-half_patch_size+patch_size]
                # Make sure that the input has a minimal amount of standard deviation from mean. If
                # not resample
                if patch_Is[looper].std() > 15:
                    isWhiteFlag = 0
                else:
                    isWhiteFlag = 1
            patch_Is_resized[looper] = cv2.resize(patch_Is[looper],(self.learner.IMAGE_SIZE,self.learner.IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        #subtract mean
        patch_Is_resized = patch_Is_resized - 111
        self.fc8_vals = self.learner.test_one_batch(patch_Is_resized)
        # Normalizing angle uncertainity
        wf = [0.25, 0.5, 0.25]
        self.fc8_norm_vals = copy.deepcopy(self.fc8_vals)
        for looper in xrange(num_samples):
            for norm_looper in xrange(num_angle):
                self.fc8_norm_vals[looper,norm_looper] = (wf[1]*self.fc8_norm_vals[looper,norm_looper] +
                        wf[0]*self.fc8_norm_vals[looper,(norm_looper-1)%num_angle] +
                        wf[2]*self.fc8_norm_vals[looper,(norm_looper+1)%num_angle])
        # Normalize to probability distribution
        self.fc8_prob_vals = self.to_prob_map(self.fc8_norm_vals)
        # Sample from probability distribution
        self.patch_id, self.theta_id = self.sample_from_map(self.fc8_prob_vals)
        #self.patch_id, self.theta_id = np.unravel_index(self.fc8_prob_vals.argmax(), self.fc8_prob_vals.shape)
        self.patch_hs = patch_hs
        self.patch_ws = patch_ws
        self.best_patch = patch_Is[self.patch_id]
        self.best_patch_h = patch_hs[self.patch_id]
        self.best_patch_w = patch_ws[self.patch_id]
        self.best_patch_resized = patch_Is_resized[self.patch_id]
        self.patch_Is_resized = patch_Is_resized
        self.patch_Is = patch_Is
        return self.best_patch_h, self.best_patch_w, self.theta_id
