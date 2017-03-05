#!/usr/bin/env python
import random
import os
import tensorflow as tf
import numpy as np
import cv2
import pickle
import scipy
from scipy import ndimage
import time

class grasp_shake_data:
        def __init__(self, data_dir, batch_size=128, isTrain=True, posFraction=0.5):
                self.BATCH_SIZE = batch_size
                self.num_pos = int(self.BATCH_SIZE * posFraction)
                self.num_neg = int(self.BATCH_SIZE - self.num_pos)
                self.data_dir = data_dir
                self.IMAGE_W = 224
                self.IMAGE_H = 224
                self.IMAGE_C = 3

                self.pos_filenames = []
                self.pos_labels = []
                self.neg_filenames = []
                self.neg_labels = []
                for grasp_folder in os.listdir(self.data_dir):
                    cur_dir = os.path.join(self.data_dir,grasp_folder)
                    cur_grasp_pickle = os.path.join(cur_dir,'grasp_info.p')
                    cur_shake_pickle = os.path.join(cur_dir,'shake_info.p')
                    cur_img = os.path.join(cur_dir,'approach.jpg')
                    p_h,p_w,p_t,p_s,p_success = self.pickle_to_patch(cur_grasp_pickle)
                    if os.path.isfile(cur_shake_pickle) == True:
                        s_a, s_success = self.pickle_to_action(cur_shake_pickle)
                        self.pos_filenames.append(cur_img)
                        self.pos_labels.append([p_h,p_w,p_t,p_s,s_a,s_success])
                    elif p_success == False:
                        self.neg_filenames.append(cur_img)
                        self.neg_labels.append([p_h,p_w,p_t,p_s])
                self.total_pos = len(self.pos_filenames)
                self.total_neg = len(self.neg_filenames)
                # Shuffle data
                pos_nums = range(self.total_pos)
                pos_inds = np.random.permutation(pos_nums)
                self.pos_filenames = np.array(self.pos_filenames)[pos_inds]
                self.pos_labels = np.array(self.pos_labels)[pos_inds]
                self.pos_looper = 0
                neg_nums = range(self.total_neg)
                neg_inds = np.random.permutation(neg_nums)
                self.neg_filenames = np.array(self.neg_filenames)[neg_inds]
                self.neg_labels = np.array(self.neg_labels)[neg_inds]
                self.neg_looper = 0

        def pickle_to_patch(self, cur_pickle):
            grasp_info_dict = pickle.load( open( cur_pickle, "rb" ) )
            return grasp_info_dict['patch_h'], grasp_info_dict['patch_w'],grasp_info_dict['patch_theta'],grasp_info_dict['patch_size'],grasp_info_dict['success']

        def pickle_to_action(self, cur_pickle):
            shake_info_dict = pickle.load( open( cur_pickle, "rb" ) ) 
            return shake_info_dict['action_id'], shake_info_dict['shake_success']

        def get_batch(self):
                I_batch = np.zeros((self.BATCH_SIZE,self.IMAGE_W,self.IMAGE_H,self.IMAGE_C))
                I_shake = np.zeros((self.num_pos,self.IMAGE_W,self.IMAGE_H,self.IMAGE_C))
                theta_labels = np.zeros(self.BATCH_SIZE).astype(int)
                success_labels = np.zeros(self.BATCH_SIZE).astype(int)

                for looper in xrange(self.num_pos):
                        curI = cv2.imread(self.pos_filenames[self.pos_looper])
                        p_h,p_w,p_t,p_s,_,_ = self.pos_labels[self.pos_looper]
                        new_theta_label = np.random.randint(18)
                        new_theta_float = -float(new_theta_label)*np.pi/18 + np.pi/2 - np.pi/(2*18)
                        p_theta = p_t*np.pi/18 - np.pi/2 + np.pi/(2*18)
                        new_p_theta = p_theta + new_theta_float
                        patI = self.rotateImageAndExtractPatch(curI, new_p_theta, [p_h,p_w], p_s)
                        patI = self.preprocess(patI)
                        shakeI = self.rotateImageAndExtractPatch(curI, p_theta, [p_h,p_w], p_s)
                        shakeI = self.preprocess(shakeI)
                        curG = new_theta_label
                        curS = 1
                        I_batch[looper] = patI
                        theta_labels[looper] = curG
                        success_labels[looper] = curS
                        I_shake[looper] = shakeI
                        self.pos_looper = (self.pos_looper + 1)%self.total_pos
                for looper in xrange(self.num_neg):
                        curI = cv2.imread(self.neg_filenames[self.neg_looper])
                        p_h,p_w,p_t,p_s = self.neg_labels[self.neg_looper]
                        new_theta_label = np.random.randint(18)
                        new_theta_float = -float(new_theta_label)*np.pi/18 + np.pi/2 - np.pi/(2*18)
                        p_theta = p_t*np.pi/18 - np.pi/2 + np.pi/(2*18)
                        new_p_theta = p_theta + new_theta_float
                        patI = self.rotateImageAndExtractPatch(curI, new_p_theta, [p_h,p_w], p_s)
                        patI = self.preprocess(patI)
                        curG = new_theta_label
                        curS = 0
                        I_batch[self.num_pos + looper] = patI
                        theta_labels[self.num_pos + looper] = curG
                        success_labels[self.num_pos + looper] = curS
                        self.neg_looper = (self.neg_looper + 1)%self.total_neg
                return I_batch, theta_labels, success_labels, I_shake

        def preprocess(self,im):
                im = im.astype(float)
                im = cv2.resize(im,(self.IMAGE_H,self.IMAGE_W), interpolation=cv2.INTER_CUBIC)
                im = im - 111
                im = im/144 # To scale from -1 to 1
                return im
        def rotateImageAndExtractPatch(self, img, angle, center, size):
            angle = angle*180/np.pi
            padX = [img.shape[1] - center[1], center[1]]
            padY = [img.shape[0] - center[0], center[0]]
            imgP = np.pad(img, [padY, padX, [0,0]], 'constant')
            imgR = scipy.misc.imrotate(imgP, angle)
            #imgR = ndimage.rotate(imgP, angle, reshape=False, order =1)
            half_size = int(size/2)
            return imgR[padY[0] + center[0] - half_size: padY[0] + center[0] + half_size, padX[0] + center[1] - half_size : padX[0] + center[1] + half_size, :]

