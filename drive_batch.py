#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:38:13 2017

@author: jaerock
"""
#import matplotlib.pyplot as plt
import cv2
import numpy as np
#import keras
#import sklearn
#import resnet
from progressbar import ProgressBar

from net_model import NetModel
from drive_data import DriveData
from config import Config
from image_process import ImageProcess

###############################################################################
#
class DriveBatch:
    
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
    def __init__(self, model_path):
        
        self.model = None
        self.num_test_samples = 0        
        self.config = Config()
        
        self.net_model = NetModel(model_path)
        self.net_model.load()
        
        self.image_process = ImageProcess()


    ###########################################################################
    #
    def _prepare_data(self, data_path):
        
        folder_name = data_path[data_path.rfind('/'):] # get folder name
        folder_name = folder_name.strip('/')
        csv_path = data_path + '/' + folder_name + '.csv' # use it for csv file name 
        self.drive = DriveData(csv_path)

        self.drive.read()
    
        self.test_data = list(zip(self.drive.image_names, self.drive.measurements))
        self.num_test_samples = len(self.test_data)
        
        print('\nTest samples: ', self.num_test_samples)
    

   ###########################################################################
    #
    def run(self, data_path):
        
        self._prepare_data(data_path)
        fname = data_path + '_log.csv'
        
        file = open(fname, 'w')

        #print('image_name', 'label', 'predict', 'abs_error')
        bar = ProgressBar()
        
        file.write('image_name, label, predict, abs_error\n')
        for image_name, measurement in bar(self.test_data):   
            image_fname = data_path + '/' + image_name + self.config.fname_ext 
            image = cv2.imread(image_fname)
            image = cv2.resize(image, (self.config.image_size[0],
                                       self.config.image_size[1]))
            image = self.image_process.process(image)
            
            npimg = np.expand_dims(image, axis=0)
            predict = self.net_model.model.predict(npimg)
            predict = predict / self.config.raw_scale
            
            #print(image_name, measurement[0], predict[0][0],\ 
            #                  abs(measurement[0]-predict[0][0]))
            log = image_name+','+str(measurement[0])+','+str(predict[0][0])\
                            +','+str(abs(measurement[0]-predict[0][0]))
            file.write(log+'\n')
        
        file.close()
        print(fname, 'created.')