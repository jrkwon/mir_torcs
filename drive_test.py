#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017

@author: jaerock
"""

import cv2
import numpy as np
#import keras
import sklearn
from progressbar import ProgressBar

#import resnet
from net_model import NetModel
from drive_data import DriveData
from config import Config
from image_process import ImageProcess

###############################################################################
#
class DriveTest:
    
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
    def __init__(self, model_path):
        
        self.test_generator = None
        self.data_path = None
        
        self.num_test_samples = 0        
        self.config = Config()
        
        self.net_model = NetModel(model_path)
        self.net_model.load()
        
        self.image_process = ImageProcess()


    ###########################################################################
    #
    def _prepare_data(self, data_path):
        
        self.data_path = data_path
        
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
    def _prep_generator(self):
        
        if self.data_path == None:
            raise NameError('data_path must be set.')
            
        def _generator(samples, batch_size=self.config.batch_size):

            num_samples = len(samples)

            while True: # Loop forever so the generator never terminates
                
                bar = ProgressBar()
                
                samples = sklearn.utils.shuffle(samples)
                for offset in bar(range(0, num_samples, batch_size)):

                    batch_samples = samples[offset:offset+batch_size]
        
                    images = []
                    measurements = []
                    for image_name, measurement in batch_samples:
                        image_path = self.data_path + '/' + image_name + \
                                     self.config.fname_ext
                        image = cv2.imread(image_path)
                        image = cv2.resize(image, (self.config.image_size[0],
                                                   self.config.image_size[1]))
                        image = self.image_process.process(image)
                        images.append(image)
        
                        steering_angle, throttle = measurement
                        #angles.append(float(steering_angle))
                        #measurements.append(steering_angle)
                        measurements.append([steering_angle*self.config.raw_scale,
                                             throttle*self.config.raw_scale_throttle])
                        
                    X_train = np.array(images)
                    y_train = np.array(measurements)
                    yield sklearn.utils.shuffle(X_train, y_train)     
                
        self.test_generator = _generator(self.test_data)
        
    
    ###########################################################################
    #
    def _start_test(self):

        if (self.test_generator == None):
            raise NameError('Generators are not ready.')
        
        print("\nEvaluating the model with test data sets ...")
        ## Note: Do not use multiprocessing or more than 1 worker.
        ##       This will genereate threading error!!!
        score = self.net_model.model.evaluate_generator(self.test_generator, 
                                self.num_test_samples//self.config.batch_size) 
                                #workers=1)
        print("\nLoss: ", score)#[0], "Accuracy: ", score[1])
        #print("\nLoss: ", score[0], "rmse: ", score[1])
        
    

   ###########################################################################
    #
    def test(self, data_path):
        self._prepare_data(data_path)
        self._prep_generator()
        self._start_test()
