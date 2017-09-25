#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017

@author: jaerock
"""

import cv2
import numpy as np
import keras
import sklearn
from progressbar import ProgressBar

import resnet
from drive_data import DriveData
from config import Config


###############################################################################
#
class DriveTest:
    
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
    def __init__(self, model_path, data_path):
        
        folder_name = data_path[data_path.rfind('/'):] # get folder name
        folder_name = folder_name.strip('/')
        csv_path = data_path + '/' + folder_name + '.csv' # use it for csv file name 
        
        self.csv_path = csv_path

        self.model = None        
        self.test_generator = None
        self.num_test_samples = 0        
        self.config = Config()
        
        self.data_path = data_path
        self.model_path = model_path
        
        self.drive = DriveData(self.csv_path)


    ###########################################################################
    #
    def _prepare_data(self):
        
        self.drive.read()
    
        self.test_data = list(zip(self.drive.image_names, self.drive.measurements))
        self.num_test_samples = len(self.test_data)
        
        print('Test samples: ', self.num_test_samples)
    
    
    ###########################################################################
    # This model must be same as the one in DriveTrain
    def _model(self):
        return resnet.ResnetBuilder.build_resnet_50(
                    (self.config.image_size[2], 
                     self.config.image_size[1], 
                     self.config.image_size[0]), 
                     self.config.num_outputs)
                    
    ##########################################################################
    # This model compile must be same as the one in DriveTrain
    def _model_compile(self):
        self.model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    
    ###########################################################################
    #
    def _build_model(self, show_summary=True):
        
        def _generator(samples, batch_size=self.config.batch_size):

            num_samples = len(samples)

#            import threading                 
#            lock = threading.Lock()

            while True: # Loop forever so the generator never terminates
                
                bar = ProgressBar()
                
                samples = sklearn.utils.shuffle(samples)
                for offset in bar(range(0, num_samples, batch_size)):
#                    lock.acquire()

                    batch_samples = samples[offset:offset+batch_size]
        
                    images = []
                    measurements = []
                    for image_name, measurement in batch_samples:
                        image_path = self.data_path + '/' + image_name + \
                                     self.config.fname_ext
                        image = cv2.imread(image_path)
                        images.append(image)
        
                        steering_angle, throttle = measurement
                        #angles.append(float(steering_angle))
                        measurements.append(measurement)
                        
#                        print('image_path: ', image_path)
#                        print('measurement: ', measurement)
#                        import matplotlib.pyplot as plt
#                        plt.imshow(image)
                        
                    X_train = np.array(images)
                    y_train = np.array(measurements)
                    yield sklearn.utils.shuffle(X_train, y_train)     
                
#                    lock.release()
                
        self.test_generator = _generator(self.test_data)
        
        self._load_model()
        
        if (show_summary):
            self.model.summary()
    
    ###########################################################################
    #
    def _start_test(self):
        
        if (self.test_generator == None):
            raise NameError('Generators are not ready.')
        
        print("\nEvaluating the model with test data sets ...")
        score = self.model.evaluate_generator(self.test_generator, 
                                self.num_test_samples//self.config.batch_size) 
                                #workers=1)
        print("\nLoss: ", score[0], "Accuracy: ", score[1])
        
    
    ###########################################################################
    # model_path = '../data/2007-09-22-12-12-12.
    def _load_model(self):
        
        from keras.models import model_from_json
        
        self.model = model_from_json(open(self.model_path+'.json').read())
        self.model.load_weights(self.model_path+'.h5')
        self._model_compile()
        

   ###########################################################################
    #
    def test(self, show_summary = True):
        
        self._prepare_data()
        self._build_model(show_summary)
        self._start_test()
