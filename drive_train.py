#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 21:29:53 2017

@author: jaerock
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import keras
import sklearn
import resnet

from drive_data import DriveData
from config import Config


###############################################################################
#
class DriveTrain:
    
    ###########################################################################
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
    def __init__(self, data_path):
        
        model_name = data_path[data_path.rfind('/'):] # get folder name
        model_name = model_name.strip('/')
        csv_path = data_path + '/' + model_name + '.csv' # use it for csv file name 
        
        self.csv_path = csv_path
        self.model = None        
        self.train_generator = None
        self.valid_generator = None
        self.train_hist = None
        self.drive = None
        
        self.config = Config() #model_name)
        
        self.data_path = data_path
        self.model_name = model_name
        
        self.drive = DriveData(self.csv_path)
        
        
    ###########################################################################
    #
    def _prepare_data(self):
    
        self.drive.read()
        
        from sklearn.model_selection import train_test_split
        
        samples = list(zip(self.drive.image_names, self.drive.measurements))
        self.train_data, self.valid_data = train_test_split(samples, test_size=0.3)
        
        self.num_train_samples = len(self.train_data)
        self.num_valid_samples = len(self.valid_data)
        
        print('Train samples: ', self.num_train_samples)
        print('Valid samples: ', self.num_valid_samples)
    
    
    ###########################################################################
    #
    def _model(self):
        return resnet.ResnetBuilder.build_resnet_50(
                    (self.config.image_size[2], 
                     self.config.image_size[1], 
                     self.config.image_size[0]), 
                     self.config.num_outputs)
                    
    ##########################################################################
    #
    def _model_compile(self):
        self.model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

                                        
    ###########################################################################
    #
    def _build_model(self, show_summary=True):
        
        def _generator(samples, batch_size=self.config.batch_size):
            num_samples = len(samples)
            while True: # Loop forever so the generator never terminates
                samples = sklearn.utils.shuffle(samples)
                for offset in range(0, num_samples, batch_size):
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
        
                        # add the flipped image of the original
                        images.append(cv2.flip(image,1))
                        measurement = (steering_angle*-1.0, measurement[1]) 
                        measurements.append(measurement)
        
                    X_train = np.array(images)
                    y_train = np.array(measurements)
                    yield sklearn.utils.shuffle(X_train, y_train)
        
        self.train_generator = _generator(self.train_data)
        self.valid_generator = _generator(self.valid_data)
        
        self.model = self._model()
        self._model_compile()

        if (show_summary):
            self.model.summary()
    
    ###########################################################################
    #
    def _start_training(self):
        
        if (self.train_generator == None):
            raise NameError('Generators are not ready.')
        
        ######################################################################
        # callbacks
        from keras.callbacks import ModelCheckpoint, EarlyStopping
        
        # checkpoint
        callbacks = []
        checkpoint = ModelCheckpoint(self.model_name+'.h5', monitor='val_acc', 
                                     verbose=1, save_best_only=True, mode='min')
        callbacks.append(checkpoint)
        
        # early stopping
        earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=0, 
                                  verbose=1, mode='min')
        callbacks.append(earlystop)
        
        self.train_hist = self.model.fit_generator(
                            self.train_generator, 
                            steps_per_epoch=self.num_train_samples//self.config.batch_size, 
                            epochs=self.config.num_epochs, 
                            validation_data=self.valid_generator,
                            validation_steps=self.num_valid_samples//self.config.batch_size,
                            verbose=1, callbacks=callbacks)
    
    ###########################################################################
    #
    # save model
    def _save_model(self):
        
        json_string = self.model.to_json()
        open(self.data_path+'.json', 'w').write(json_string)
        self.model.save_weights(self.data_path+'.h5', overwrite=True)
    
    
    ###########################################################################
    # model_path = '../data/2007-09-22-12-12-12.
    def load_model(self):
        
        from keras.models import model_from_json
        
        self.model = model_from_json(open(self.data_path+'.json').read())
        self.model.load_weights(self.data_path+'.h5')
        self._model_compile()
        

    ###########################################################################
    #
    def _plot_training_history(self):
    
        print(self.train_hist.history.keys())
        
        ### plot the training and validation loss for each epoch
        plt.plot(self.train_hist.history['loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set'], loc='upper right')
        plt.show()
        
    ###########################################################################
    #
    def train(self, show_summary=True):
        
        self._prepare_data()
        self._build_model(show_summary)
        self._start_training()
        self._save_model()
        self._plot_training_history()
