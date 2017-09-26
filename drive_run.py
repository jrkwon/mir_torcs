#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017

@author: jaerock
"""

import keras
import numpy as np

from config import Config


###############################################################################
#
class DriveRun:
    
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
    def __init__(self, model_path, img, show_summary=True):
        
        self.model = None        
        self.config = Config()
        
        self.model_path = model_path
        self.img = img
        
        self._load_model(show_summary)
            
    
    ##########################################################################
    # This model compile must be same as the one in DriveTrain
    def _model_compile(self):
        self.model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    
    ###########################################################################
    # model_path = '../data/2007-09-22-12-12-12.
    def _load_model(self, show_summary):
        
        from keras.models import model_from_json
        
        self.model = model_from_json(open(self.model_path+'.json').read())
        self.model.load_weights(self.model_path+'.h5')
        if (show_summary == True):
            self._model_compile()
        

   ###########################################################################
    #
    def run(self):
        npimg = np.expand_dims(self.img, axis=0)
        measurements = self.model.predict(npimg)
        return measurements
