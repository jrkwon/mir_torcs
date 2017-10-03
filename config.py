#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 21:30:08 2017

@author: jaerock
"""

###############################################################################
#
class Config:
    def __init__(self): # model_name):
        self.version = (0, 2) # version 0.2
        self.image_size = (64, 64, 3) #(320, 70, 3)
        self.num_outputs = 1  # steering_angle, throttle
        #self.model_name = model_name # 'torcs_2017-05-31-20-49-09'
        self.fname_ext = '.jpg'
        self.num_epochs = 20
        self.batch_size = 64
