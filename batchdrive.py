#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:46:15 2017

@author: jaerock
"""

import sys
#import cv2

from drive_batch import DriveBatch
#from config import Config
    
#'../mir_torcs_drive_data/2017-05-31-20-49-09' 

###############################################################################
#       
def main():
#    config = Config()
    
    try:
        if (len(sys.argv) != 3):
            print('Use model_name drive_data_folder.')
            return
        
        drive_batch = DriveBatch(sys.argv[1]) # pretrained network model name
        drive_batch.run(sys.argv[2]) # data folder path to test

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
       

###############################################################################
#       
if __name__ == '__main__':
    main()
