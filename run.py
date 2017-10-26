#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:46:15 2017

@author: jaerock
"""

import sys
import cv2

from drive_run import DriveRun
from config import Config
from image_process import ImageProcess
    
#'../mir_torcs_drive_data/2017-05-31-20-49-09' 

###############################################################################
#       
def main():
    config = Config()
    image_process = ImageProcess()
    
    try:
        if (len(sys.argv) != 3):
            print('Use model_path image_file_name.')
            return
        
        image = cv2.imread(sys.argv[2])
        image = cv2.resize(image, (config.image_size[0],
                                   config.image_size[1]))
        image = image_process.process(image)

        
        if (len(image) == 0):
            print('File open error: ', sys.argv[2])
            return
        
        drive_run = DriveRun(sys.argv[1])
        measurments = drive_run.run(image) 
        print(measurments)

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
       

###############################################################################
#       
if __name__ == '__main__':
    main()
