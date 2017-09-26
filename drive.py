#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:46:15 2017

@author: jaerock
"""

import sys
import cv2

from drive_run import DriveRun
    
#'../mir_torcs_drive_data/2017-05-31-20-49-09' 

###############################################################################
#       
def main():
    try:
        if (len(sys.argv) != 3):
            print('Use model_name image_file_name.')
            return
        
        img = cv2.imread(sys.argv[2])
        if (len(img) == 0):
            print('File open error: ', sys.argv[2])
        
        drive_run = DriveRun(sys.argv[1], img, show_summary = False)
        measurments = drive_run.run() 
        print(measurments)

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
       

###############################################################################
#       
if __name__ == '__main__':
    main()
