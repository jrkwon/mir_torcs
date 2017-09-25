#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:49:23 2017

@author: jaerock
"""


import sys

from drive_test import DriveTest
    
#'../mir_torcs_drive_data/2017-05-31-20-49-09' 

###############################################################################
#       
def main():
    try:
        if (len(sys.argv) != 3):
            print('Use model_name folder_name_to_drive_data.')
            return
        
        drive_test = DriveTest(sys.argv[1], sys.argv[2])
        drive_test.test(show_summary = False)    

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
       

###############################################################################
#       
if __name__ == '__main__':
    main()
