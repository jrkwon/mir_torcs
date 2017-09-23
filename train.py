#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:50:52 2017

@author: jaerock
"""

import sys, traceback

from drive_train import DriveTrain
    
#'../mir_torcs_drive_data/2017-05-31-20-49-09' 
        
def main():
    try:
        if (len(sys.argv) != 2):
            print('Give a folder name of drive data.')
            return
        
        drive_train = DriveTrain(sys.argv[1])
        drive_train.train(show_summary=False)    

    except KeyboardInterrupt:
        print ('Shutdown requested. Exiting...')
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)
        

if __name__ == '__main__':
    main()
