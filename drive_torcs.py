#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 17:29:46 2017

@author: mir-lab
"""

import sys
import threading
import snakeoil
import cv2
import numpy as np
from PIL import Image
import time

from drive_run import DriveRun
from screen import LocalScreenGrab
from pid_controller import PID

prediction = 0

class Thread_Torcs(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
        
    def run(self):
        global prediction
        
#       establish connection with TORCS server
        C = snakeoil.Client()
        
#       load PID controller and set vehicle speed
        pid = PID(1.0, 0.1, 0.1)
        pid.setPoint(15.5)
        try:
            while True:
                C.get_servers_input()
                R = C.R.d
                S = C.S.d
                R['steer'] = prediction
                R['accel'] = pid.update(S['speedX'])
                R['accel'] = np.clip(R['accel'], 0, 0.1)
                snakeoil.drive_example(C)
                C.respond_to_server()
            C.shutdown()
        except KeyboardInterrupt:
            print('\nShutdown requested. Exiting...')
            
class Thread_Prediction(threading.Thread):
    def __init__(self,name):
        threading.Thread.__init__(self)
        self.name = name
        
    def run(self):
        global prediction
        
#       define size of bounding box and pass it to LocalScreenGrab class
        bbox = (65,270,705,410)
        local_grab = LocalScreenGrab(bbox)
        
#       load model
        drive_run = DriveRun(sys.argv[1])
        print('model loaded...')
        try:
            while True:
                arr_screenshot = (local_grab.grab()).reshape(140, 640, 3)
                game_image = cv2.resize(arr_screenshot, (0,0), fx = 0.25, fy = 0.25)
                prediction = (float(drive_run.run(game_image))) / 255
                #print(prediction)
#                if (abs(prediction) < 0.05) is True:
#                    prediction = 0
                #print ('time_taken to get prediction {}'.format(time.time()-last_time))
        except KeyboardInterrupt:
            print('\nShutdown requested. Exiting...')
          
def main():
    try:
        if(len(sys.argv) != 2):
            print('Use model_name')
            return
        Thread_Torcs("TORCS").start()
        Thread_Prediction('Prediction').start()
        
    except KeyboardInterrupt:
            print('\nShutdown requested. Exiting...')
            
if __name__ == '__main__':
    main()
