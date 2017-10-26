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

from drive_run import DriveRun
from screen import LocalScreenGrab
from config import Config
from image_process import ImageProcess

use_threading = True

prediction = 0
is_driving = False
 
class ThreadTorcs(threading.Thread):
    
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
        self.config = Config()
       
    def run(self):
        global prediction, is_driving
       
        # establish connection with TORCS server
        client = snakeoil.Client()
       
        is_driving = True
        while True:
            if client.get_servers_input() == False:
                break
            
            R = client.R.d
            
            R['steer'] = -prediction      
            snakeoil.drive(client)
            client.respond_to_server()
            
        client.shutdown()
        is_driving = False
        
           
class ThreadPrediction(threading.Thread):
    
    def __init__(self,name):
        threading.Thread.__init__(self)
        self.name = name
        self.config = Config()
        self.image_process = ImageProcess()

       
    def run(self):
        global prediction, is_driving
       
        # load model
        drive = DriveRun(sys.argv[1])
        print('model loaded...')
        
        while True:
           # define size of bounding box and pass it to LocalScreenGrab class
            bbox = self.config.capture_area
            local_grab = LocalScreenGrab(bbox)
            
            screenshot = (local_grab.grab()).reshape(self.config.capture_size)
            image = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)     
            image = cv2.resize(image, (self.config.image_size[0],
                                       self.config.image_size[1]))
            image = self.image_process.process(image)
            #        cv2.imwrite('a.jpg', image)
            
            prediction = drive.run(image)
            if abs(prediction) < self.config.jitter_tolerance:
                prediction = 0
                
            if is_driving == False:
                break
         

def run(drive):
    config = Config()
    image_process = ImageProcess()

    # establish connection with TORCS server
    client = snakeoil.Client()

    while True: 
        if client.get_servers_input() == False:
            break
        R = client.R.d

        # define size of bounding box and pass it to LocalScreenGrab class
        bbox = config.capture_area
        local_grab = LocalScreenGrab(bbox)
        
        screenshot = (local_grab.grab()).reshape(config.capture_size)
        image = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)     
        image = cv2.resize(image, (config.image_size[0],
                                        config.image_size[1]))
        image = image_process.process(image)

        
        prediction = drive.run(image)
        if abs(prediction) < config.jitter_tolerance:
            prediction = 0
            
        R['steer'] = -prediction
        snakeoil.drive(client)
        client.respond_to_server()
    client.shutdown()
        
    
def main():
    try:
        if(len(sys.argv) != 2):
            print('Use model_name')
            return
        
        if use_threading == True:
            ThreadTorcs('TORCS').start()
            ThreadPrediction('Prediction').start()
        else:
            # load model
            drive = DriveRun(sys.argv[1])
            print('model loaded...')
            
            run(drive)

        
    except KeyboardInterrupt:    
        print('\nShutdown requested. Exiting...')
            
if __name__ == '__main__':
    main()
