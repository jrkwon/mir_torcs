import sys
import threading
import snakeoil
import cv2
import numpy as np
import pyscreenshot as ImageGrab
import time

from drive_run import DriveRun

prediction = 0

class Thread_Torcs(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
        
    def run(self):
        global prediction
        C = snakeoil.Client()
        try:
            while True:
                C.get_servers_input()
                R = C.R.d
                R['steer'] = prediction
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
        drive_run = DriveRun(sys.argv[1])
        print('model loaded...')
        try:
            while True:
                last_time = time.time()
                game_image = cv2.resize(cv2.cvtColor(np.array(ImageGrab.grab(bbox=(65,170,380,230))), cv2.COLOR_RGB2YUV),(64,64))
                game_image[:,:,0] = cv2.equalizeHist(game_image[:,:,0])
                game_image = cv2.cvtColor(game_image, cv2.COLOR_YUV2BGR)
                prediction = float(drive_run.run(game_image))
                print ('time_taken to get prediction {}'.format(time.time()-last_time))
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