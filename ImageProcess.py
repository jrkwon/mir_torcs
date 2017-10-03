#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:09:03 2017

@author: jaerock
"""

import cv2

class ImageProcess:

    # img is expected as BGR         
    def equalize_histogram(self, img, bgr = True):

        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        if (bgr == True):
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        else:
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return img