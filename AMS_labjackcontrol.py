#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@AMS_2-0 docstring
Created on Thu Mar  8 12:00:46 2018

@author: vgdev
"""

try: 
    import u3 
except: 
    print('appending u3 library path')
    import sys
    sys.path.insert(0, "/usr/local/lib/python2.7/dist-packages/")
    import u3

from sys import argv
import time 
import numpy as np
import subprocess as sp 
import os
from scipy import misc
from scipy.ndimage.filters import gaussian_filter, laplace, gaussian_laplace
from matplotlib import pyplot as plt

IMG_STACK_SAVE_PATH = 'IMAGE_STACK_TEST/'
capture_img_exe_path = "AMS_SynchronousGrab/Console/Build/Make/binary/x86_64bit/SynchronousGrabConsole"

def connect_labjack():
    '''Documentation for this method
    
    '''
    
    print('connecting to u3 device')
    device = u3.U3() 
    print('getting calibration data' )
    device.getCalibrationData()
    return device



def take_focal_stack(device): 
    ''' documentation for this method 
    
    ''' 
    
    print('setting nominal voltage: 2.5V')
    DACO_val = device.voltageToDACBits(2.5, 0, False)
    device.getFeedback(u3.DAC0_8(DACO_val))
    input('adjust focus, press enter when finished')
    focus_points = np.arange(2.0,3.25,0.25)
    print('focus points: ' + str(focus_points))
    
    print('starting voltage ramp')
    for voltage in focus_points:
        print('voltage: ' + str(voltage))
        DACO_val = device.voltageToDACBits(voltage, 0, False)
        device.getFeedback(u3.DAC0_8(DACO_val))
        time.sleep(0.1)
        #input('waiting for you to choose focus, press enter when done')
        p = sp.Popen(capture_img_exe_path, stdin = sp.PIPE, universal_newlines = True) #universal_newlines means stdin can be a string instead of bytes
        p.communicate(IMG_STACK_SAVE_PATH + "stack-test_" + str(voltage) + ".bmp")


def get_sharpness(img, sigma=10):

  blurred = gaussian_filter(img, sigma)
  #sharp = np.absolute(img - blurred)
  #TAKE THE LOG (laplacian of a guassian) INSTEAD
  sharp = np.absolute(laplace(blurred))
  return sharp

# used in place of np.where because this was running into a memory issue
def get_best_indicies(sharp, best_sharp, best_pixels, frame): 
    shp = sharp.shape
    sharp = sharp.flatten()
    best_sharp = best_sharp.flatten()
    best_pixels = best_pixels.flatten()
    frame = frame.flatten()
    offset = 0
    
    for i in range(len(best_sharp)): 
        if (sharp[i] > best_sharp[i] + offset): 
            best_sharp[i] = sharp[i]
            best_pixels[i] = frame[i]
        
    return best_sharp.reshape(shp), best_pixels.reshape(shp)

def simple_focus_stacking(SIGMA=2): 
    # taken from https://gist.github.com/celoyd/a5f57e8c9d0aedee285fb6f43cb5900c
    stack_names = os.listdir(IMG_STACK_SAVE_PATH)
    imgs = []
    for name in stack_names: 
        imgs.append(misc.imread(IMG_STACK_SAVE_PATH + str(name), flatten=False, mode='F'))

    frame_count = len(stack_names)
    
    max_img = imgs[0]
    for img in imgs[1:]: 
        max_img = np.maximum(max_img, img)
    
    best_pixels = np.mean(imgs, axis=0) # change last val to 3 for color channels
    best_sharpness = get_sharpness(best_pixels, sigma=SIGMA)
    misc.imsave('EXTENDED_DOF_IMGS/' + '2mean' + str(SIGMA) + '.bmp', best_pixels)

    
    for i,frame in enumerate(imgs):
      print ("%s/%s" % (i+1, frame_count))
      
      sharpness = get_sharpness(frame, sigma=SIGMA)
      best_sharpness, best_pixels = get_best_indicies(sharpness, best_sharpness, best_pixels, frame)
    
#    plt.imshow(best_pixels)
#    plt.show()
    misc.imsave('EXTENDED_DOF_IMGS/' + '2extended_DOF_img-sigma=' + str(SIGMA) + '.bmp', best_pixels)
    misc.imsave('EXTENDED_DOF_IMGS/' + '2extended_DOF_img-MAXPIXVAL' + str(SIGMA) + '.bmp', max_img)

def main(): 
    device = connect_labjack()
    take_focal_stack(device) 
    print('image stack acquired')
    simple_focus_stacking()

simple_focus_stacking()    
#main()
#for i in range(1,20,3):
    #simple_focus_stacking(SIGMA=i)