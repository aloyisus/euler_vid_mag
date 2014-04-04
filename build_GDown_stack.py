# -*- coding: utf-8 -*-

import cv2
import numpy as np
import Blur
import sys


#grab next frame in video, convert to float, if necessary convert to YUV, blur and subdivide
def return_next_frame_blurred(vid,level,colourSpace):

	retval,temp = vid.read()
	temp = temp.astype(np.float32)

	if colourSpace == 'yuv':
		temp = cv2.cvtColor(temp,cv2.COLOR_BGR2YUV)

	return Blur.blurDnClr(temp,level)




# GDOWN_STACK = build_GDown_stack(VID_FILE, START_INDEX, END_INDEX, LEVEL)
# 
# Apply Gaussian pyramid decomposition on VID_FILE from START_INDEX to
# END_INDEX and select a specific band indicated by LEVEL
# 
# GDOWN_STACK: stack of one band of Gaussian pyramid of each frame 
# the first dimension is the time axis
# the second dimension is the y axis of the video
# the third dimension is the x axis of the video
# the forth dimension is the color channel
# 
# Copyright (c) 2011-2012 Massachusetts Institute of Technology, 
# Quanta Research Cambridge, Inc.
#
# Authors: Hao-yu Wu, Michael Rubinstein, Eugene Shih, 
# License: Please refer to the LICENCE file
# Date: June 2012
#
def build_GDown_stack(vidFile, startIndex, endIndex, level, colourSpace = 'rgb'):


    # Read video
    vid = cv2.VideoCapture(vidFile)
    fps = vid.get(cv2.cv.CV_CAP_PROP_FPS)
    framecount = vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    vidWidth = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    vidHeight = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))


    #firstFrame
    blurred = return_next_frame_blurred(vid,level,colourSpace)
    
    # create pyr stack
    GDown_stack = np.zeros((endIndex - startIndex +1, blurred.shape[0],blurred.shape[1],blurred.shape[2]))
    GDown_stack[0,:,:,:] = blurred


    for k in range(1,endIndex-startIndex+1):

        #process the video frame and add it to the stack
        GDown_stack[k,:,:,:] = return_next_frame_blurred(vid,level,colourSpace)
        
        #progress indicator
        sys.stdout.write('.')
        sys.stdout.flush()
       
    return GDown_stack
    



def main():
    return



if __name__=="__main__":
    
    main()