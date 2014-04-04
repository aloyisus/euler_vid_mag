# -*- coding: utf-8 -*-

import scipy.signal as sc


# RES = corrDn(IM, FILT, EDGES, STEP, START, STOP)
#
# Compute correlation of matrices IM with FILT, followed by
# downsampling. 
#
# Downsampling factors are determined by STEP (optional, default=[1 1]), 
# which should be a 2-vector [y,x].
# 
# The window over which the convolution occurs is specfied by START 
# (optional, default=[1,1], and STOP (optional, default=size(IM)).
# 
# NOTE: this operation corresponds to multiplication of a signal
# vector by a matrix whose rows contain copies of the FILT shifted by
# multiples of STEP.


def corrDn(im, filt, edges='reflect1', step=(1,1), start=(0,0), stop=0):

    #default value of stop is size of image
    if stop==0:
        stop = im.shape

    # Reverse order of taps in filt, to do correlation instead of convolution
    filt = filt[::-1,::-1]

    #convolution here
    tmp = sc.convolve2d(im,filt,mode='valid',boundary = 'symm')
        
    #this is the downsampling line
    res = tmp[start[0]:stop[0]+1:step[1],start[1]:stop[1]+1:step[1]]
    
    return res
    
    
    
def main():
    return



if __name__=="__main__":
    
    main()