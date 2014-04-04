# FILTERED = ideal_bandpassing(INPUT,DIM,WL,WH,SAMPLINGRATE)
# 
# Apply ideal band pass filter on INPUT along dimension DIM.
# 
# WL: lower cutoff frequency of ideal band pass filter
# WH: higher cutoff frequency of ideal band pass filter
# SAMPLINGRATE: sampling rate of INPUT
# 
# Copyright (c) 2011-2012 Massachusetts Institute of Technology, 
# Quanta Research Cambridge, Inc.
#
# Authors: Hao-yu Wu, Michael Rubinstein, Eugene Shih, 
# License: Please refer to the LICENCE file
# Date: June 2012
#

# -*- coding: utf-8 -*-


import numpy as np
import cv2

   
def shiftdim(x, n):  
    return x.transpose(np.roll(range(x.ndim), -n))


def repmat(a,m):
    #First, pad out a so it has same dimensionality as m
    for i in range(0,m.ndim-a.ndim):
        a = np.expand_dims(a,1)
    #Now just use numpy tile and return result
    return np.tile(a,m.shape)


#The sampling rate is a frequency. In most of the examples it is 30, which is NTSC video freq.
#wl and wh in examples given seen to be temporal frequencies also - in baby2 example
#wl = 140/60 and wh = 160/60. The baby's heartbeat is approx 150hz
def ideal_bandpassing(input, dim, wl, wh, samplingRate):

    #if dim is greater than the dimensionality (2d, 3d etc) of the input, quit
    if (dim > len(input.shape)):
        print 'Exceed maximum dimension'
        return
        
    #This has the effect that input_shifted[0] = input[dim]
    input_shifted = shiftdim(input,dim-1)
         
    #Put the dimensions of input_shifted in a 1d array
    Dimensions = np.asarray(input_shifted.shape)
            
    #how many things in the first dimension of input_shifted
    n = Dimensions[0]
    
    #get the dimensionality (eg. 2d, 3d etc) of input_shifted
    dn = input_shifted.ndim
        
    #creates a vector [1,...,n], the same length as the first dimension of input_shifted
    Freq = np.arange(1.0,n+1)
            
    #Equivalent in python: Freq = (Freq-1)/n*samplingRate
    Freq = (Freq-1)/n*samplingRate
           
    #Create boolean mask same size as Freq, true in between the frequency limits wl,wh
    mask = (Freq > wl) & (Freq < wh)
 
    Dimensions[0] = 1
    mask = repmat(mask,np.ndarray(Dimensions))

    #F = fft(X,[],dim) and F = fft(X,n,dim) applies the FFT operation across the dimension dim.
    #Python: F = np.fftn(a=input_shifted,axes=0)
    F = np.fft.fftn( a=input_shifted, axes=[0] )
    
    #So we are indexing array F using boolean not mask, and setting those values of F to zero, so the others pass thru
    #Python: F[ np.logical_not(mask) ]
    F[ np.logical_not(mask) ] = 0
    
    #Get the real part of the inverse fourier transform of the filtered input
    filtered = np.fft.ifftn( a=F, axes=[0] ).real
    
    filtered = filtered.astype(np.float32)
    
    filtered = shiftdim(filtered,dn-(dim-1))
    
    return filtered
    

def main():
    return


if __name__=="__main__":
    
    main()
