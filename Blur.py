# -*- coding: utf-8 -*-


import corrDn
corrDn = corrDn.corrDn
import numpy as np
import namedFilter

import cv2

# RES = blurDn(IM, LEVELS, FILT)
#
# Blur and downsample an image.  The blurring is done with filter
# kernel specified by FILT (default = 'binom5'), which can be a string
# (to be passed to namedFilter), a vector (applied separably as a 1D
# convolution kernel in X and Y), or a matrix (applied as a 2D
# convolution kernel).  The downsampling is always by 2 in each
# direction.
#
# The procedure is applied recursively LEVELS times (default=1).
# Eero Simoncelli, 3/97.
def blurDn(im, nlevs=1, filt='binom5'):

    #if filt is a string, pass it to namedFilter, which returns a 1d kernel
    if isinstance(filt,basestring):
        filt = namedFilter.named_filter(filt)

    #Normalize filt. Applying this more than once has no effect - once it's normalized it's normalized
    filt = filt/np.sum(filt)

    #Recursively call BlurDn, passing the normalized filt, and taking one off nlevs
    if nlevs > 1:
        im = blurDn(im,nlevs-1,filt)


    if (nlevs >= 1):
        #if im is 1d
        if (len(im.shape) == 1):
            if not (1 in filt.shape):
                print 'Cant  apply 2D filter to 1D signal'
                return
                
            if (im.shape[1] == 1):
                filt = filt.flatten()
            else:
                filt = numpy.transpose(filt.flatten())
            
            res = corrDn(im,filt,'reflect1',tuple(map(lambda x: int(not x==1)+1,im.shape)))
        
        #else if im is 2d, but the filter is 1d
        elif (len(filt.shape) == 1):
            filt = filt.flatten()
            res = corrDn(im, filt, 'reflect1', (2,1))
            res = corrDn(res,numpy.transpose(filt), 'reflect1', (1,2))
        else:
            res = corrDn(im, filt, 'reflect1', (2,2))
    
    else:
        res = im
    
    return res



# 3-color version of blurDn.
def blurDnClr(im, nlevs=1, filt='binom5'):

    tmp = blurDn(im[:,:,0], nlevs, filt);
    out = np.zeros((tmp.shape[0], tmp.shape[1], im.shape[2]));
    out[:,:,0] = tmp;
    for clr in range(1,im.shape[2]):
        out[:,:,clr] = blurDn(im[:,:,clr], nlevs, filt);

    
    return out



def main():
    return


if __name__=="__main__":
    
    main()

