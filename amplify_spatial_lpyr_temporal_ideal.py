from filter import ideal_bandpassing
from build_lpyr_stack import *
import numpy as np
import pyrtools as pt
import cv2


def process_input(
    filename,
    alpha,
    lambda_c,
    fl,
    fh,
    sample_rate,
    progress_callback = lambda x: x
):
    #Read video information
    vid = cv2.VideoCapture(filename)
    framecount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vidWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vidHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # compute Laplacian pyramid for each frame
    pyr_stack, pind = build_lpyr_stack(vid, progress_callback)

    # apply temporal filter
    filtered_stack=ideal_bandpassing(pyr_stack,fl,fh,sample_rate)

    nLevels=len(pind)
    delta=lambda_c / 8 / (1 + alpha)
    
    # the factor to boost alpha above the bound we have in the
    # paper. (for better visualization)
    exaggeration_factor = 2;
    
    # compute the representative wavelength lambda for the lowest spatial 
    # freqency band of Laplacian pyramid
    lambda_ = (vidHeight ** 2 + vidWidth ** 2) ** 0.5 / 3

    for level, num in iterate_pyramid_levels_coarse_to_fine(filtered_stack,pind):
        # compute modified alpha for this level
        currAlpha = lambda_ / delta / 8 - 1
        currAlpha = currAlpha*exaggeration_factor        

        if (num == nLevels-1 or num == 0): # ignore the highest and lowest frequency band
            level[:,:,:] = 0
        elif (currAlpha > alpha): # representative lambda exceeds lambda_c
            level[:,:,:] = alpha * level
        else:
            level[:,:,:] = currAlpha * level 

        # go one level down on pyramid, 
        # representative lambda will reduce by factor of 2
        lambda_=lambda_ / 2

    vid.release()
    vid = cv2.VideoCapture(filename)  

    ## Render on the input video
    for i in range(0,framecount):
        retval,temp = vid.read()
        frame = temp.astype(np.float32)

        filtered = frame.copy()

        # Reconstruct the filtered frame from the individual
        # pyramid layers. This is an ugly hack for which I am ashamed.
        for channel in [0,1,2]:
            pyr = pt.pyramids.LaplacianPyramid(frame[:,:,channel])
            for level,num in iterate_pyramid_levels_fine_to_coarse(filtered_stack[:,channel,i], pind):
               pyrkey = (num,0)
               pyr.pyr_coeffs[pyrkey] = level
            filtered[:,:,channel] = pyr.recon_pyr()

        frame = frame + filtered

        frame = np.clip(frame,0,255)
        frame = cv2.convertScaleAbs(frame)

        progress_callback(50/framecount)

        yield frame

    print('Finished')
    vid.release()
