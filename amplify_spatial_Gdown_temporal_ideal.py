# -*- coding: utf-8 -*-

# amplify_spatial_Gdown_temporal_ideal(vidFile, outDir, alpha, 
#                                      level, fl, fh, 
#                                      chromAttenuation,colourSpace)
#
# Spatial Filtering: Gaussian blur and down sample
# Temporal Filtering: Ideal bandpass

def amplify_spatial_Gdown_temporal_ideal(vidFile,outDir, alpha,level,
                     fl,fh, chromAttenuation, colourSpace = 'rgb'):
 
    import sys
    import cv2
    import numpy as np
    from Filter import ideal_bandpassing
    from build_GDown_stack import build_GDown_stack
    import os

  
    vidName = os.path.basename(vidFile)
    vidName = vidName[:-4]
    outName = (outDir + vidName + '_' + str(fl) +
                           '-to-' + str(fh) +
                           '-alpha-' + str(alpha) +
                           '-lvl-' + str(level)
                           + colourSpace + '.mov')


    #Read video information
    vid = cv2.VideoCapture(vidFile)
    fr = vid.get(cv2.cv.CV_CAP_PROP_FPS)
    len = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    vidWidth = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    vidHeight = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    startIndex = 0
    endIndex = len-1
    
    
    print 'vidWidth, vidHeight ', str(vidWidth) + ', ' + str(vidHeight)
    print 'fps of captured video ', fr
    print 'length ',len


    # Define the codec and create VideoWriter object
    capSize = (vidWidth,vidHeight) # this is the size of my source video
    fourcc = cv2.cv.CV_FOURCC('j', 'p', 'e', 'g')  # note the lower case
    vidOut = cv2.VideoWriter()
    success = vidOut.open(outName,fourcc,fr,capSize,True)
    
    print outName
    

    # compute Gaussian blur stack
    print 'Spatial filtering...'
    Gdown_stack = build_GDown_stack(vidFile, startIndex, endIndex, level, colourSpace)
    print 'Finished'
    
    
    # Temporal filtering
    print 'Temporal filtering...'
    filtered_stack = ideal_bandpassing(Gdown_stack, 1, fl, fh, fr)
    print 'Finished'
        
    
    ## amplify
    if   colourSpace == 'yuv':
		filtered_stack[:,:,:,0] = filtered_stack[:,:,:,0] * alpha
		filtered_stack[:,:,:,1] = filtered_stack[:,:,:,1] * alpha * chromAttenuation
		filtered_stack[:,:,:,2] = filtered_stack[:,:,:,2] * alpha * chromAttenuation
    elif colourSpace == 'rgb':
        filtered_stack = filtered_stack * alpha
 



    ## Render on the input video
    print 'Rendering...'
    # output video
    for k in range(0,endIndex-startIndex+1):

        retval,temp = vid.read()
        frame = temp.astype(np.float32)
        
        filtered = np.squeeze(filtered_stack[k,:,:,:])          
        filtered = cv2.resize(filtered,(vidWidth, vidHeight),0,0,cv2.INTER_LINEAR)
        
        if   colourSpace == 'yuv':
			frame = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
			frame[:,:,1:] = frame[:,:,1:] + filtered[:,:,1:] 
			frame = cv2.cvtColor(frame,cv2.COLOR_YUV2BGR)
        elif colourSpace == 'rgb':
            frame = frame + filtered
               
        frame = np.clip(frame,0,255)
        frame = cv2.convertScaleAbs(frame)

        vidOut.write(frame)
        sys.stdout.write('.')
        sys.stdout.flush()


    print 'Finished'
    vid.release()
    vidOut.release() 





def main():
    return



if __name__=="__main__":
    
    main()