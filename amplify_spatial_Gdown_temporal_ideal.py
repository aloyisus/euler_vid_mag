import cv2
import numpy as np
from filter import ideal_bandpassing
from build_gdown_stack import *


def process_input(
    filename,
    alpha,
    level,
    fl,
    fh,
    sampling_rate,
    progress_callback = lambda x: x,
):
    #Read video information
    vid = cv2.VideoCapture(filename)
    framecount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vidWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vidHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    gstack = GStack()

    # compute Gaussian blur stack
    print('Spatial filtering...')
    gstack.build_gdown_stack(vid, level, progress_callback)
    print('Finished')

    # Temporal filtering
    print('Temporal filtering...')
    gstack.data = ideal_bandpassing(gstack.data, fl, fh, sampling_rate)
    print('Finished')

    ## amplify
    gstack.data[:,:,:] = alpha * gstack.data

    vid.release()
    vid = cv2.VideoCapture(filename)    

    ## Render on the input video
    print('Rendering...')
    # output video
    for filtered in gstack.iterate_over_stack_frames():
        retval,temp = vid.read()
        frame = temp.astype(np.float32)

        filtered = cv2.resize(filtered,(vidWidth, vidHeight),0,0,cv2.INTER_CUBIC)
        frame = frame + filtered
        # frame = filtered

        frame = np.clip(frame,0,255)
        frame = cv2.convertScaleAbs(frame)
        progress_callback(50/framecount)
        yield frame

    print('Finished')
    vid.release()
