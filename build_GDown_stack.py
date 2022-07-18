import cv2
import numpy as np
import blur

class GStack:
    def __init__(self):
        self.data = None
        self.size = (0,0)
        self.level = 0

    def build_gdown_stack(
        self,
        videocapture,
        level,
        progress_callback = lambda x: x,
    ):
        self.level = level

        # Read video
        self.count = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))
        vidWidth = int(videocapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        vidHeight = int(videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        #firstFrame
        blurred = return_next_frame_blurred(videocapture, self.level)
        self.size = blurred.shape

        # create pyr stack
        self.data = np.zeros((
                blurred.shape[0]*blurred.shape[1],
                3,
                self.count,
        ))
        for channel in [0,1,2]:
            self.data[:,channel,0] = blurred[:,:,channel].flatten()

        progress_callback(50/self.count)
        for k in range(1,self.count):

            #process the video frame and add it to the stack
            blurred = return_next_frame_blurred(videocapture, self.level)
            for channel in [0,1,2]:        
                self.data[:,channel,k] = blurred[:,:,channel].flatten()

            progress_callback(50/self.count)

        videocapture.release()
        
        return self.data

    def iterate_over_stack_frames(self):
        for k in range(self.count):  
            yield self.data[:, :, k].reshape(self.size[0], self.size[1], 3)

def return_next_frame_blurred(videocapture, level):

	retval,temp = videocapture.read()
	temp = temp.astype(np.float32)
	return blur.blur_dn_clr(temp, level)
