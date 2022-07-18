import numpy as np
import pyrtools as pt
import cv2

    
# format: [flattend stack, colour channels, time index]
def build_lpyr_stack(
    videocapture,
    progress_callback = lambda x: x,
):

    # Read video
    framecount = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))
    vidWidth = int(videocapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    vidHeight = int(videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    retval, temp = videocapture.read()
    frame = temp.astype(np.float32)

    pyr = pt.pyramids.LaplacianPyramid(frame[:,:,0]) # operates on greyscale image
    nLevels = pyr.num_scales
    
    lpyr_stack = np.zeros([sum([i*j for i,j in pyr.pyr_size.values()]),3,framecount]) # construct stack of appropriate size
    lpyr_stack = lpyr_stack.astype(np.float32)
    lpyr_stack[:,0,0] = np.concatenate(
        [im.flatten() for im in pyr.pyr_coeffs.values()])
    lpyr_stack[:,1,0] = np.concatenate(
        [im.flatten() for im in pt.pyramids.LaplacianPyramid(frame[:,:,1]).pyr_coeffs.values()])
    lpyr_stack[:,2,0] = np.concatenate(
        [im.flatten() for im in pt.pyramids.LaplacianPyramid(frame[:,:,2]).pyr_coeffs.values()])

    for k in range(1,framecount):
        retval, temp = videocapture.read()
        frame = temp.astype(np.float32)

        lpyr_stack[:,0,k] = np.concatenate(
            [im.flatten() for im in pt.pyramids.LaplacianPyramid(frame[:,:,0]).pyr_coeffs.values()])
        lpyr_stack[:,1,k] = np.concatenate(
            [im.flatten() for im in pt.pyramids.LaplacianPyramid(frame[:,:,1]).pyr_coeffs.values()])
        lpyr_stack[:,2,k] = np.concatenate(
            [im.flatten() for im in pt.pyramids.LaplacianPyramid(frame[:,:,2]).pyr_coeffs.values()])

        progress_callback(50/framecount)

    videocapture.release()
    return lpyr_stack, np.array(list(pyr.pyr_size.values()))

def iterate_pyramid_levels_fine_to_coarse(pyr_stack, pinds):
    if len(pyr_stack.shape) == 3:
        target_shape = lambda i: (pinds[i][0],pinds[i][1],3,-1)
        target_stack = lambda ix: pyr_stack[ix,:,:]
    else:
        target_shape = lambda i: pinds[i]
        target_stack = lambda ix: pyr_stack[ix]
    nLevels = len(pinds)
    ind = 0
    for k in range(0,nLevels):
        indices = slice(ind, ind + np.prod(pinds[k]))    
        yield target_stack(indices).reshape(target_shape(k)), k
        ind += np.prod(pinds[k])

def iterate_pyramid_levels_coarse_to_fine(pyr_stack, pinds):
    if len(pyr_stack.shape) == 3:
        target_shape = lambda i: (pinds[i][0],pinds[i][1],3,-1)
        target_stack = lambda ix: pyr_stack[ix,:,:]
    else:
        target_shape = lambda i: pinds[i]
        target_stack = lambda ix: pyr_stack[ix]
    nLevels = len(pinds)
    ind = pyr_stack[:,0,0].shape[0] # get size of each flattened stack    
    for k in range(nLevels,0,-1):
        indices = slice(ind - np.prod(pinds[k-1]),ind)

        yield target_stack(indices).reshape(target_shape(k-1)), k-1

        ind = ind - np.prod(pinds[k-1])
