import numpy as np


def ideal_bandpassing(input, wl, wh, samplingRate):
            
    # get the framecount
    framecount = input.shape[2]
        
    #set up frequency buckets for the FFT
    Freq = np.arange(1.0,framecount+1)

    Freq = (Freq-1)/framecount*samplingRate

    #Create boolean mask same size as Freq, true in between the frequency limits wl,wh
    mask = (Freq > wl) & (Freq < wh)
    mask = np.broadcast_to(mask,input.shape)
    filtered = np.zeros(input.shape)
    filtered = np.fft.fft(a=input, axis=2)
    filtered[np.logical_not(mask)] = 0
    filtered = np.fft.ifft(a=filtered, axis=2).real
    filtered = filtered.astype(np.float32)
    
    return filtered
