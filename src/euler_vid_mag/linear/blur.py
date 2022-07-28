from scipy.signal import correlate2d
import numpy as np

from . import named_filter


def blur_dn(im, nlevs=1, filt='binom5'):

    #if filt is a string, pass it to namedFilter, which returns a 1d kernel
    if isinstance(filt,str):
        filt = named_filter.named_filter(filt)

    #Normalize filt. Applying this more than once has no effect - once it's normalized it's normalized
    filt = filt/np.concatenate(filt).sum()

    #Recursively call BlurDn, passing the normalized filt, and taking one off nlevs
    if nlevs > 1:
        im = blur_dn(im,nlevs-1,filt)

    if (nlevs >= 1):
        res = correlate2d(im,filt,boundary='symm')
        res = res[::2,::2]
    else:
        res = im
    
    return res


# 3-color version of blurDn.
def blur_dn_clr(im, nlevs=1, filt='binom5'):

    tmp = blur_dn(im[:,:,0], nlevs, filt);
    out = np.zeros((tmp.shape[0], tmp.shape[1], im.shape[2]));
    out[:,:,0] = tmp;
    for clr in range(1,im.shape[2]):
        out[:,:,clr] = blur_dn(im[:,:,clr], nlevs, filt);

    return out
