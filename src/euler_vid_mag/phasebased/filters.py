import cv2
import numpy as np
from scipy.signal import firwin


def getFilters(dimension, rVals, orientations, twidth=1):
    filters = []
    angle, log_rad = getPolarGrid(dimension) # Get polar coordinates of frequency plane
    himask, lomaskPrev = getRadialMaskPair(rVals[0], log_rad, twidth)
    filters.append(himask)
    for k in range(1,len(rVals)):
        himask, lomask = getRadialMaskPair(rVals[k], log_rad, twidth)
        radMask = np.multiply(himask, lomaskPrev)
        for j in range(0, orientations):
            anglemask = getAngleMask(j, orientations, angle)
            filters.append(np.multiply(radMask, anglemask/2))
        lomaskPrev = lomask
    filters.append(lomask)
    return filters

def getPolarGrid( dimension ):
    center = np.floor((np.array(dimension)+0.5)/2).astype(int)
    # Create rectangular grid
    xramp, yramp = np.meshgrid(
        (np.arange(dimension[1])-center[1])/(dimension[1]/2),
        (np.arange(dimension[0])-center[0])/(dimension[0]/2)
    )  
    # Convert to polar coordinates
    angle = np.arctan2(yramp, xramp)
    rad = np.sqrt(np.power(xramp,2) + np.power(yramp,2))
    # Eliminate places where rad is zero, so logarithm is well defined
    rad[center[0], center[1]] =  rad[center[0], center[1]-1]
    return angle, rad

def getRadialMaskPair(r, rad, twidth):
    log_rad  = np.log2(rad)-np.log2(r)
    himask = log_rad
    himask = np.clip(himask, -twidth, 0)
    himask = himask * np.pi/(2 * twidth)
    himask = np.abs(np.cos(himask))     
    lomask = np.sqrt(1 - himask**2)    
    return himask, lomask

def getAngleMask(b,  orientations, angle):
    order = orientations-1
    const = (2**(2*order))*(np.math.factorial(order)**2)/(orientations*np.math.factorial(2*order)) # Scaling constant
    angle = np.mod(np.pi+angle - np.pi*(b)/orientations,2*np.pi)-np.pi # Mask angle mask
    anglemask = np.multiply(2*np.sqrt(const)*np.cos(angle)**order, (np.abs(angle)<np.pi/2))  # Make falloff smooth
    return anglemask

def getAngleMaskSmooth(b, nbands, angle, complexFilt=None):
    ## Computes anglemask for getRadialFilters3
    order = nbands-1
    const = (2**(2*order))*(np.math.factorial(order)**2)/(nbands*np.math.factorial(2*order))
    angle = np.mod(np.pi+angle - np.pi*(b)/nbands,2*np.pi)-np.pi   
    if complexFilt:
        anglemask = np.multiply(np.sqrt(const)*np.power(np.cos(angle), order), (np.abs(angle)<np.pi/2))
    else:
        anglemask = np.abs(np.power(np.sqrt(const)*np.cos(angle), order))
    return anglemask

def maxSCFpyrHt(im):
    maxHeight = np.floor(np.log2(np.min(im.shape))) - 2
    return maxHeight

def getFiltersSmoothWindow(dims,  orientations, cosOrder=6, filtersPerOctave=6, complex=True, height=None):
    if not height:
        height = maxSCFpyrHt(np.zeros(dims))

    complexFilt = complex
    htOct = height

    angle, rad = getPolarGrid(dims)

    rad = np.log2(rad)
    rad = (htOct+rad)/htOct
    filts = filtersPerOctave*htOct
    rad = rad*(np.pi/2+np.pi/7*filts)

    def windowFnc(x, center):
        return np.abs(x-center) < np.pi/2

    radFilters = []
    total = np.zeros(dims)
    const = (2**(2*cosOrder))*(np.math.factorial(cosOrder)**2)/((cosOrder+1)*np.math.factorial(2*cosOrder))
    for k in range(int(filts),0,-1):
        shift = np.pi/(cosOrder+1)*k+2*np.pi/7
        radFilters.append(np.sqrt(const)*np.multiply(np.power(np.cos(rad-shift), cosOrder), windowFnc(rad,shift)))
        total = total + np.power(radFilters[-1], 2)
    total[total > 1.0] = 1.0

    # Compute lopass residual
    center = np.floor((np.array(dims)+0.5)/2).astype(int)
    lodims = np.ceil((center+0.5)/4).astype(int)
    # We crop the sum image so we don't also compute the high pass
    totalCrop = total[center[0]-lodims[0]:center[0]+lodims[0]+1,center[1]-lodims[1]:center[1]+lodims[1]+1]
    lopass = np.zeros(dims)
    lopass[center[0]-lodims[0]:center[0]+lodims[0]+1,center[1]-lodims[1]:center[1]+lodims[1]+1] = np.abs(np.sqrt(1-totalCrop))

    # Compute high pass residual
    total = total + np.power(lopass, 2)
    total[total > 1.0] = 1.0
    hipass = np.abs(np.sqrt(1-total))

    # If either dimension is even, this fixes some errors
    if np.mod(dims[0], 2) == 0: #even
        for k in range(0, len(radFilters)):
            temp = radFilters[k]
            temp[0,:] = 0
            radFilters[k] = temp

        hipass[0,:] = 1
        lopass[0,:] = 0

    if np.mod(dims[1], 2) == 0:
        for k in range(0, len(radFilters)):
            temp = radFilters[k]
            temp[:,0] = 0
            radFilters[k] = temp

        hipass[:,0] = 1
        lopass[:,0] = 0

    anglemask = []
    for k in range(0,orientations):
        anglemask.append(getAngleMaskSmooth(k, orientations, angle, complexFilt))

    out = []
    out.append(hipass)
    for k in range(0, len(radFilters)):
        for j in range(0, len(anglemask)):
            out.append(np.multiply(anglemask[j], radFilters[k]))
    out.append(lopass)

    return out

def getIDXFromFilter( filter ):
    aboveZero = filter > 1e-10
    dim1 = np.sum(aboveZero,1) > 0
    dim1 = np.logical_or(dim1, np.flip(dim1))
    dim2 = np.sum(aboveZero, 0) > 0
    dim2 = np.logical_or(dim2, np.flip(dim2))
    dims = filter.shape
    idx1 = np.arange(0,dims[0])
    idx2 = np.arange(0,dims[1])

    idx1 = idx1[dim1]
    try:
        idx1 = slice(np.min(idx1), np.max(idx1)+1)
    except ValueError:
        idx1 = np.array([])

    idx2 = idx2[dim2]
    try:
        idx2 = slice(np.min(idx2), np.max(idx2)+1)
    except ValueError:
        idx1 = np.array([])

    return idx1, idx2

def getFilterIDX( filters ):
    nFilts = len(filters)
    filtIDX = np.empty((nFilts,2),dtype=object)
    croppedFilters = np.empty((nFilts,1),dtype=object)
    
    for k in range(0,nFilts):
        indices = getIDXFromFilter(filters[k])
        filtIDX[k,0] = indices[0]
        filtIDX[k,1] = indices[1]
        try:
            croppedFilters[k][0] = filters[k][indices[0], indices[1]]
        except IndexError:
            croppedFilters[k][0] = np.array([])

    return croppedFilters, filtIDX

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def AmplitudeWeightedBlur(input, weight, sigma):
    eps = 2.2204e-16
    if sigma != 0:
        kernel = matlab_style_gauss2D((np.ceil(4*sigma),np.ceil(4*sigma)), sigma)
        weight = weight+eps
        out = cv2.filter2D(np.multiply(input, weight),-1,kernel)
        weightMat = cv2.filter2D(weight,-1,kernel)
        out = np.divide(out, weightMat)
    else:
        out = input
    return out

def FIRWindowBP(delta, fl, fh):
    returndelta = delta.copy()
    timeDimension = 2
    length = delta.shape[2]
    fl = fl*2 #Scale to be fraction of Nyquist frequency
    fh = fh*2
    B = firwin(length+1,[fl, fh],pass_zero=False,scale=True)
    M = delta.shape[0]
    batches = 20  
    batchSize = int(np.ceil(M/batches))
    B = B[0:length]
    temp = np.fft.fft(np.fft.ifftshift(B))

    transferFunction = np.zeros((1,1,temp.shape[0])).astype(complex)
    transferFunction[0,0,:] = temp
    for k in range(1, batches+1):   
        idx = slice(int(batchSize*(k-1)), int(np.minimum(k*batchSize, M)))
        freqDom = np.fft.fft(returndelta[idx,:,:], axis=timeDimension)
        freqDom = np.multiply(freqDom, np.tile(transferFunction, [freqDom.shape[0], freqDom.shape[1]]).reshape(freqDom.shape))
        returndelta[idx,:,:] = np.fft.ifft(freqDom, axis=timeDimension).real
      
    return returndelta

# function delta = differenceOfIIR(delta, rl, rh)
#     timeDimension = 3;
#     len = size(delta, timeDimension);
#     lowpass1 = delta(:,:,1);
#     lowpass2 = lowpass1;    
#     delta(:,:,1) = 0;
#     for i = 2:len       
#         lowpass1 = (1-rh)*lowpass1 + rh*delta(:,:,i);
#         lowpass2 = (1-rl)*lowpass2 + rl*delta(:,:,i);
#         delta(:,:,i) = lowpass1-lowpass2;   
#     end
# end


# function delta = differenceOfButterworths( delta, fl, fh )   
#     timeDimension = 3;
    
#     [low_a, low_b] = butter(1, fl, 'low');
#     [high_a, high_b] = butter(1, fh, 'low');
    
#     len = size(delta,timeDimension);
    
#     lowpass1 = delta(:,:,1);
#     lowpass2 = lowpass1;
#     prev = lowpass1;    
#     delta(:,:,1) = 0;   
#     for i = 2:len
#         lowpass1 = (-high_b(2).*lowpass1 + high_a(1).*delta(:,:,i)+high_a(2).*prev)./high_b(1);
#         lowpass2 = (-low_b(2).*lowpass2 + low_a(1).*delta(:,:,i)+low_a(2).*prev)./low_b(1);
#         prev = delta(:,:,i);
#         delta(:,:,i) = lowpass1-lowpass2;        
#     end

# end

