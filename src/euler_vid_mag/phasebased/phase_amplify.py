import cv2
import numpy as np

from pathlib import Path
import sys

from .filters import *


def get_phase_amplified_frames(
    filename,
    magphase,
    fl,
    fh,
    fs,
    progress_callback = lambda x: x,
    attenuateOtherFreq = False,
    pyrtype = 'octave',
    sigma = 0,
    temporalFilter = FIRWindowBP,
    scalevideo = 1,
    frames = None,
):
    """Yields phase-amplified frames from the input video.
    
    Takes input video file and motion-magnifies the frequencies that are
    within a given passband by the specified amount. Yields the resulting
    frames for consumption.

    Required arguments:
    filename: Name of input video file.
    magphase: The amount of magnification.
    fl: Lower frequency cutoff in Hz.
    fh: Upper frequency cutoff in Hz.
    fs: Sampling rate in Hz.

    Optional arguments:
    progress_callback: callback function to track progress from 0-100%.
    attenuateOtherFrequencies: Whether to attenuate frequencies in the stopband.  
    pyrtype: Spatial representation to use (see paper).
    sigma: Amount of spatial smoothing (in pixel) to apply to phases.
    temporalFilter: What temporal filter to use.
    scalevideo: Operate on scaled video size to reduce memory usage.
    frames: Specify start and end frames of video.

    Returns:
    Yields the amplified frames.
    """
    ## Read Video
    print("Preparing capture")
    vr = cv2.VideoCapture(filename)
    nF = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = vr.read()
    h, w, nC = frame.shape

    print("Loading: ", filename)
    vid = np.zeros([h, w, nC, nF],frame.dtype)
    vid[:,:,:,0] = frame
    for k in range(1, nF):
        retval, frame = vr.read()
        vid[:,:,:,k] = frame
        progress_callback(10/nF)
        sys.stdout.write('.')
        sys.stdout.flush()
    print('\n')
    vr.release()

    if not frames:
        frames = (0, nF-1)

    ## Compute spatial filters        
    vid = vid[:,:,:,frames[0]:frames[1]+1]
    h, w, nC, nF = vid.shape
    if scalevideo != 1:
        h, w = int(scalevideo*h), int(scalevideo*w)

    print('Computing spatial filters\n')
    ht = maxSCFpyrHt(np.zeros((h,w)))
    if pyrtype == 'octave':
        filters = getFilters((h, w), np.power(2,np.arange(0,-ht-1,-1).astype(float)), 4)
        print('Using octave bandwidth pyramid\n')    
    elif pyrtype == 'halfOctave':           
        filters = getFilters((h, w), np.power(2,np.arange(0,-ht-0.5,-0.5).astype(float)), 8, twidth=0.75)
        print('Using half octave bandwidth pyramid\n')
    elif pyrtype == 'smoothHalfOctave':
        filters = getFiltersSmoothWindow((h, w), 8, filtersPerOctave = 2)        
        print('Using half octave pyramid with smooth window.\n')
    elif pyrtype == 'quarterOctave':
        filters = getFiltersSmoothWindow((h, w), 8, filtersPerOctave = 4)
        print('Using quarter octave pyramid.\n')
    else:
        raise ValueError('Invalid Filter Types')

    croppedFilters, filtIDX = getFilterIDX(filters)
    
    ## Initialization of motion magnified luma component
    magnifiedLumaFFT = np.zeros((h,w,nF),np.complex64)

    def buildLevel(im_dft,k):
        return np.fft.ifft2(
            np.fft.ifftshift(
                np.multiply(croppedFilters[k][0], im_dft[filtIDX[k,0], filtIDX[k,1]])
            )
        )

    def reconLevel(im_dft, k):
        return 2*(np.multiply(croppedFilters[k][0], np.fft.fftshift(np.fft.fft2(im_dft))))

    ## First compute phase differences from reference frame
    numLevels = len(filters)    
    print('Moving video to Fourier domain\n')
    vidFFT = np.zeros((h,w,nF),dtype=np.complex64)
    for k in range(0, nF):
        originalFrame = vid[:,:,:,k].astype(np.float32)
        originalFrame = cv2.cvtColor(originalFrame, cv2.COLOR_BGR2YUV)
        tVid = cv2.resize(originalFrame[:,:,0], (w, h), 0, 0, cv2.INTER_CUBIC)
        vidFFT[:,:,k] = np.fft.fftshift(np.fft.fft2(tVid))
        sys.stdout.write('.')
        sys.stdout.flush()
        progress_callback(20/nF)
    print('\n')

    refFrame = 0
    for level in range(1, numLevels-1):
        ## Compute phases of level
        # We assume that the video is mostly static
        pyrRef = buildLevel(vidFFT[:,:,refFrame], level)
        pyrRefPhaseOrig = np.divide(pyrRef, np.abs(pyrRef))
        pyrRef = np.angle(pyrRef)

        delta = np.zeros((pyrRef.shape[0], pyrRef.shape[1] ,nF), np.float32)
        print('Processing level {0} of {1}\n'.format(level+1, numLevels))

        for frameIDX in range(0, nF):
            filterResponse = buildLevel(vidFFT[:,:,frameIDX], level)
            pyrCurrent = np.angle(filterResponse)
            delta[:,:,frameIDX] = np.mod(np.pi+pyrCurrent-pyrRef,2*np.pi)-np.pi
  
        ## Temporal Filtering
        print('Bandpassing phases\n')
        delta = temporalFilter(delta, fl/fs, fh/fs) 

        ## Apply magnification
        eps = 2.2204e-16
        print('Applying magnification\n')
        for frameIDX in range(0, nF):

            phaseOfFrame = delta[:,:,frameIDX]
            originalLevel = buildLevel(vidFFT[:,:,frameIDX], level)
            ## Amplitude Weighted Blur        
            if (sigma != 0):
                phaseOfFrame = AmplitudeWeightedBlur(phaseOfFrame, np.abs(originalLevel)+eps, sigma)

            # Increase phase variation
            phaseOfFrame = magphase * phaseOfFrame
            
            if attenuateOtherFreq:
                tempOrig = np.multiply(np.abs(originalLevel), pyrRefPhaseOrig)
            else:
                tempOrig = originalLevel

            tempTransformOut = np.multiply(np.exp(1j*phaseOfFrame), tempOrig)

            curLevelFrame = reconLevel(tempTransformOut, level)
            magnifiedLumaFFT[filtIDX[level, 0], filtIDX[level, 1], frameIDX] = \
                curLevelFrame + magnifiedLumaFFT[filtIDX[level, 0], filtIDX[level, 1], frameIDX]
        
        progress_callback(50/(numLevels-2))

    ## Add unmolested lowpass residual
    level = len(filters)-1
    for frameIDX in range(0, nF):
        try:
            lowpassFrame = np.multiply(vidFFT[filtIDX[level, 0], filtIDX[level, 1] ,frameIDX], np.power(croppedFilters[-1][0], 2))
            magnifiedLumaFFT[filtIDX[level, 0], filtIDX[level, 1], frameIDX] = \
                magnifiedLumaFFT[filtIDX[level, 0], filtIDX[level, 1], frameIDX] + lowpassFrame
        except IndexError:
            print("IndexError")
        progress_callback(10/nF)

    for k in range(0, nF):
        magnifiedLuma = np.fft.ifft2(np.fft.ifftshift(magnifiedLumaFFT[:,:,k])).real
        originalFrame = cv2.cvtColor(vid[:,:,:,k].astype(np.float32), cv2.COLOR_BGR2YUV)
        originalFrame = cv2.resize(originalFrame, (w, h), 0, 0, cv2.INTER_CUBIC)
        originalFrame[:,:,0] = magnifiedLuma
        originalFrame = cv2.cvtColor(originalFrame, cv2.COLOR_YUV2BGR)
        originalFrame = cv2.convertScaleAbs(originalFrame)
        yield originalFrame
        progress_callback(10/nF)


def phase_amplify_to_file(
    filename,
    magphase,
    fl,
    fh,
    fs,
    outdir = './',
    attenuateOtherFreq = False,
    pyrtype = 'octave',
    sigma = 0,
    temporalFilter = FIRWindowBP,
    scalevideo = 1,
    frames = None,
):
    """Writes phase-amplified video file from given input video.
    
    Takes input video file and motion-magnifies the frequencies that are
    within a given passband by the specified amount. Writes the result
    to an output file.

    Required arguments:
    filename: Name of input video file.
    magphase: The amount of magnification.
    fl: Lower frequency cutoff in Hz.
    fh: Upper frequency cutoff in Hz.
    fs: Sampling rate in Hz.

    Optional arguments:
    outdir: Location to write output video file.
    attenuateOtherFrequencies: Whether to attenuate frequencies in the stopband.  
    pyrtype: Spatial representation to use (see paper).
    sigma: Amount of spatial smoothing (in pixel) to apply to phases.
    temporalFilter: What temporal filter to use.
    scalevideo: Operate on scaled video size to reduce memory usage.
    frames: Specify start and end frames of video.

    Returns:
    None.
    """
    vr = cv2.VideoCapture(filename)
    writetag = Path(filename).stem
    nF = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
    framerate = vr.get(cv2.CAP_PROP_FPS)
    ret, frame = vr.read()
    h, w, nC = frame.shape
    if not frames:
        frames = (0, nF-1)
    if scalevideo != 1:
        h, w = int(scalevideo*h), int(scalevideo*w)
    vr.release()

    outname = \
        "{0}-{1}-band{2:0.2f}-{3:0.2f}-sr{4}-alpha{5}-mp{6}-sigma{7}-scale{8:0.2f}-frames{9}-{10}-{11}.mov".format(
        writetag,
        temporalFilter.__name__,
        fl,
        fh,
        fs,
        magphase,
        attenuateOtherFreq,
        sigma,
        scalevideo,
        frames[0],
        frames[1],
        pyrtype,
    )
    dir = Path(outdir)
    outfilename = str(dir / outname)
    vw = cv2.VideoWriter()
    fourcc = vw.fourcc('j','p','e','g')
    success = vw.open(outfilename,fourcc,framerate,(w,h),True)
    for frame in get_phase_amplified_frames(
        filename,
        magphase,
        fl,
        fh,
        fs,
        attenuateOtherFreq=attenuateOtherFreq,
        pyrtype=pyrtype,
        sigma=sigma,
        temporalFilter=temporalFilter,
        scalevideo=scalevideo,
        frames=frames,        
    ):
        vw.write(frame)
    vw.release()
