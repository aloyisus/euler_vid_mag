euler_vid_mag
=============

A python implementation of one of MIT's Eulerian Video Magnification algorithms from the Siggraph 2012 paper. The technique described reveals otherwise hidden information in an ordinary video stream by magnifying small colour changes in an ordinary video stream. This can be used, for example, to reveal the blood flow in a person's face.

The paper can be found here http://people.csail.mit.edu/mrub/vidmag/ and is a great read.

This python code is basically a translation of the matlab code provided by the authors, specifically the
"amplify_spatial_Gdown_temporal_ideal.m" module.

I am using opencv for video capture/output, and also numpy for it's fast fourier transform functions.

To use, import the module "amplify_spatial_Gdown_temporal_ideal", then call the function
"amplify_spatial_Gdown_temporal_ideal" with the following parameters:

vidFile - video file to be processed
outDir - output directory
alpha - amplification factor
level - number of levels in the stack
fl - low frequency cutoff
fh - hi frequency cutoff
sample - frequency of sampling eg 30hz for ntsc footage
chromAttenuation - can apply bias to uv components
colourspace - work in 'rgb' or 'yuv'

e.g. amplify_spatial_Gdown_temporal_ideal('sample/face_source.mov','sample/',50,4,50/60.0,60/60.0,30,1,'yuv')

Below are the results obtained by processing the supplied "face.mov" file provided by the authors:

Unprocessed:

![screenshot](https://raw.githubusercontent.com/aloyisus/euler_vid_mag/master/unprocessed.gif)

Processed:

![screenshot](https://raw.githubusercontent.com/aloyisus/euler_vid_mag/master/processed.gif)
