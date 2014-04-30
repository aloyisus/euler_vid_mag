euler_vid_mag
=============

A python implementation of one of MIT's Eulerian Video Magnification algorithms from the Siggraph 2012 paper

The paper can be found here http://people.csail.mit.edu/mrub/vidmag/ and is a great read.

This python code is basically a translation of the matlab code provided by the authors, specifically the
"amplify_spatial_Gdown_temporal_ideal.m" module. This is the temporal filter.

I am using opencv for video capture/output, and also numpy.

To use, import the module "amplify_spatial_Gdown_temporal_ideal", then call the function
"amplify_spatial_Gdown_temporal_ideal" with the following parameters:

vidFile - video file to be processed
outDir - output directory
alpha - amplification factor
level - 
fl - low frequency cutoff
fh - hi frequency cutoff
chromAttenuation - 

e.g. asGti.amplify_spatial_Gdown_temporal_ideal('sample/face_source.mov','sample/',50,4,50/60,60/60,30)
