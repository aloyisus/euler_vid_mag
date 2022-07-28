euler_vid_mag
=============

A python implementation MIT's Eulerian Video Magnification algorithms from both the Siggraph 2012 paper (Hao-Yu Wu, Michael Rubinstein, Eugene Shih, John Guttag, Frédo Durand and William T. Freeman) and also the "new-and-improved" phase-based approach from the 2013 paper (Neal Wadhwa, Michael Rubinstein, Frédo Durand and William T. Freeman). The code presented here is a python implementation of the matlab sources provided by MIT, which are used under license with all credit going to the authors of these algorithms; additionally, the linear implementation leans heavily on https://github.com/LabForComputationalVision/pyrtools for generating Laplacian pyramids.

The original papers can be found here (along with more recent developments) http://people.csail.mit.edu/mrub/vidmag/ and they make for an excellent read.

Installation:
Installing into a virtualenv is recommended. At the command prompt, enter:
> pip install -e .

To reproduce the results for the Siggraph 2013 paper:
1) Download the [source videos][videos] (1.3GB) from the
  [project web page][phase] into a directory called `sample`.
2) Type `reproduce_results_siggraph13` to reproduce the results in the paper. 

[videos]: http://people.csail.mit.edu/nwadhwa/phase-video/video/Source%20and%20Result%20Videos.zip
[phase]: http://people.csail.mit.edu/nwadhwa/phase-video/

There is also a little tkinter UI for demo purposes, which can be launched from the shell like this:
> phase_amplify_app

The gif below is the MIT file 'car_engine.avi' after processing with the phase-based algorithm:

![screenshot](https://raw.githubusercontent.com/aloyisus/euler_vid_mag/master/car_engine_pb.gif)

Note that OpenCV 4.x is required (should be compatible with 3.x versions too).
