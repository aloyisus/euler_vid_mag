euler_vid_mag
=============

A python implementation of one of MIT's Eulerian Video Magnification algorithms from the Siggraph 2012 paper by Hao-Yu Wu, Michael Rubinstein, Eugene Shih, John Guttag, Frédo Durand and William T. Freeman. Their work is used under license. The technique described reveals otherwise hidden information in an ordinary video stream by magnifying small colour changes in an ordinary video stream. This can be used, for example, to reveal the blood flow in a person's face.

The paper can be found here http://people.csail.mit.edu/mrub/vidmag/ and is a great read.

The code presented here is basically a python translation of the matlab sources provided by MIT, and as such all credit goes to the authors of these algorithms, I've just put them into a form which doesn't require a matlab license. Additionally this implementation leans heavily on https://github.com/LabForComputationalVision/pyrtools for generating Laplacian pyramids.

There are two demo UIs, colour_amplify_demo and motion_magnify_demo. They can be launched from the shell using e.g.
python motion_magnify_demo.py

Various parameters are exposed, but particularly important are the low and high frequency cutoffs - which allows the algorithm to focus on a particular range of frequencies - and the sampling rate, which much match the framerate of the original video capture.

The screenshots below are the before/afters of processing the "face.mp4" and "guitar.avi" files (supplied by MIT):
http://people.csail.mit.edu/mrub/evm/video/face.mp4
http://people.csail.mit.edu/mrub/evm/video/guitar.mp4
using the colour and motion demos respectively with their default settings.

Unprocessed:

![screenshot](https://raw.githubusercontent.com/aloyisus/euler_vid_mag/master/guitar.gif)

![screenshot](https://raw.githubusercontent.com/aloyisus/euler_vid_mag/master/face.gif)

Processed:

![screenshot](https://raw.githubusercontent.com/aloyisus/euler_vid_mag/master/guitar_processed.gif)

![screenshot](https://raw.githubusercontent.com/aloyisus/euler_vid_mag/master/face_processed.gif)

Note that OpenCV 4.x is required (should be compatible with 3.x versions too) along with the associated python bindings. tkinter support is required for the UI. Necessary python3 modules are listed in requirements.txt file, pip install -r requirements.txt to install them.
