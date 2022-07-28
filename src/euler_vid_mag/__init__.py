"""
This package exports two public functions which perform
phase-based motion amplification on the supplied video file.

Example:
========

To amplify motion in the file car_engine.avi in the range 15-25Hz:
>>> from euler_vid_mag import phase_amplify_to_file
>>> phase_amplify_to_file('car_engine.avi',15,15,25,400,sigma=3)

"""

from .phasebased.phase_amplify import get_phase_amplified_frames, phase_amplify_to_file
