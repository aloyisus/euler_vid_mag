from setuptools import setup

setup(
    name='euler_vid_mag',
    version='0.1.0',
    package_dir = {'':'src'},
    packages=['euler_vid_mag'],
    scripts=[
        'scripts/phase_amplify_app',
        'scripts/reproduce_results_siggraph13'
     ],
    description='Python implementation of MIT Video Magnification algorithms',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.23",
        "pyrtools >= 1",
        "scipy >= 1.8",
        "opencv-python >= 3",
    ],
)
