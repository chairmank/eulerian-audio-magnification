eulerian-audio-magnification
============================

Like [Eulerian video
magnification](http://people.csail.mit.edu/mrub/vidmag/) but for audio.

Requirements
------------
You need virtualenv and pip.

You also need a bunch of dependencies to install NumPy and SciPy and
matplotlib.

SciPy requires you to export two environment variables:

export BLAS=/path/to/your/libfblas.a
export LAPACK=/path/to/your/libflapack.a

Follow instructions at
http://www.scipy.org/Installing_SciPy/BuildingGeneral

Getting started
---------------
Run

$ ./setup.sh

to create a Python virtualenv and install NumPy and AudioLab.

To activate the virtualenv, run

$ source python-virtualenv/bin/activate

License
-------

Distributed under the MIT license.
