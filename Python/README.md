These packages have been tested on OSX in Python 3.7.  
Although they are intended to be platform-independent, they have not been tested on other operating systems.


The following packages are required:
- scipy (also installs numpy)
- hdf5storage
- psutil

Demo scripts may also require the following packages:
- matplotlib
- jupyter (see https://jupyter.readthedocs.io/en/latest/install.html)


To install, use the following command (note the escaped ``&`` before ``subdirectory`` - this may not be necessary on Windows, but on that OS, the URL may 
need to be enclosed by double quotes)

``pip3 install git+https://github.com/jmichaelb/LocalBasisFunction.git@master/#egg=uw-highP-geophysics-tools\&subdirectory=Python``



