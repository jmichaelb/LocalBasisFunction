These packages have been tested on OSX in Python 3.7.  
Although they are intended to be platform-independent, they have not been tested on other operating systems.


The following packages are required:
- scipy (also installs numpy)
- hdf5storage
- psutil

Demo scripts may also require the following packages:
- matplotlib
- [jupyter](https://jupyter.readthedocs.io/en/latest/install.html)
- [tkinter](https://docs.python.org/3/library/tkinter.html?highlight=tkinter#module-tkinter)


To install, first make sure you have the [most recent version of pip](https://pip.pypa.io/en/stable/installing/#upgrading-pip) 
installed.  Use the following command.  Note the escaped ``&`` before ``subdirectory`` - this may not be necessary on 
Windows, but on that OS, the URL may instead need to be enclosed by double quotes).

``pip install git+https://github.com/jmichaelb/LocalBasisFunction.git@master#egg=uw-highP-geophysics-tools\&subdirectory=Python``



