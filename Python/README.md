These packages have been tested with Python 3.7 on OSX and with Python 3.6.8 on Windows. 
Although they are intended to be platform-independent, they have not been tested on other operating systems.


The following packages are required:
- scipy (also installs numpy)
- hdf5storage
- psutil

Demo scripts may also require the following packages:
- matplotlib
- [mayavi](https://docs.enthought.com/mayavi/mayavi/installation.html?highlight=install) 
- [jupyter](https://jupyter.readthedocs.io/en/latest/install.html)
- [tkinter](https://docs.python.org/3/library/tkinter.html?highlight=tkinter#module-tkinter)


Installation requires that [git](https://www.git-scm.com/downloads) be installed on the system.  
Also make sure you have the [most recent version of pip](https://pip.pypa.io/en/stable/installing/#upgrading-pip) 
installed.  Run the following command in Terminal on OSX or in Git Bash on Windows.  

``pip install git+https://github.com/jmichaelb/LocalBasisFunction.git@v0.7#egg=uw-highP-geophysics-tools\&subdirectory=Python``

If you need to uninstall, use 
``pip uninstall uw-highP-geophysics-tools``

Note that scripts can be run in the regular command prompt on Windows; it is only the install that needs to be done 
in Git Bash.




