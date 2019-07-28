These packages have been tested with Python 3.7 on OSX and with Python 3.6.8 on Windows. 
Although they are intended to be platform-independent, they have not been tested on other operating systems.

Installation requires [git](https://www.git-scm.com/book/en/v2/Getting-Started-Installing-Git).  
Also make sure you have the [most recent version of pip](https://pip.pypa.io/en/stable/installing/#upgrading-pip) 
installed.  

The following packages will be installed:
- [scipy](https://scipy.org/scipylib/index.html) (also installs [numpy](http://numpy.org/))
- [hdf5storage](https://pythonhosted.org/hdf5storage/)
- [psutil](https://pypi.org/project/psutil/)

Demo scripts may also require the following packages, which must be [installed](https://packaging.python.org/tutorials/installing-packages/) manually:
- [matplotlib](https://matplotlib.org/users/installing.html)
- [jupyter](https://jupyter.readthedocs.io/en/latest/install.html)

Run the following command in Terminal on OSX or in Git Bash on Windows.  

``pip install git+https://github.com/jmichaelb/LocalBasisFunction.git@master#egg=uw-highP-geophysics-tools\&subdirectory=Python``

If you need to uninstall, use 
``pip uninstall uw-highP-geophysics-tools``

Note that scripts can be run in the regular command prompt on Windows; it is only the install that needs to be done 
in Git Bash.




