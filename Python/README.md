These packages have been tested on OSX in Python 3.7.  
Although they are intended to be platform-independent, they have not been tested on other operating systems.


The following packages are required:
- scipy (also installs numpy)
- hdf5storage
- psutil

Demo scripts may also require the following packages:
- matplotlib
- jupyter (see https://jupyter.readthedocs.io/en/latest/install.html)




To get Python to recognize this package, you need to tell Python to look in this directory for packages.
This can be accomplished in myriad ways.  For a more comprehensive explanation,
see https://docs.python.org/3/tutorial/modules.html#the-module-search-path,
https://docs.python.org/3/library/sys.html#sys.path, and https://docs.python.org/3/library/site.html#module-site.
The simplest way is probably to do this is to navigate to the parent directory of mlbspline and
execute the following command in Terminal (assuming you're running on a Mac): ``export PYTHONPATH=`pwd`:$PYTHONPATH``
The main problem with this approach is that you must do it every time you open Terminal.  It is better to install
