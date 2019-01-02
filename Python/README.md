Tested in Python 3.7.0
But targeted to 3.6 (Anaconda)

Packages to install

If using Anaconda, replace `pip` with `conda`
- `pip install --upgrade pip`
- `pip install scipy` - automatically installs numpy
- `pip install jupyter` - See also https://jupyter.readthedocs.io/en/latest/install.html
- `pip install matplotlib`
- `pip install psutil`
- `pip install hdf5storage`

You may need to repeat these steps whenever upgrading Python.

To get Python to recognize this package, you need to tell Python to look in this directory for packages.
This can be accomplished in myriad ways.  For a more comprehensive explanation,
see https://docs.python.org/3/tutorial/modules.html#the-module-search-path,
https://docs.python.org/3/library/sys.html#sys.path, and https://docs.python.org/3/library/site.html#module-site.
The simplest way is probably to do this is to navigate to the parent directory of mlbspline and
execute the following command in Terminal (assuming you're running on a Mac): ``export PYTHONPATH=`pwd`:$PYTHONPATH``
The main problem with this approach is that you must do it every time you open Terminal.  It is better to install
