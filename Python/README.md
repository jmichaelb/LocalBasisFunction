Tested in Python 3.7.0

Packages to install
- `pip3 install --upgrade pip`
- `pip3 install scipy` - automatically installs numpy
- `pip3 install jupyter`  See also https://jupyter.readthedocs.io/en/latest/install.html
- `pip3 install matplotlib`
- `pip3 install psutil`

You may need to repeat these steps whenever upgrading Python.

To get Python to recognize this package, you need to tell Python to look in this directory for packages.
This can be accomplished in myriad ways.  For a more comprehensive explanation,
see https://docs.python.org/3/tutorial/modules.html#the-module-search-path,
https://docs.python.org/3/library/sys.html#sys.path, and https://docs.python.org/3/library/site.html#module-site.
The simplest way is probably to do this is to navigate to the parent directory of mlbspline and
execute the following command in Terminal (assuming you're running on a Mac): ``export PYTHONPATH=`pwd`:$PYTHONPATH``
The main problem with this approach is that you must do it every time you open Terminal.  It is better to install
