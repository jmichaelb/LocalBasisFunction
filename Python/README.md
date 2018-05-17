Tested in Python 3.6.5

Packages to install
- `pip3 install --upgrade pip`
- `pip install scipy` - automatically installs numpy
- `pip3 install jupyter`  See also https://jupyter.readthedocs.io/en/latest/install.html
- `pip3 install matplotlib`

To get Python to recognize this package, you need to tell Python to look in this directory for packages.
This can be accomplished in myriad ways.  For a more comprehensive explanation,
see https://docs.python.org/3/tutorial/modules.html#the-module-search-path,
https://docs.python.org/3/library/sys.html#sys.path, and https://docs.python.org/3/library/site.html#module-site.
The simplest way is probably to do this is to navigate to the parent directory of mlbspline and
execute the following command in Terminal (assuming you're running on a Mac): ``export PYTHONNPATH=`pwd`:$PYTHONPATH``
