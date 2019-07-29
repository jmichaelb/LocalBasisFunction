import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='uw-highP-geophysics-tools',
    version='0.8.1',
    author='pennythewho',
    author_email='pennythewho@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    description='load and evaluate splines developed in MatLab -- for Gibbs energy splines, also calculates several ' +
                'other thermodynamic variables for pure substances or single-solute solutions',
    url='https://github.com/jmichaelb/LocalBasisFunction',
    packages=['mlbspline', 'lbftd'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent"
    ],
    install_requires=['scipy', 'hdf5storage', 'psutil'],
    data_files=[('', ['LICENSE.txt'])]
)