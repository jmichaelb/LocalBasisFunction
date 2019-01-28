import setuptools

setuptools.setup(
    name='uw-highP-geophysics-tools',
    version='0.1a1',
    author='pennythewho',
    author_email='pennythewho@gmail.com',
    description='load and evaluate splines developed in MatLab -- for Gibbs energy splines, also calculates several ' +
                'other thermodynamic variables for pure substances or single-solute solutions',
    url='https://github.com/jmichaelb/LocalBasisFunction/tree/master/Python',
    packages=['mlbspline', 'lbftd'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent"
    ]
)