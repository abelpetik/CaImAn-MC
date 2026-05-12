#!/usr/bin/env python

from setuptools import setup

"""
    Installation script for anaconda installers
"""

with open('README.md', 'r') as rmf:
    readme = rmf.read()

with open('VERSION', 'r') as verfile:
    version = verfile.read().strip()

setup(
    name='caiman',
    version=version,
    author='Andrea Giovannucci, Eftychios Pnevmatikakis, Johannes Friedrich, Valentina Staneva, Ben Deverett, Erick Cobos, Jeremie Kalfon',
    author_email='pgunn@flatironinstitute.org',
    url='https://github.com/flatironinstitute/CaImAn',
    license='GPL-2',
    description='Advanced algorithms for ROI detection and deconvolution of Calcium Imaging datasets.',
    long_description=readme,
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Researchers',
        'Topic :: Calcium Imaging :: Analysis Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GPL-2 License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
    keywords='fluorescence calcium ca imaging deconvolution ROI identification',
    packages=['caiman', 'caiman.base', 'caiman.utils'],
    install_requires=[
        'h5py>=3.4.0',
        'ipyparallel',
        'matplotlib',
        'numpy>=2.0.0',
        'opencv-python',
        'psutil',
        'python-dateutil',
        'scikit-image>=0.19.0',
        'scipy>=1.10.1',
        'tifffile',
        'tqdm',
    ],
    ext_modules=[]
)
