#! /usr/bin/env python
#
# Copyright (C) 2018 Mikko Kotila

DESCRIPTION = "Wrangle - Data Preparation for Deep Learning"
LONG_DESCRIPTION = """\
Wrangle takes the headache away from most common data preparation
and wrangling tasks.
"""

DISTNAME = 'wrangle'
MAINTAINER = 'Mikko Kotila'
MAINTAINER_EMAIL = 'mailme@mikkokotila.com'
URL = 'http://autonom.io'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/autonomio/wrangle/'
VERSION = '0.3.1'

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

def check_dependencies():

    install_requires = []

    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import pandas
    except ImportError:
        install_requires.append('pandas')


    return install_requires

if __name__ == "__main__":

    install_requires = check_dependencies()

    setup(name=DISTNAME,
          author=MAINTAINER,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=install_requires,
          packages=['wrangle'],

          classifiers=[
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: 3.6',
                     'License :: OSI Approved :: MIT License',
                     'Topic :: Scientific/Engineering :: Human Machine Interfaces',
                     'Topic :: Scientific/Engineering :: Artificial Intelligence',
                     'Topic :: Scientific/Engineering :: Mathematics',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
)
