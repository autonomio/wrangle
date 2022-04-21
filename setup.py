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
VERSION = '0.7.2'

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

install_requires = ['numpy',
                    'pandas',
                    'statsmodels>=0.11.0',
                    'scipy',
                    'sklearn',
                    'tensorflow']

if __name__ == "__main__":

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
          packages=['wrangle',
                    'wrangle.utils',
                    'wrangle.dic',
                    'wrangle.df',
                    'wrangle.col',
                    'wrangle.array'],

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
