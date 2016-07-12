#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import sys

classifiers = [
    'Development Status :: 4 - Beta',

    # Indicate who your project is intended for
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',

    # Pick your license as you wish (should match "license" above)
     'License :: OSI Approved :: GNU General Public License (GPL)',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 2.7',
]

keywords = 'rolling-shutter camera imu gyroscope simulation'

requires = [ 'numpy',
             'scipy',
             'matplotlib',
             'crisp',
             'imusim'
]

scripts = [os.path.join('scripts/', fname) for fname in [
    'rsimurun.py',
]]


setup(name='rsimusim',
      version='0.1',
      author="Hannes Ovrén",
      author_email="hannes.ovren@liu.se",
      description="Rolling-shutter camera and IMU simulation toolbox",
      license="GPL",
      packages=['rsimusim', 'rsimusim.inertial'],
      scripts=scripts,
      classifiers=classifiers,
      install_requires=requires,
      requires=requires,
      keywords=keywords,
    )
