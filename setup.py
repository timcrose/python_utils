#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

setup(name = 'python_utils',
      description = 'Package for Python tools.',
      author = 'T. Rose',
      url = 'https://github.com/timcrose/python_utils',
      packages = find_packages()
      #install_requires = ['astropy', 'numpy']
     )