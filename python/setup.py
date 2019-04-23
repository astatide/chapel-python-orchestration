from distutils.core import setup, Extension
import numpy as np
from numpy.distutils.core import setup, Extension

module = Extension('gjallarbru', sources = ['gjallarbru.c'])
setup(name = 'gjallarbru', version = '1.0', ext_modules = [module])
