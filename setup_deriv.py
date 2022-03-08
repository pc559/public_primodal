## # Run this to perform the set-up:
## # python3 setup_deriv.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("evol_deriv.pyx"),
    include_dirs=[numpy.get_include()]
)
