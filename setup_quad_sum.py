## # Run this to perform the set-up:
## # python3 setup_quad_sum.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize

## How to use -O3?
setup(
    ext_modules=cythonize("quad_sum.pyx"),# extra_compile_args=["-O3"]),
)
