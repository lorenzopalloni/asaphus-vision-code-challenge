"""
Compile with the following command:
python setup.py build_ext --inplace
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("cython_my_utils.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
)
