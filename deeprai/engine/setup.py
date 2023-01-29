from distutils.core import setup
from Cython.Build import cythonize

# setup(ext_modules=cythonize('conv2d.pyx'))
setup(ext_modules=cythonize('loss.pyx'))