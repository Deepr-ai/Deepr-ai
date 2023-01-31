from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('dense_train_loop.pyx'))
setup(ext_modules=cythonize('optimizers.pyx'))
setup(ext_modules=cythonize('dense_operations.pyx'))