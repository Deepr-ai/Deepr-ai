import sys
from setuptools import setup
from setuptools import find_packages
from setuptools.extension import Extension
import os
import numpy
from Cython.Build import cythonize

with open("README.md", "r") as f:
    long_description = f.read()

extensions = [
    Extension("deeprai.engine.cython.activation", ["deeprai/engine/cython/activation.pyx"], include_dirs=[numpy.get_include()]),
    Extension("deeprai.engine.cython.dense_operations", ["deeprai/engine/cython/dense_operations.pyx"], include_dirs=[numpy.get_include()]),
    Extension("deeprai.engine.cython.dense_train_loop", ["deeprai/engine/cython/dense_train_loop.pyx"], include_dirs=[numpy.get_include()]),
    Extension("deeprai.engine.cython.loss",["deeprai/engine/cython/loss.pyx"], include_dirs=[numpy.get_include()]),
    Extension("deeprai.engine.cython.optimizers", ["deeprai/engine/cython/optimizers.pyx"], include_dirs=[numpy.get_include()]),
    Extension("deeprai.engine.cython.regression", ["deeprai/engine/cython/regression.pyx"], include_dirs=[numpy.get_include()]),
    Extension("deeprai.engine.cython.positional_embedding", ["deeprai/engine/cython/positional_embedding.pyx"], include_dirs=[numpy.get_include()]),
    Extension("deeprai.engine.cython.knn", ["deeprai/engine/cython/knn.pyx"],include_dirs=[numpy.get_include()]),
    Extension("deeprai.engine.cython.knn_distance", ["deeprai/engine/cython/knn_distance.pyx"], include_dirs=[numpy.get_include()]),

]

compiler_directives = {"language_level": 3, "embedsignature": True}

setup(
    name='DeeprAI',
    version='0.0.15',
    author='Kieran Carter',
    description='A easy to use and beginner friendly neural network tool box that anyone can pick up and explorer machine learning! ',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Deepr-ai/Deepr-ai",
    packages=find_packages(),
    install_requires=[
        'cython',
        'numpy',
        'pyndb',
        'alive_progress',
        'colorama',
        'cryptography',
        'matplotlib'
    ],
    classifiers=["Programming Language :: Python :: 3",
                 "Programming Language :: Cython",
                 "License :: OSI Approved :: Apache Software License",
                 "Operating System :: OS Independent", ],
    package_data={'deeprai.engine.cython': ['*.pyx']},
    exclude_package_data={'deeprai.engine.cython': ['*.c']},
    ext_modules=cythonize(extensions, compiler_directives=compiler_directives),
)


# For compiling from scratch
# from distutils.core import setup
# from Cython.Build import cythonize
# import numpy

# setup(ext_modules=cythonize('deeprai/engine/cython/activation.pyx'), include_dirs=[numpy.get_include()])
# setup(ext_modules=cythonize('deeprai/engine/cython/loss.pyx'), include_dirs=[numpy.get_include()])
# setup(ext_modules=cythonize('deeprai/engine/cython/optimizers.pyx'), include_dirs=[numpy.get_include()])
# setup(ext_modules=cythonize('deeprai/engine/cython/dense_train_loop.pyx'), include_dirs=[numpy.get_include()])
# setup(ext_modules=cythonize('deeprai/engine/cython/dense_operations.pyx'), include_dirs=[numpy.get_include()])
# setup(ext_modules=cythonize('deeprai/engine/cython/regression.pyx'), include_dirs=[numpy.get_include()])
# setup(ext_modules=cythonize('deeprai/engine/cython/positional_embedding.pyx'), include_dirs=[numpy.get_include()])
# setup(ext_modules=cythonize('deeprai/engine/cython/knn.pyx'), include_dirs=[numpy.get_include()])
# setup(ext_modules=cythonize('deeprai/engine/cython/knn_distance.pyx'), include_dirs=[numpy.get_include()])