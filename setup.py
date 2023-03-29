import sys
from setuptools import setup
from setuptools import find_packages
from setuptools.extension import Extension
import shutil
import os
from Cython.Build import cythonize

with open("README.md", "r") as f:
    long_description = f.read()

extensions = [
    Extension("deeprai.engine.cython.activation", ["deeprai/engine/cython/activation.pyx"]),
    Extension("deeprai.engine.cython.dense_operations", ["deeprai/engine/cython/dense_operations.pyx"]),
    Extension("deeprai.engine.cython.dense_train_loop", ["deeprai/engine/cython/dense_train_loop.pyx"]),
    Extension("deeprai.engine.cython.loss", ["deeprai/engine/cython/loss.pyx"]),
    Extension("deeprai.engine.cython.optimizers", ["deeprai/engine/cython/optimizers.pyx"]),
    Extension("deeprai.engine.cython.regression", ["deeprai/engine/cython/regression.pyx"])
]

compiler_directives = {"language_level": 3, "embedsignature": True}
extensions = cythonize(extensions, compiler_directives=compiler_directives)

setup(
    name='DeeprAI',
    version='0.0.10',
    author='Kieran Carter',
    description='A easy to use and beginner friendly neural network tool box that anyone can pick up and explorer machine learning! ',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Deepr-ai/Deepr-ai",
    packages=find_packages(),
    ext_modules=extensions,
    install_requires=[
        'cython',
        'numpy',
        'pyndb',
        'alive_progress',
        'colorama',
        'cryptography'
    ],
    classifiers=["Programming Language :: Python :: 3",
                 "Programming Language :: Cython",
                 "License :: OSI Approved :: Apache Software License",
                 "Operating System :: OS Independent", ],
    package_data={'deeprai.engine.cython': ['*.pyx']},
    exclude_package_data={'deeprai.engine.cython': ['*.c']},
)

for file in os.listdir('.'):
    if file.endswith('.so') or file.endswith('.pyd'):
        src = os.path.join(os.getcwd(), file)
        dst = os.path.join(os.getcwd(), 'deeprai/engine/cython', file)
        os.rename(src, dst)

# from distutils.core import setup
# from Cython.Build import cythonize
#
# setup(ext_modules=cythonize('deeprai/engine/cython/activation.pyx'))
# setup(ext_modules=cythonize('deeprai/engine/cython/loss.pyx'))
# setup(ext_modules=cythonize('deeprai/engine/cython/optimizers.pyx'))
# setup(ext_modules=cythonize('deeprai/engine/cython/dense_train_loop.pyx'))
# setup(ext_modules=cythonize('deeprai/engine/cython/dense_operations.pyx'))
# setup(ext_modules=cythonize('deeprai/engine/cython/regression.pyx'))

