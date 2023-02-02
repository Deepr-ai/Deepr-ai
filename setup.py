import sys
from setuptools import setup
from Cython.Build import cythonize
from setuptools import find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from os import listdir, remove


with open("README.md", "r") as f:
    long_description = f.read()

extensions = cythonize("deeprai/engine/cython/*.pyx")
class build_ext(_build_ext):
    def build_extension(self, ext):
        ext.extra_compile_args.append('-g')
        if sys.platform == 'win32':
            print("s")
            ext.extra_compile_args.append('/Zi')
            ext.filename = ext.name + '.pyd'
        else:
            ext.filename = ext.name + '.so'
        _build_ext.build_extension(self, ext)

setup(
    name='deeprai',
    version='0.0.2',
    author='Kieran Carter',
    description='A easy to use and beginner friendly neural network tool box that anyone can pick up and explorer machine learning! ',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Deepr-ai/Deepr-ai",
    packages=find_packages(),
    ext_modules=extensions,
    package_data={'deeprai.engine.cython': ['deeprai/engine/cython/*.pyd', 'deeprai/engine/cython/*.so']},
    options={'build_ext': {'build_lib': 'deeprai/engine/cython/'}},
    cmdclass={'build_ext': build_ext},
    install_requires=[
        'cython',
        'numpy',
        'pyndb',
        'alive-progress',
        'colorama'
    ],
classifiers=["Programming Language :: Python :: 3",
             "Programming Language :: Cython",
             "License :: OSI Approved :: Apache Software License",
             "Operating System :: OS Independent",]

)
for file in listdir('.'):
    if file.endswith('.so') or file.endswith('.pyd'):
        remove(file)
