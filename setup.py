import sys
from setuptools import setup
from setuptools import find_packages
from setuptools.extension import Extension
import shutil
import os
try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

with open("README.md", "r") as f:
    long_description = f.read()

def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


extensions = [
    Extension("deeprai/engine/cython/activation", ["deeprai/engine/cython/activation.c"]),
    Extension("deeprai/engine/cython/dense_operations", ["deeprai/engine/cython/dense_operations.c"]),
    Extension("deeprai/engine/cython/dense_train_loop", ["deeprai/engine/cython/dense_train_loop.c"]),
    Extension("deeprai/engine/cython/loss",["deeprai/engine/cython/loss.c"]),
    Extension("deeprai/engine/cython/optimizers", ["deeprai/engine/cython/optimizers.c"]),
    ]
CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 0))) and cythonize is not None

if CYTHONIZE:
    compiler_directives = {"language_level": 3, "embedsignature": True}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)

setup(
    name='deeprai',
    version='0.0.7',
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
             "Operating System :: OS Independent",]

)
for file in os.listdir('.'):
    if file.endswith('.so') or file.endswith('.pyd'):
        src = os.path.join(os.getcwd(), file)
        dst = os.path.join(os.getcwd(), 'deeprai/engine/cython', file)
        os.rename(src, dst)