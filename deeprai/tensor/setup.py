from setuptools import setup, Extension

tensor = Extension('tensor',
                         sources=['tensor.c'],
                         include_dirs=['libs'])


setup(name='TensorPackage',
      version='1.0',
      description='Tensor package',
      ext_modules=[tensor])
