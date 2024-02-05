from setuptools import setup, Extension

module = Extension('tensor_scaler',
                   sources=['../../tensor.c', 'tensor_math.c'],
                   include_dirs=['libs'])

setup(name='YourPackage',
      version='1.0',
      description='This is a demo package',
      ext_modules=[module])
