from setuptools import setup, Extension

tensor = Extension('tensor',
                         sources=['tensor.c'],
                         include_dirs=['libs'])

tensor_scalers = Extension('tensor_scalers',
                           sources=['tensor_core/tensor_scalers/tensor_math.c'],
                           include_dirs=['libs'])

setup(name='TensorPackage',
      version='1.0',
      description='Tensor package',
      ext_modules=[tensor, tensor_scalers])
