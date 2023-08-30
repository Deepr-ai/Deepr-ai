from setuptools import setup, Extension

tensor_module = Extension('tensor',
                          sources=['tensor.c'])

setup(
    name='tensor',
    version='1.0',
    description='Tensor Module',
    ext_modules=[tensor_module]
)
