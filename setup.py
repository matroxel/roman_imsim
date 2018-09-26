from setuptools import setup

setup(
   name='wfirst_imsim',
   version='0.0',
   description='Image simulations suite for WFIRST',
   license="MIT",
   author='Michael Troxel',
   author_email='michael.a.troxel@gmail.com',
   url="",
   packages=['wfirst_imsim'],
   install_requires=['galsim','ngmix', 'fitsio', 'astropy', 'mpi4py', 'meds', 'yaml', 'healpy', 'numpy', 'logging', 'matplotlib', 'scipy', 'ipython'],
)
