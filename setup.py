from setuptools import setup

setup(
   name='roman_imsim',
   version='0.0',
   description='Image simulations suite for the Roman Space Telescope',
   license="MIT",
   author='Michael Troxel',
   author_email='michael.troxel@duke.edu',
   url="",
   packages=['roman_imsim'],
   install_requires=['galsim','ngmix', 'fitsio', 'astropy', 'mpi4py', 'meds', 'pyyaml', 'healpy', 'numpy', 'matplotlib', 'scipy', 'ipython'],
)
