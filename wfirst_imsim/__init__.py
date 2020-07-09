"""
An image simulation suite for geneerating synthetic data 
for the Roman Space Telescope.

Built using GalSim functionality, which is
Copyright (c) 2012-2017 by the GalSim developers team on GitHub
https://github.com/GalSim-developers
"""

# from __future__ import division
# from __future__ import print_function

# from future import standard_library
# standard_library.install_aliases()
# from builtins import str
# from builtins import range
# from past.builtins import basestring
# from builtins import object
# from past.utils import old_div

import numpy as np
import healpy as hp
import sys, os, io
import math
import copy
import logging
import time
import yaml
import copy
import galsim as galsim
import galsim.wfirst as wfirst
import galsim.config.process as process
import galsim.des as des
# import ngmix
import fitsio as fio
import pickle as pickle
import pickletools
from astropy.time import Time
from mpi4py import MPI
# from mpi_pool import MPIPool
import cProfile, pstats, psutil
import glob
import shutil
import h5py
# from ngmix.jacobian import Jacobian
# from ngmix.observation import Observation, ObsList, MultiBandObsList,make_kobs
# from ngmix.galsimfit import GalsimRunner,GalsimSimple,GalsimTemplateFluxFitter
# from ngmix.guessers import R50FluxGuesser
# from ngmix.bootstrap import PSFRunner
# from ngmix import priors, joint_prior
# import mof
# import meds
# import psc

import matplotlib
matplotlib.use ('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pylab
from scipy.interpolate import interp1d

from .sim import wfirst_sim
from .output import accumulate_output_disk
from .image import draw_image 
from .detector import modify_image
from .universe import init_catalogs
from .telescope import pointing 
from .misc import ParamError
from .misc import except_func
from .misc import save_obj
from .misc import load_obj
from .misc import convert_dither_to_fits
from .misc import convert_gaia
from .misc import convert_galaxia
from .misc import create_radec_fits
from .misc import hsm
from .misc import get_filename
from .misc import get_filenames
from .misc import write_fits

# if sys.version_info[0] == 3:
#     string_types = str,
# else:
#     string_types = basestring,

BAD_MEASUREMENT = 1
CENTROID_SHIFT  = 2
MAX_CENTROID_SHIFT = 1.

big_fft_params = galsim.GSParams(maximum_fft_size=9796)

# Dict to convert GalSim WFIRST filter names to filter names for fluxes in:
# https://github.com/WFIRST-HLS-Cosmology/Docs/wiki/Home-Wiki#wfirstlsst-simulated-photometry-catalog-based-on-candels
filter_flux_dict = {
    'J129' : 'j_WFIRST',
    'F184' : 'F184W_WFIRST',
    'Y106' : 'y_WFIRST',
    'H158' : 'h_WFIRST'
}

# Converts galsim WFIRST filter names to indices in Chris' dither file.
filter_dither_dict = {
    'J129' : 3,
    'F184' : 1,
    'Y106' : 4,
    'H158' : 2
}
filter_dither_dict_ = {
    3:'J129',
    1:'F184',
    4:'Y106',
    2:'H158'
}


def condor_cleanup(out_path):

    import tarfile

    tar = tarfile.open('output.tar', 'w:gz')
    tar.add(out_path, arcname='output.tar')
    tar.close()

def syntax_proc():

    print('Possible syntax for running: ')
    print('')
    print('To set up truth catalog (must be run before any other modes):')
    print('    python simulate.py <yaml settings file> <filter> setup')
    print('')
    print('To run in image simulation mode: ')
    print('    python simulate.py <yaml settings file> <filter> <dither id> [verify string]')
    print('')
    print('To set up index information for meds making mode (must be run before attempting meds making): ')
    print('    python simulate.py <yaml settings file> <filter> meds setup')
    print('')
    print('To create a meds file and run shape measurement on it: ')
    print('    python simulate.py <yaml settings file> <filter> meds <pixel id>')
    print('')
    print('To cleanup meds/shape run and concatenate output files: ')
    print('    python simulate.py <yaml settings file> <filter> meds cleanup')
    print('')
    print('')
    print('Value definitions: ')
    print('yaml settings file : A yaml file with settings for the simulation run.')
    print('filter : Filter name. Either one of the keys of filter_dither_dict or None. None will interpret the filter from the dither simulation file. A string filter name will override the appropriate filter from the dither simulation.')
    print('dither id : An integer dither identifier. Either an index into the dither simulation file or an integer specifying a line from a provided file listing indices into the dither simulation file (useful for setting up parallel runs).')
    print("""verify string : A string 'verify_output'. Reruns simulation mode checking for failed runs and filling in missing files. Will only recalculate necessary missing files.""")
    print('pixel id : Healpix id for MEDS generation. Each MEDS file corresponds to a healpixel on the sky with nside defined in the yaml settings file. Can be either a healpixel index or an integer specifying a line from a provided file listing potential healpix indices (useful for setting up parallel runs).')
    sys.exit()
