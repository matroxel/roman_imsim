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
from .universe import setupCCM_ab
from .universe import addDust
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

path, filename = os.path.split(__file__)
sedpath_Star   = os.path.join(galsim.meta_data.share_dir, 'SEDs', 'vega.txt')

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

# # Uncomment for profiling
# # pr = cProfile.Profile()

# if __name__ == "__main__":
#     """
#     """

#     # Uncomment for profiling
#     # pr.enable()

#     t0 = time.time()

#     # process = psutil.Process(os.getpid())
#     # print(process.memory_info().rss/2**30)
#     # print(process.memory_info().vms/2**30)

#     try:
#         param_file = sys.argv[1]
#         filter_ = sys.argv[2]
#         dither = sys.argv[3]
#     except:
#         syntax_proc()

#     if (param_file.lower() == 'help') or (filter_.lower()=='help') or (dither.lower()=='help'):
#         syntax_proc()

#     # This instantiates the simulation based on settings in input param file
#     sim = wfirst_sim(param_file)
#     print(sim.params.keys())

#     if sim.params['condor']==True:
#         condor=True
#     else:
#         condor=False

#     # This sets up some things like input truth catalogs and empty objects
#     if dither=='setup':
#         sim.setup(filter_,dither,setup=True)
#     elif dither=='meds':
#         condor_build = False
#         if len(sys.argv)<5:
#             syntax_proc()
#         if sys.argv[4]=='setup':
#             setup = True
#             pix = -1
#         elif sys.argv[4]=='condor_build':
#             condor_build = True
#             setup = False
#             pix = -1
#         elif sys.argv[4]=='cleanup':
#             setup = True
#             pix = -1
#             m = accumulate_output_disk( param_file, filter_, pix, sim.comm, ignore_missing_files = False, setup = setup )
#             m.cleanup()
#         elif sys.argv[4]=='shape':
#             print(sys.argv)
#             if (sim.params['meds_from_file'] is not None) & (sim.params['meds_from_file'] != 'None'):
#                 pix = int(np.loadtxt(sim.params['meds_from_file'])[int(sys.argv[5])-1])
#             else:
#                 pix = int(sys.argv[5])
#             m = accumulate_output_disk( param_file, filter_, pix, sim.comm,shape=True, shape_iter = int(sys.argv[6]), shape_cnt = int(sys.argv[7]))
#             m.get_coadd_shape()
#             print('out of coadd_shape')
#             sys.exit()
#         else:
#             setup = False
#             if (sim.params['meds_from_file'] is not None) & (sim.params['meds_from_file'] != 'None'):
#                 pix = int(np.loadtxt(sim.params['meds_from_file'])[int(sys.argv[4])-1])
#             else:
#                 pix = int(sys.argv[4])
#         m = accumulate_output_disk( param_file, filter_, pix, sim.comm, ignore_missing_files = False, setup = setup, condor_build = condor_build )
#         if setup or condor_build:
#             print('exiting')
#             sys.exit()
#         m.comm.Barrier()
#         skip = False
#         if sim.rank==0:
#             for i in range(1,sim.size):
#                 m.comm.send(m.skip, dest=i)
#             skip = m.skip
#         else:
#             skip = m.comm.recv(source=0)
#         if not skip:
#             m.comm.Barrier()
#             if not condor:
#                 m.get_coadd_shape()
#             print('out of coadd_shape')
#             # print 'commented out finish()'
#             m.finish(condor=sim.params['condor'])
#             # pr.disable()
#             # ps = pstats.Stats(pr).sort_stats('time')
#             # ps.print_stats(200)

#     else:

#         if (sim.params['dither_from_file'] is not None) & (sim.params['dither_from_file'] != 'None'):
#             dither=np.loadtxt(sim.params['dither_from_file'])[int(dither)-1] # Assumes array starts with 1
#         print(sys.argv)
#         sca = int(sys.argv[4])
#         if 'verify_output' in sys.argv:
#             if sim.check_file(sca,int(dither),filter_):
#                 print('exists',dither,sca)
#                 sys.exit()
#         if sim.setup(filter_,int(dither),sca=sca):
#             sys.exit()

#         #tmp_name_id = int(sys.argv[6])

#         # Loop over SCAs
#         sim.comm.Barrier()
#         # This sets up the object that will simulate various wfirst detector effects, noise, etc. Instantiation creates a noise realisation for the image.
#         sim.modify_image = modify_image(sim.params)
#         # This is the main thing - iterates over galaxies for a given pointing and SCA and simulates them all
#         sim.comm.Barrier()
#         print(time.time()-t0)
#         # print(process.memory_info().rss/2**30)
#         # print(process.memory_info().vms/2**30)
#         sim.iterate_image()
#         sim.comm.Barrier()

#         # Uncomment for profiling
#         # pr.disable()
#         # ps = pstats.Stats(pr).sort_stats('time')
#         # ps.print_stats(50)

#     # if sim.params['condor']==True:
#     #     condor_cleanup(sim.params['out_path'])

# # test round galaxy recovered to cover wcs errors

# # same noise in different runs? same noise
# # distance to edge to reject images?