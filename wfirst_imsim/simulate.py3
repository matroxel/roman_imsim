"""
An implementation of galaxy and star image simulations for WFIRST.
Built from the WFIRST GalSim module.

Built with elements from galsim demo13...
# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
# 
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

import wfirst_imsim


# from ngmix.jacobian import Jacobian
# from ngmix.observation import Observation, ObsList, MultiBandObsList,make_kobs
# from ngmix.galsimfit import GalsimRunner,GalsimSimple,GalsimTemplateFluxFitter
# from ngmix.guessers import R50FluxGuesser
# from ngmix.bootstrap import PSFRunner
# from ngmix import priors, joint_prior
# import mof
# import meds
# import psc

# import matplotlib
# matplotlib.use ('agg')
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from matplotlib.colors import LogNorm
# import matplotlib.gridspec as gridspec
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# import pylab
# from scipy.interpolate import interp1d

# path, filename = os.path.split(__file__)
# sedpath_Star   = os.path.join(galsim.meta_data.share_dir, 'SEDs', 'vega.txt')

# if sys.version_info[0] == 3:
#     string_types = str,
# else:
#     string_types = basestring,

# # Chip coordinates
# cptr = np.array([
# 0.002689724,  1.000000000,  0.181995021, -0.002070809, -1.000000000,  0.807383134,  1.000000000,  0.004769437,  1.028725015, -1.000000000, -0.000114163, -0.024579913,
# 0.003307633,  1.000000000,  1.203503349, -0.002719257, -1.000000000, -0.230036847,  1.000000000,  0.006091805,  1.028993582, -1.000000000, -0.000145757, -0.024586416,
# 0.003888409,  1.000000000,  2.205056241, -0.003335597, -1.000000000, -1.250685466,  1.000000000,  0.007389324,  1.030581048, -1.000000000, -0.000176732, -0.024624426,
# 0.007871078,  1.000000000, -0.101157485, -0.005906926, -1.000000000,  1.095802866,  1.000000000,  0.009147586,  2.151242511, -1.000000000, -0.004917673, -1.151541644,
# 0.009838715,  1.000000000,  0.926774753, -0.007965112, -1.000000000,  0.052835488,  1.000000000,  0.011913584,  2.150981875, -1.000000000, -0.006404157, -1.151413352,
# 0.011694346,  1.000000000,  1.935534773, -0.009927853, -1.000000000, -0.974276664,  1.000000000,  0.014630945,  2.153506744, -1.000000000, -0.007864196, -1.152784334,
# 0.011758070,  1.000000000, -0.527032681, -0.008410887, -1.000000000,  1.529873670,  1.000000000,  0.012002262,  3.264990040, -1.000000000, -0.008419930, -2.274065453,
# 0.015128555,  1.000000000,  0.510881058, -0.011918799, -1.000000000,  0.478274989,  1.000000000,  0.016194244,  3.262719942, -1.000000000, -0.011359106, -2.272508364,
# 0.018323436,  1.000000000,  1.530828790, -0.015281655, -1.000000000, -0.558879607,  1.000000000,  0.020320244,  3.264721809, -1.000000000, -0.014251259, -2.273955111,
# -0.002689724,  1.000000000,  0.181995021,  0.002070809, -1.000000000,  0.807383134,  1.000000000, -0.000114163, -0.024579913, -1.000000000,  0.004769437,  1.028725015,
# -0.003307633,  1.000000000,  1.203503349,  0.002719257, -1.000000000, -0.230036847,  1.000000000, -0.000145757, -0.024586416, -1.000000000,  0.006091805,  1.028993582,
# -0.003888409,  1.000000000,  2.205056241,  0.003335597, -1.000000000, -1.250685466,  1.000000000, -0.000176732, -0.024624426, -1.000000000,  0.007389324,  1.030581048,
# -0.007871078,  1.000000000, -0.101157485,  0.005906926, -1.000000000,  1.095802866,  1.000000000, -0.004917673, -1.151541644, -1.000000000,  0.009147586,  2.151242511,
# -0.009838715,  1.000000000,  0.926774753,  0.007965112, -1.000000000,  0.052835488,  1.000000000, -0.006404157, -1.151413352, -1.000000000,  0.011913584,  2.150981875,
# -0.011694346,  1.000000000,  1.935534773,  0.009927853, -1.000000000, -0.974276664,  1.000000000, -0.007864196, -1.152784334, -1.000000000,  0.014630945,  2.153506744,
# -0.011758070,  1.000000000, -0.527032681,  0.008410887, -1.000000000,  1.529873670,  1.000000000, -0.008419930, -2.274065453, -1.000000000,  0.012002262,  3.264990040,
# -0.015128555,  1.000000000,  0.510881058,  0.011918799, -1.000000000,  0.478274989,  1.000000000, -0.011359106, -2.272508364, -1.000000000,  0.016194244,  3.262719942,
# -0.018323436,  1.000000000,  1.530828790,  0.015281655, -1.000000000, -0.558879607,  1.000000000, -0.014251259, -2.273955111, -1.000000000,  0.020320244,  3.264721809 ])

# BAD_MEASUREMENT = 1
# CENTROID_SHIFT  = 2
# MAX_CENTROID_SHIFT = 1.

# big_fft_params = galsim.GSParams(maximum_fft_size=9796)

# # SCAs' central coordinates
# sca_center = np.array([
#     [21.94, 13.12],
#     [-22.09, -31,77],
#     [-22.24, -81.15],
#     [-65.82, 23.76],
#     [-66.32, -20.77],
#     [-66.82, -70.15],
#     [-109.70, 44.12],
#     [-110.46, 0.24],
#     [-111.56, -49.15],
#     [21.94, 13.12],
#     [22.09, -31.77],
#     [22.24, -81.15],
#     [65.82, 23.76],
#     [66.32, -20.77],
#     [66.82, -70.15],
#     [109.70, 44.12],
#     [110.46, 0.24],
#     [111.56, -49.15]])

# # Dict to convert GalSim WFIRST filter names to filter names for fluxes in:
# # https://github.com/WFIRST-HLS-Cosmology/Docs/wiki/Home-Wiki#wfirstlsst-simulated-photometry-catalog-based-on-candels
# filter_flux_dict = {
#     'J129' : 'j_WFIRST',
#     'F184' : 'F184W_WFIRST',
#     'Y106' : 'y_WFIRST',
#     'H158' : 'h_WFIRST'
# }

# # Converts galsim WFIRST filter names to indices in Chris' dither file.
# filter_dither_dict = {
#     'J129' : 3,
#     'F184' : 1,
#     'Y106' : 4,
#     'H158' : 2
# }
# filter_dither_dict_ = {
#     3:'J129',
#     1:'F184',
#     4:'Y106',
#     2:'H158'
# }


# class ParamError(Exception):
#   def __init__(self, value):
#     self.value = value
#   def __str__(self):
#     return repr(self.value)

# def except_func(logger, proc, k, res, t):
#     print(proc, k)
#     print(t)
#     raise res

# def save_obj(obj, name ):
#     """
#     Helper function to save some data as a pickle to disk.
#     """
#     with open(name, 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# def load_obj(name ):
#     """
#     Helper function to read some data from a pickle on disk.
#     """
#     with open(name, 'rb') as f:
#         return pickle.load(f)

# def convert_dither_to_fits(ditherfile='observing_sequence_hlsonly'):
#     """
#     Helper function to used to convert Chris survey dither file to fits and extract HLS part.
#     """

#     dither = np.genfromtxt(ditherfile+'.dat',dtype=None,names = ['date','f1','f2','ra','dec','pa','program','filter','f8','f9','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21'])
#     dither = dither[['date','ra','dec','pa','filter']][dither['program']==5]
#     fio.write(ditherfile+'.fits',dither,clobber=True)

#     return

# def convert_gaia():
#     """
#     Helper function to convert gaia data to star truth catalog.
#     """

#     n=100000000
#     ra=[-5,80.]
#     dec=[-64,3]
#     ra1 = np.random.rand(n)*(ra[1]-ra[0])/180.*np.pi+ra[0]/180.*np.pi
#     d0 = old_div((np.cos((dec[0]+90)/180.*np.pi)+1),2.)
#     d1 = old_div((np.cos((dec[1]+90)/180.*np.pi)+1),2.)
#     dec1 = np.arccos(2*np.random.rand(n)*(d1-d0)+2*d0-1)
#     out = np.empty(n,dtype=[('ra',float)]+[('dec',float)]+[('H158',float)]+[('J129',float)]+[('Y106',float)]+[('F184',float)])
#     out['ra']=ra1
#     out['dec']=dec1-old_div(np.pi,2.)

#     g_band     = galsim.Bandpass('/users/PCON0003/cond0083/GalSim/galsim/share/bandpasses/gaia_g.dat', wave_type='nm').withZeropoint('AB')
#     star_sed   = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')

#     gaia = fio.FITS('../distwf-result.fits.gz')[-1].read()['phot_g_mean_mag'][:]
#     h,b = np.histogram(gaia,bins=np.linspace(3,22.5,196))
#     b = old_div((b[1:]+b[:-1]),2)
#     x = np.random.choice(np.arange(len(b)),len(out),p=1.*h/np.sum(h),replace=True)
#     for i,filter_ in enumerate(['J129','F184','Y106','H158']):
#         print(filter_)
#         bpass = wfirst.getBandpasses(AB_zeropoint=True)[filter_]
#         b_=np.zeros(len(b))
#         for ind in range(len(b)):
#             star_sed_  = star_sed.withMagnitude(b[ind],g_band)
#             b_[ind]    = star_sed_.calculateMagnitude(bpass)
#         out[filter_]   = b_[x]

#     fio.write('gaia_stars.fits',out,clobber=True)

#     return

# def convert_galaxia():
#     """
#     Helper function to convert galaxia data to star truth catalog.
#     """

#     j_band     = galsim.Bandpass('/users/PCON0003/cond0083/GalSim/galsim/share/bandpasses/UKIRT_UKIDSS.J.dat.txt', wave_type='nm').withZeropoint('AB')
#     star_sed   = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')

#     g = fio.FITS('/users/PCON0003/cond0083/galaxia_stars.fits')[-1].read()
#     out = np.empty(len(g),dtype=[('ra',float)]+[('dec',float)]+[('H158',float)]+[('J129',float)]+[('Y106',float)]+[('F184',float)])
#     out['ra']=g['ra']
#     out['dec']=g['dec']
#     for i,filter_ in enumerate(['J129','F184','Y106','H158']):
#         print(filter_)
#         bpass = wfirst.getBandpasses(AB_zeropoint=True)[filter_]
#         star_sed_  = star_sed.withMagnitude(23,j_band)
#         factor    = star_sed_.calculateMagnitude(bpass)-23
#         out[filter_] = g['J']+factor

#     s = np.random.choice(np.arange(len(out)),len(out),replace=False)
#     out=out[s]
#     fio.write('galaxia_stars_full.fits',out,clobber=True)

#     return

# def create_radec_fits(ra=[25.,27.5],dec=[-27.5,-25.],n=1500000):
#     """
#     Helper function that just creates random positions within some ra,dec range.
#     """

#     ra1 = np.random.rand(n)*(ra[1]-ra[0])/180.*np.pi+ra[0]/180.*np.pi
#     d0 = old_div((np.cos((dec[0]+90)/180.*np.pi)+1),2.)
#     d1 = old_div((np.cos((dec[1]+90)/180.*np.pi)+1),2.)
#     dec1 = np.arccos(2*np.random.rand(n)*(d1-d0)+2*d0-1)
#     out = np.empty(n,dtype=[('ra',float)]+[('dec',float)])
#     out['ra']=ra1*180./np.pi
#     out['dec']=dec1*180./np.pi-90
#     fio.write('ra_'+str(ra[0])+'_'+str(ra[1])+'_dec_'+str(dec[0])+'_'+str(dec[1])+'_n_'+str(n)+'.fits.gz',out,clobber=True)

# def hsm(im, psf=None, wt=None):
#     """
#     Not used currently, but this is a helper function to run hsm via galsim.
#     """

#     out = np.zeros(1,dtype=[('e1','f4')]+[('e2','f4')]+[('T','f4')]+[('dx','f4')]+[('dy','f4')]+[('flag','i2')])
#     try:
#         if psf is not None:
#             shape_data = galsim.hsm.EstimateShear(im, psf, weight=wt, strict=False)
#         else:
#             shape_data = im.FindAdaptiveMom(weight=wt, strict=False)
#     except:
#         # print(' *** Bad measurement (caught exception).  Mask this one.')
#         out['flag'] |= BAD_MEASUREMENT
#         return out

#     if shape_data.moments_status != 0:
#         # print('status = ',shape_data.moments_status)
#         # print(' *** Bad measurement.  Mask this one.')
#         out['flag'] |= BAD_MEASUREMENT

#     out['dx'] = shape_data.moments_centroid.x - im.true_center().x
#     out['dy'] = shape_data.moments_centroid.y - im.true_center().y
#     if out['dx']**2 + out['dy']**2 > MAX_CENTROID_SHIFT**2:
#         # print(' *** Centroid shifted by ',out['dx'],out['dy'],'.  Mask this one.')
#         out['flag'] |= CENTROID_SHIFT

#     # Account for the image wcs
#     if im.wcs.isPixelScale():
#         out['e1'] = shape_data.observed_shape.g1
#         out['e2'] = shape_data.observed_shape.g2
#         out['T']  = 2 * shape_data.moments_sigma**2 * im.scale**2
#     else:
#         e1 = shape_data.observed_shape.e1
#         e2 = shape_data.observed_shape.e2
#         s = shape_data.moments_sigma

#         jac = im.wcs.jacobian(im.trueCenter())
#         M = np.matrix( [[ 1 + e1, e2 ], [ e2, 1 - e1 ]] ) * s*s
#         J = jac.getMatrix()
#         M = J * M * J.T
#         scale = np.sqrt(M/2./s/s)

#         e1 = old_div((M[0,0] - M[1,1]), (M[0,0] + M[1,1]))
#         e2 = old_div((2.*M[0,1]), (M[0,0] + M[1,1]))
#         out['T'] = M[0,0] + M[1,1]

#         shear = galsim.Shear(e1=e1, e2=e2)
#         out['e1'] = shear.g1
#         out['e2'] = shear.g2

#     return out

# def get_filename( out_path, path, name, var=None, name2=None, ftype='fits', overwrite=False, make=True ):
#     """
#     Helper function to set up a file path, and create the path if it doesn't exist.
#     """
#     if var is not None:
#         name += '_' + var
#     if name2 is not None:
#         name += '_' + name2
#     name += '.' + ftype
#     fpath = os.path.join(out_path,path)
#     if make:
#         if not os.path.exists(out_path):
#             os.mkdir(out_path)
#         if not os.path.exists(fpath):
#             os.mkdir(fpath)
#     filename = os.path.join(fpath,name)
#     if (overwrite)&(os.path.exists(filename)):
#         os.remove(filename)
#     return filename

# def get_filenames( out_path, path, name, var=None, name2=None, ftype='fits' ):
#     """
#     Helper function to set up a file path, and create the path if it doesn't exist.
#     """
#     if var is not None:
#         name += '_' + var
#     if name2 is not None:
#         name += '_' + name2
#     name += '*.' + ftype
#     fpath = os.path.join(out_path,path)
#     if not os.path.exists(out_path):
#         os.mkdir(out_path)
#     if not os.path.exists(fpath):
#         os.mkdir(fpath)
#     filename = os.path.join(fpath,name)
#     return glob.glob(filename)

# def write_fits(filename,img):

#     hdr={}
#     img.wcs.writeToFitsHeader(hdr,img.bounds)
#     hdr['GS_XMIN'] = hdr['GS_XMIN'][0]
#     hdr['GS_XMIN'] = hdr['GS_YMIN'][0]
#     hdr['GS_WCS']  = hdr['GS_WCS'][0]
#     fits = fio.FITS(filename,'rw')
#     fits.write(img.array)
#     fits[0].write_keys(hdr)
#     fits.close()

#     return

# def setupCCM_ab(wavelen):
#     """
#     Calculate a(x) and b(x) for CCM dust model. (x=1/wavelen).
#     If wavelen not specified, calculates a and b on the own object's wavelength grid.
#     Returns a(x) and b(x) can be common to many seds, wavelen is the same.
#     This method sets up extinction due to the model of
#     Cardelli, Clayton and Mathis 1989 (ApJ 345, 245)

#     Taken tempoarily for testing from https://github.com/lsst/sims_photUtils/blob/master/python/lsst/sims/photUtils/Sed.py
#     """
#     # This extinction law taken from Cardelli, Clayton and Mathis ApJ 1989.
#     # The general form is A_l / A(V) = a(x) + b(x)/R_V  (where x=1/lambda in microns),
#     # then different values for a(x) and b(x) depending on wavelength regime.
#     # Also, the extinction is parametrized as R_v = A_v / E(B-V).
#     # Magnitudes of extinction (A_l) translates to flux by a_l = -2.5log(f_red / f_nonred).
#     a_x = np.zeros(len(wavelen), dtype='float')
#     b_x = np.zeros(len(wavelen), dtype='float')
#     # Convert wavelength to x (in inverse microns).
#     x = np.empty(len(wavelen), dtype=float)
#     nm_to_micron = 1/1000.0
#     x = 1.0 / (wavelen * nm_to_micron)
#     # Dust in infrared 0.3 /mu < x < 1.1 /mu (inverse microns).
#     condition = (x >= 0.3) & (x <= 1.1)
#     if len(a_x[condition]) > 0:
#         y = x[condition]
#         a_x[condition] = 0.574 * y**1.61
#         b_x[condition] = -0.527 * y**1.61
#     # Dust in optical/NIR 1.1 /mu < x < 3.3 /mu region.
#     condition = (x >= 1.1) & (x <= 3.3)
#     if len(a_x[condition]) > 0:
#         y = x[condition] - 1.82
#         a_x[condition] = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4
#         a_x[condition] = a_x[condition] + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
#         b_x[condition] = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4
#         b_x[condition] = b_x[condition] - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
#     # Dust in ultraviolet and UV (if needed for high-z) 3.3 /mu< x< 8 /mu.
#     condition = (x >= 3.3) & (x < 5.9)
#     if len(a_x[condition]) > 0:
#         y = x[condition]
#         a_x[condition] = 1.752 - 0.316*y - 0.104/((y-4.67)**2 + 0.341)
#         b_x[condition] = -3.090 + 1.825*y + 1.206/((y-4.62)**2 + 0.263)
#     condition = (x > 5.9) & (x < 8)
#     if len(a_x[condition]) > 0:
#         y = x[condition]
#         Fa_x = np.empty(len(a_x[condition]), dtype=float)
#         Fb_x = np.empty(len(a_x[condition]), dtype=float)
#         Fa_x = -0.04473*(y-5.9)**2 - 0.009779*(y-5.9)**3
#         Fb_x = 0.2130*(y-5.9)**2 + 0.1207*(y-5.9)**3
#         a_x[condition] = 1.752 - 0.316*y - 0.104/((y-4.67)**2 + 0.341) + Fa_x
#         b_x[condition] = -3.090 + 1.825*y + 1.206/((y-4.62)**2 + 0.263) + Fb_x
#     # Dust in far UV (if needed for high-z) 8 /mu < x < 10 /mu region.
#     condition = (x >= 8) & (x <= 11.)
#     if len(a_x[condition]) > 0:
#         y = x[condition]-8.0
#         a_x[condition] = -1.073 - 0.628*(y) + 0.137*(y)**2 - 0.070*(y)**3
#         b_x[condition] = 13.670 + 4.257*(y) - 0.420*(y)**2 + 0.374*(y)**3
#     return a_x, b_x

# def addDust(a_x, b_x, A_v=None, ebv=None, R_v=3.1):
#     """
#     Add dust model extinction to the SED, modifying flambda and fnu.
#     Get a_x and b_x either from setupCCMab or setupODonnell_ab
#     Specify any two of A_V, E(B-V) or R_V (=3.1 default).

#     Taken tempoarily for testing from https://github.com/lsst/sims_photUtils/blob/master/python/lsst/sims/photUtils/Sed.py
#     """
#     _ln10_04 = 0.4*np.log(10.0)

#     # The extinction law taken from Cardelli, Clayton and Mathis ApJ 1989.
#     # The general form is A_l / A(V) = a(x) + b(x)/R_V  (where x=1/lambda in microns).
#     # Then, different values for a(x) and b(x) depending on wavelength regime.
#     # Also, the extinction is parametrized as R_v = A_v / E(B-V).
#     # The magnitudes of extinction (A_l) translates to flux by a_l = -2.5log(f_red / f_nonred).
#     #
#     # Input parameters for reddening can include any of 3 parameters; only 2 are independent.
#     # Figure out what parameters were given, and see if self-consistent.
#     if R_v == 3.1:
#         if A_v is None:
#             A_v = R_v * ebv
#         elif (A_v is not None) and (ebv is not None):
#             # Specified A_v and ebv, so R_v should be nondefault.
#             R_v = A_v / ebv
#     if (R_v != 3.1):
#         if (A_v is not None) and (ebv is not None):
#             calcRv = A_v / ebv
#             if calcRv != R_v:
#                 raise ValueError("CCM parametrization expects R_v = A_v / E(B-V);",
#                                  "Please check input values, because values are inconsistent.")
#         elif A_v is None:
#             A_v = R_v * ebv
#     # R_v and A_v values are specified or calculated.

#     A_lambda = (a_x + b_x / R_v) * A_v
#     # dmag_red(dust) = -2.5 log10 (f_red / f_nored) : (f_red / f_nored) = 10**-0.4*dmag_red
#     dust = np.exp(-A_lambda*_ln10_04)
#     return dust

# class pointing(object):
#     """
#     Class to manage and hold informaiton about a wfirst pointing, including WCS and PSF.
#     """

#     def __init__(self, params, logger, filter_=None, sca=None, dither=None, sca_pos=None, max_rad_from_boresight=0.009,rank=None):
#         """
#         Initializes some information about a pointing.

#         Input
#         params                  : Parameter dict.
#         logger                  : logger instance
#         filter_                 : The filter name for this pointing.
#         sca                     : The SCA number (1-18)
#         dither                  : The index of this pointing in the survey simulation file.
#         sca_pos                 : Used to simulate the PSF at a position other than the center
#                                     of the SCA.
#         max_rad_from_boresight  : Distance around pointing to attempt to simulate objects.
#         chip_enlarge            : Factor to enlarge chip geometry by to account for small
#                                     inaccuracies relative to precise WCS.
#         """

#         self.params             = params
#         self.ditherfile         = params['dither_file']
#         self.n_waves            = params['n_waves'] # Number of wavelenghts of PSF to simulate
#         self.approximate_struts = params['approximate_struts'] # Whether to approsimate struts
#         self.extra_aberrations  = params['extra_aberrations']  # Extra aberrations to include in the PSF model. See galsim documentation.

#         self.logger = logger
#         self.rank   = rank
#         self.sca    = None
#         self.PSF    = None
#         self.WCS    = None
#         self.dither = None
#         self.filter = None
#         self.los_motion = None

#         if filter_ is not None:
#             self.get_bpass(filter_)

#         if sca is not None:
#             self.update_sca(sca,sca_pos=sca_pos)

#         if dither is not None:
#             self.update_dither(dither)

#         self.bore           = max_rad_from_boresight
#         self.sbore2         = np.sin(old_div(max_rad_from_boresight,2.))
#         self.chip_enlarge   = params['chip_enlarge']

#     def get_bpass(self, filter_):
#         """
#         Read in the WFIRST filters, setting an AB zeropoint appropriate for this telescope given its
#         diameter and (since we didn't use any keyword arguments to modify this) using the typical
#         exposure time for WFIRST images.  By default, this routine truncates the parts of the
#         bandpasses that are near 0 at the edges, and thins them by the default amount.

#         Input
#         filter_ : Fiter name for this pointing.
#         """

#         self.filter = filter_
#         self.bpass  = wfirst.getBandpasses(AB_zeropoint=True)[self.filter]

#     def update_dither(self,dither):
#         """
#         This updates the pointing to a new dither position.

#         Input
#         dither     : Pointing index in the survey simulation file.
#         sca        : SCA number
#         """

#         self.dither = dither

#         d = fio.FITS(self.ditherfile)[-1][self.dither]

#         # Check that nothing went wrong with the filter specification.
#         # if filter_dither_dict[self.filter] != d['filter']:
#         #     raise ParamError('Requested filter and dither pointing do not match.')

#         self.ra     = d['ra']  * np.pi / 180. # RA of pointing
#         self.dec    = d['dec'] * np.pi / 180. # Dec of pointing
#         self.pa     = d['pa']  * np.pi / 180.  # Position angle of pointing
#         self.sdec   = np.sin(self.dec) # Here and below - cache some geometry stuff
#         self.cdec   = np.cos(self.dec)
#         self.sra    = np.sin(self.ra)
#         self.cra    = np.cos(self.ra)
#         self.spa    = np.sin(self.pa)
#         self.cpa    = np.cos(self.pa)
#         self.date   = Time(d['date'],format='mjd').datetime # Date of pointing


#         if self.filter is None:
#             self.get_bpass(filter_dither_dict_[d['filter']])

#     def update_sca(self,sca):
#         """
#         This assigns an SCA to the pointing, and evaluates the PSF and WCS.

#         Input
#         dither     : Pointing index in the survey simulation file.
#         sca        : SCA number
#         """

#         self.sca    = sca
#         self.get_wcs() # Get the new WCS
#         self.get_psf() # Get the new PSF
#         radec           = self.WCS.toWorld(galsim.PositionI(old_div(wfirst.n_pix,2),old_div(wfirst.n_pix,2)))
#         if self.rank==0:
#             print('SCA is at position ',old_div(radec.ra,galsim.degrees),old_div(radec.dec,galsim.degrees))
#         self.sca_sdec   = np.sin(radec.dec) # Here and below - cache some geometry  stuff
#         self.sca_cdec   = np.cos(radec.dec)
#         self.sca_sra    = np.sin(radec.ra)
#         self.sca_cra    = np.cos(radec.ra)

#     def get_psf(self, sca_pos=None, high_accuracy=False):
#         """
#         This updates the pointing to a new SCA, replacing the stored PSF to the new SCA.

#         Input
#         sca_pos : Used to simulate the PSF at a position other than the center of the SCA.
#         """

#         # Add extra aberrations that vary sca-to-sca across the focal plane
#         extra_aberrations = None
#         # gradient across focal plane
#         if 'gradient_aberration' in self.params:
#             if self.params['gradient_aberration']:
#                 extra_aberrations = sca_center[self.sca-1][1]*np.array(self.extra_aberrations)*np.sqrt(3.)/88.115
#         # random assignment chip-to-chip
#         if 'random_aberration' in self.params:
#             if self.params['random_aberration']:
#                 np.random.seed(self.sca)
#                 extra_aberrations = np.array(self.extra_aberrations)*np.random.rand()

#         # Time-dependent oscillation of the aberrations
#         # Unlike others, must pass unit self.extra_aberrations array
#         if 'oscillating_aberration' in self.params:
#             if self.params['oscillating_aberration']:
#                 extra_aberrations = np.array(self.extra_aberrations)*self.time_aberration()

#         # Define a high-frequency smearing to convolve the PSF by later
#         if 'los_motion' in self.params:
#             # symmetric smearing
#             if self.params['los_motion'] is not None:
#                 self.los_motion = galsim.Gaussian(fwhm=2.*np.sqrt(2.*np.log(2.))*self.params['los_motion'])
#             # assymetric smearing
#             if ('los_motion_e1' in self.params) and ('los_motion_e2' in self.params):
#                 if (self.params['los_motion_e1'] is not None) and (self.params['los_motion_e2'] is not None):
#                     self.los_motion = self.los_motion.shear(g1=self.params['los_motion_e1'],g2=self.params['los_motion_e2']) # assymetric jitter noise
#                 if (self.params['los_motion_e1'] is None) or (self.params['los_motion_e2'] is None):
#                     raise ParamError('Must provide both los motion e1 and e2.')

#         # assymetric smearing on random subset of pointings
#         if 'random_los_motion' in self.params:
#             if self.params['random_los_motion']:
#                 np.random.seed(self.dither)
#                 if np.random.rand()>0.15:
#                     self.los_motion = None

#         # aberration gradient across chip
#         if 'random_aberration_gradient' in self.params:
#             if self.params['random_aberration_gradient']:
#                 np.random.seed(self.sca)
#                 extra_aberrations = np.array(self.extra_aberrations)*np.random.rand()*np.sqrt(3.)/(wfirst.n_pix/2.)
#         else:
#             self.params['random_aberration_gradient'] = False

#         # No special changes, populate from yaml file
#         if extra_aberrations is None:
#             extra_aberrations = self.extra_aberrations

#         if self.params['random_aberration_gradient']:

#             self.PSF = None
#             self.extra_aberrations = extra_aberrations

#         elif 'gauss_psf' in self.params:
#             if self.params['gauss_psf'] is not None:
#                 self.PSF = galsim.Gaussian(half_light_radius=self.params['gauss_psf'])
#         else:

#             # print(self.sca,self.filter,sca_pos,self.bpass.effective_wavelength)
#             self.PSF = wfirst.getPSF(self.sca,
#                                     self.filter,
#                                     SCA_pos             = sca_pos,
#                                     approximate_struts  = self.approximate_struts,
#                                     n_waves             = self.n_waves,
#                                     logger              = self.logger,
#                                     wavelength          = self.bpass.effective_wavelength,
#                                     extra_aberrations   = extra_aberrations,
#                                     high_accuracy       = high_accuracy,
#                                     )

#         # sim.logger.info('Done PSF precomputation in %.1f seconds!'%(time.time()-t0))

#     def load_psf(self,pos,star=False,sca_pos=None, high_accuracy=False):
#         """
#         Interface to access self.PSF.

#         pos : GalSim PositionI
#         """
#         if self.params['random_aberration_gradient']:

#             np.random.seed(self.sca)
#             if np.random.rand()>0.5:
#                 i = pos.x
#             else:
#                 i = pos.y
#             return wfirst.getPSF(self.sca,
#                                 self.filter,
#                                 SCA_pos             = sca_pos,
#                                 approximate_struts  = self.approximate_struts,
#                                 n_waves             = self.n_waves,
#                                 logger              = self.logger,
#                                 wavelength          = self.bpass.effective_wavelength,
#                                 extra_aberrations   = self.extra_aberrations*(i-wfirst.n_pix/2.+0.5),
#                                 high_accuracy       = high_accuracy,
#                                 )

#         else:

#             if star:
#                 return self.PSF_high
#             return self.PSF

#         return

#     def time_aberration(self):
#         """
#         A time-varying aberration. Returns a function of the datetime of pointing to modulate the extra_aberrations.
#         """

#         delta_t = 60. # s
#         fid_wavelength=1293. # nm

#         total_T = 5*365*24*60*60 # mission time [s]

#         with open('time_aberration.pickle', 'rb') as file:
#             ft=pickle.load(file,encoding='bytes')

#         t=np.linspace(0,total_T,num=len(ft))
#         ft_interp=interp1d(t,ft)

#         date = fio.FITS(self.ditherfile)[-1].read()['date']
#         mission_start_time = Time(min(date),format='mjd')# format='mjd'

#         dither_time = Time(date[self.dither],format='mjd')
#         dt = (dither_time-mission_start_time).sec/delta_t

#         time_aberration = ft_interp(dt)/fid_wavelength

#         return time_aberration

#     def get_wcs(self):
#         """
#         Get the WCS for an observation at this position. We are not supplying a date, so the routine will assume it's the vernal equinox. The output of this routine is a dict of WCS objects, one for each SCA. We then take the WCS for the SCA that we are using.
#         """
#         self.WCS = wfirst.getWCS(world_pos  = galsim.CelestialCoord(ra=self.ra*galsim.radians, \
#                                                                     dec=self.dec*galsim.radians),
#                                 PA          = self.pa*galsim.radians,
#                                 date        = self.date,
#                                 SCAs        = self.sca,
#                                 PA_is_FPA   = True
#                                 )[self.sca]

#     def in_sca(self, ra, dec):
#         """
#         Check if ra, dec falls on approximate SCA area.

#         Input
#         ra  : Right ascension of object
#         dec : Declination of object
#         """

#         # Catch some problems, like the pointing not being defined
#         if self.dither is None:
#             raise ParamError('No dither defined to check ra, dec against.')

#         if self.sca is None:
#             raise ParamError('No sca defined to check ra, dec against.')

#         # # Discard any object greater than some dec from pointing
#         # if np.abs(dec-self.dec)>self.bore:
#         #     return False

#         # Position of the object in boresight coordinates
#         mX  = -self.sdec   * np.cos(dec) * np.cos(self.ra-ra) + self.cdec * np.sin(dec)
#         mY  =  np.cos(dec) * np.sin(self.ra-ra)

#         xi  = old_div(-(self.spa * mX + self.cpa * mY), 0.0021801102) # Image plane position in chips
#         yi  = old_div((self.cpa * mX - self.spa * mY), 0.0021801102)

#         # Check if object falls on SCA
#         if hasattr(ra,'__len__'):
#             return   np.where((cptr[0+12*(self.sca-1)]*xi+cptr[1+12*(self.sca-1)]*yi  \
#                                 <cptr[2+12*(self.sca-1)]+self.chip_enlarge)       \
#                             & (cptr[3+12*(self.sca-1)]*xi+cptr[4+12*(self.sca-1)]*yi  \
#                                 <cptr[5+12*(self.sca-1)]+self.chip_enlarge)       \
#                             & (cptr[6+12*(self.sca-1)]*xi+cptr[7+12*(self.sca-1)]*yi  \
#                                 <cptr[8+12*(self.sca-1)]+self.chip_enlarge)       \
#                             & (cptr[9+12*(self.sca-1)]*xi+cptr[10+12*(self.sca-1)]*yi \
#                                 <cptr[11+12*(self.sca-1)]+self.chip_enlarge))[0]

#         if    (cptr[0+12*(self.sca-1)]*xi+cptr[1+12*(self.sca-1)]*yi  \
#                 <cptr[2+12*(self.sca-1)]+self.chip_enlarge)       \
#             & (cptr[3+12*(self.sca-1)]*xi+cptr[4+12*(self.sca-1)]*yi  \
#                 <cptr[5+12*(self.sca-1)]+self.chip_enlarge)       \
#             & (cptr[6+12*(self.sca-1)]*xi+cptr[7+12*(self.sca-1)]*yi  \
#                 <cptr[8+12*(self.sca-1)]+self.chip_enlarge)       \
#             & (cptr[9+12*(self.sca-1)]*xi+cptr[10+12*(self.sca-1)]*yi \
#                 <cptr[11+12*(self.sca-1)]+self.chip_enlarge):

#             return True

#         return False

#     def near_pointing(self, ra, dec, min_date=None, max_date=None, sca=False):
#         """
#         Returns objects close to pointing, using usual orthodromic distance.

#         Input
#         ra  : Right ascension array of objects
#         dec : Declination array of objects
#         min_date, max_date : Optional date range for transients
#         """

#         x = np.cos(dec) * np.cos(ra)
#         y = np.cos(dec) * np.sin(ra)
#         z = np.sin(dec)

#         if sca:
#             d2 = (x - self.sca_cdec*self.sca_cra)**2 + (y - self.sca_cdec*self.sca_sra)**2 + (z - self.sca_sdec)**2
#         else:
#             d2 = (x - self.cdec*self.cra)**2 + (y - self.cdec*self.sra)**2 + (z - self.sdec)**2

#         if min_date is None:
#             return np.where(old_div(np.sqrt(d2),2.)<=self.sbore2)[0]
#         else:
#             return np.where((old_div(np.sqrt(d2),2.)<=self.sbore2) & (min_date<=self.mjd) & (self.mjd<=max_date))[0]

# class init_catalogs(object):
#     """
#     Build truth catalogs if they don't exist from input galaxy and star catalogs.
#     """


#     def __init__(self, params, pointing, gal_rng, rank, size, comm=None, setup=False):
#         """
#         Initiate the catalogs

#         Input
#         params   : Parameter dictionary
#         pointing : Pointing object
#         gal_rng  : Random generator [0,1]
#         rank     : Process rank
#         comm     : MPI comm object
#         """

#         self.pointing = pointing
#         self.rank = rank
#         if rank == 0:
#             # Set up file path. Check if output truth file path exists or if explicitly remaking galaxy properties
#             filename = get_filename(params['out_path'],
#                                     'truth',
#                                     params['output_truth'],
#                                     name2='truth_gal',
#                                     overwrite=params['overwrite'])

#             # Link to galaxy truth catalog on disk
#             self.gals  = self.init_galaxy(filename,params,pointing,gal_rng,setup)
#             # Link to star truth catalog on disk
#             self.stars = self.init_star(params)
#             # Link to supernova truth catalog on disk
#             sne = self.init_supernova(params)
#             if sne is not None:
#                 self.supernovae = self.init_supernova(params)[0]
#                 self.lightcurves = self.init_supernova(params)[1]
#             else:
#                 self.supernovae = None
#                 self.lightcurves = None
#             if setup:
#                 comm.Barrier()
#                 return

#             # print 'gal check',len(self.gals['ra'][:]),len(self.stars['ra'][:]),np.degrees(self.gals['ra'][:].min()),np.degrees(self.gals['ra'][:].max()),np.degrees(self.gals['dec'][:].min()),np.degrees(self.gals['dec'][:].max())

#             if comm is not None:
#                 # Pass gal_ind to other procs
#                 self.get_near_sca()
#                 # print 'gal check',len(self.gals['ra'][:]),len(self.stars['ra'][:]),np.degrees(self.gals['ra'][:].min()),np.degrees(self.gals['ra'][:].max()),np.degrees(self.gals['dec'][:].min()),np.degrees(self.gals['dec'][:].max())

#                 self.init_sed(params)

#                 for i in range(1,size):
#                     comm.send(self.gal_ind,  dest=i)
#                     comm.send(self.gals,  dest=i)

#                 # Pass star_ind to other procs
#                 for i in range(1,size):
#                     comm.send(self.star_ind,  dest=i)
#                     comm.send(self.stars,  dest=i)

#                 # Pass seds to other procs
#                 for i in range(1,size):
#                     comm.send(self.seds,  dest=i)

#                 # Pass sne to other procs
#                 for i in range(1,size):
#                     comm.send(self.supernova_ind, dest=i)
#                     comm.send(self.supernovae, dest=i)
#         else:
#             if setup:
#                 comm.Barrier()
#                 return

#             # Get gals
#             self.gal_ind = comm.recv(source=0)
#             self.gals = comm.recv(source=0)

#             # Get stars
#             self.star_ind = comm.recv(source=0)
#             self.stars = comm.recv(source=0)

#             # Get seds
#             self.seds = comm.recv(source=0)

#             # Get sne
#             self.supernova_ind = comm.recv(source=0)
#             self.supernovae = comm.recv(source=0)

#         self.gal_ind  = self.gal_ind[rank::size]
#         self.gals     = self.gals[rank::size]
#         self.star_ind = self.star_ind[rank::params['starproc']]
#         self.stars    = self.stars[rank::params['starproc']]
#         self.supernova_ind = self.supernova_ind[rank::size]
#         self.supernovae = self.supernovae[rank::size]

#     def close(self):

#         self.gal_ind  = None
#         self.gals     = None
#         self.star_ind = None
#         self.stars    = None
#         self.supernova_ind = None
#         self.supernovae = None

#     def get_near_sca(self):

#         self.gal_ind  = self.pointing.near_pointing( self.gals['ra'][:], self.gals['dec'][:] )
#         # print len(self.gal_ind),len(self.gals['ra'][:])
#         if len(self.gal_ind)==0:
#             self.gal_ind = []
#             self.gals = []
#         else:
#             self.gals = self.gals[self.gal_ind]

#         self.star_ind = self.pointing.near_pointing( self.stars['ra'][:], self.stars['dec'][:] )
#         # print len(self.star_ind),len(self.stars['ra'][:])
#         if len(self.star_ind)==0:
#             self.star_ind = []
#             self.stars = []
#         else:
#             self.stars = self.stars[self.star_ind]

#         mask_sca      = self.pointing.in_sca(self.gals['ra'][:],self.gals['dec'][:])
#         if len(mask_sca)==0:
#             self.gal_ind = []
#             self.gals = []
#         else:
#             self.gals    = self.gals[mask_sca]
#             self.gal_ind = self.gal_ind[mask_sca]

#         mask_sca_star = self.pointing.in_sca(self.stars['ra'][:],self.stars['dec'][:])
#         if len(mask_sca_star)==0:
#             self.star_ind = []
#             self.stars = []
#         else:   
#             self.stars    = self.stars[mask_sca_star]
#             self.star_ind = self.star_ind[mask_sca_star]

#         self.supernova_ind = self.pointing.near_pointing( self.supernovae['ra'][:], 
#                                                         self.supernovae['dec'][:], 
#                                                         min_date=self.lightcurves['mjd'][self.supernovae['ptrobs_min']][:], 
#                                                         max_date=self.lightcurves['mjd'][self.supernovae['ptrobs_max'] - 1][:]) 
#         self.supernovae = self.supernovae[self.supernova_ind]

#     def add_mask(self,gal_mask,star_mask=None,supernova_mask=None):

#         if gal_mask.dtype == bool:
#             self.gal_mask = np.where(gal_mask)[0]
#         else:
#             self.gal_mask = gal_mask

#         if star_mask is None:
#             self.star_mask = []
#         elif star_mask.dtype == bool:
#             self.star_mask = np.where(star_mask)[0]
#         else:
#             self.star_mask = star_mask
#         if supernova_mask is None:
#             self.supernova_mask = []
#         elif supernova_mask.dtype == bool:
#             self.supernova_mask = np.where(supernova_mask)[0]
#         else:
#             self.supernova_mask = supernova_mask

#     def get_gal_length(self):

#         return len(self.gal_ind)

#     def get_star_length(self):

#         return len(self.star_ind)

#     def get_supernova_length(self):
        
#         return len(self.supernova_mask)

#     def get_gal_list(self):

#         return self.gal_ind,self.gals
#         return self.gal_ind[self.gal_mask],self.gals[self.gal_mask]

#     def get_star_list(self):

#         return self.star_ind,self.stars
#         return self.star_ind[self.star_mask],self.stars[self.star_mask]

#     def get_supernova_list(self):
        
#         return self.supernova_ind,self.supernovae
#         return self.supernova_ind[self.supernova_mask],self.supernovae[self.supernova_mask]

#     def get_gal(self,ind):

#         return self.gal_ind[ind],self.gals[ind]
#         return self.gal_ind[self.gal_mask[ind]],self.gals[self.gal_mask[ind]]

#     def get_star(self,ind):

#         return self.star_ind[ind],self.stars[ind]
#         return self.star_ind[self.star_mask[ind]],self.stars[self.star_mask[ind]]

#     def get_supernova(self,ind):

#         return self.supernova_ind[ind],self.supernovae[ind]
#         return self.supernova_ind[self.supernova_mask[ind]],self.supernovae[self.supernova_mask[ind]]

#     def dump_truth_gal(self,filename,store):
#         """
#         Write galaxy truth catalog to disk.

#         Input
#         filename    : Fits filename
#         store       : Galaxy truth catalog
#         """

#         fio.write(filename,store,clobber=True)

#         return fio.FITS(filename)[-1]

#     def load_truth_gal(self,filename,params):
#         """
#         Load galaxy truth catalog from disk.

#         Input
#         filename    : Fits filename
#         """

#         if 'tmpdir' in params:
#             filename2 = get_filename(params['tmpdir'],
#                                     'truth',
#                                     params['output_truth'],
#                                     name2='truth_gal',
#                                     overwrite=params['overwrite'])
#             if not params['overwrite']:
#                 if not os.path.exists(filename2):
#                     shutil.copy(filename,filename2)

#         store = fio.FITS(filename2)[-1]

#         return store

#     def fwhm_to_hlr(self,fwhm):
#         """
#         Convert full-width half-maximum to half-light radius in units of arcseconds.

#         Input
#         fwhm : full-width half-maximum
#         """

#         radius = fwhm * 0.06 / 2. # 1 pix = 0.06 arcsec, factor 2 to convert to hlr

#         return radius

#     def init_galaxy(self,filename,params,pointing,gal_rng,setup):
#         """
#         Does the work to return a random, unique object property list (truth catalog).

#         Input
#         filname  : Filename of galaxy truth catalog.
#         params   : Parameter dict
#         pointing : pointing object
#         gal_rng  : Random generator [0,1]
#         """

#         # This is a placeholder option to allow different galaxy simulatin methods later if necessary
#         if params['gal_type'] == 0:
#             # Analytic profile - sersic disk


#             if not setup:
#                 if os.path.exists(filename):
#                     # Truth file exists and no instruction to overwrite it, so load existing truth file with galaxy properties
#                     return self.load_truth_gal(filename,params)
#                 else:
#                     raise ParamError('No truth file to load.')

#             if (not params['overwrite']) and (os.path.exists(filename)):
#                 print('Reusing existing truth file.')
#                 return None

#             # Make sure galaxy distribution filename is well-formed and link to it
#             if isinstance(params['gal_dist'],string_types):
#                 # Provided an ra,dec catalog of object positions.
#                 radec_file = fio.FITS(params['gal_dist'])[-1]
#             else:
#                 raise ParamError('Bad gal_dist filename.')

#             print('-----building truth catalog------')
#             # Read in file with photometry/size/redshift distribution similar to WFIRST galaxies
#             phot       = fio.FITS(params['gal_sample'])[-1].read(columns=['fwhm','redshift',filter_flux_dict['J129'],filter_flux_dict['F184'],filter_flux_dict['Y106'],filter_flux_dict['H158']])
#             pind_list_ = np.ones(len(phot)).astype(bool) # storage list for original index of photometry catalog
#             pind_list_ = pind_list_&(phot[filter_flux_dict['J129']]<99)&(phot[filter_flux_dict['J129']]>0) # remove bad mags
#             pind_list_ = pind_list_&(phot[filter_flux_dict['F184']]<99)&(phot[filter_flux_dict['F184']]>0) # remove bad mags
#             pind_list_ = pind_list_&(phot[filter_flux_dict['Y106']]<99)&(phot[filter_flux_dict['Y106']]>0) # remove bad mags
#             pind_list_ = pind_list_&(phot[filter_flux_dict['H158']]<99)&(phot[filter_flux_dict['H158']]>0) # remove bad mags
#             pind_list_ = pind_list_&(phot['redshift']>0)&(phot['redshift']<5) # remove bad redshifts
#             pind_list_ = np.where(pind_list_)[0]

#             n_gal = radec_file.read_header()['NAXIS2']

#             # Create minimal storage array for galaxy properties
#             store = np.ones(n_gal, dtype=[('gind','i4')]
#                                         +[('ra',float)]
#                                         +[('dec',float)]
#                                         +[('g1','f4')]
#                                         +[('g2','f4')]
#                                         +[('int_e1','f4')]
#                                         +[('int_e2','f4')]
#                                         +[('rot','f4')]
#                                         +[('size','f4')]
#                                         +[('z','f4')]
#                                         +[('J129','f4')]
#                                         +[('F184','f4')]
#                                         +[('Y106','f4')]
#                                         +[('H158','f4')]
#                                         +[('pind','i4')]
#                                         +[('bflux','f4')]
#                                         +[('dflux','f4')])
#             store['gind']       = np.arange(n_gal) # Index array into original galaxy position catalog
#             store['ra']         = radec_file['ra'][:]*np.pi/180. # Right ascension
#             store['dec']        = radec_file['dec'][:]*np.pi/180. # Declination
#             r_ = np.zeros(n_gal)
#             gal_rng.generate(r_)
#             store['pind']       = pind_list_[(r_*len(pind_list_)).astype(int)] # Index array into original galaxy photometry catalog
#             r_ = np.zeros(int(old_div(n_gal,2))+n_gal%2)
#             gal_rng.generate(r_)
#             store['rot'][0::2]  = r_*2.*np.pi # Random rotation (every pair of objects is rotated 90 deg to cancel shape noise)
#             store['rot'][1::2]  = store['rot'][0:n_gal-n_gal%2:2]+np.pi
#             store['rot'][store['rot']>2.*np.pi]-=2.*np.pi
#             r_ = np.zeros(n_gal)
#             gal_rng.generate(r_)
#             r_ = (r_*len(params['shear_list'])).astype(int)
#             np.random.seed(seed=params['random_seed'])
#             store['g1']         = np.array(params['shear_list'])[r_,0] # Shears to apply to galaxy
#             store['g2']         = np.array(params['shear_list'])[r_,1]
#             store['int_e1']     = np.random.normal(scale=0.27,size=n_gal) # Intrinsic shape of galaxy
#             store['int_e2']     = np.random.normal(scale=0.27,size=n_gal)
#             store['int_e1'][store['int_e1']>0.7] = 0.7
#             store['int_e2'][store['int_e2']>0.7] = 0.7
#             store['int_e1'][store['int_e1']<-0.7] = -0.7
#             store['int_e2'][store['int_e2']<-0.7] = -0.7
#             if params['gal_model'] == 'disk': # Disk only model, no bulge or knot flux
#                 store['bflux']  = np.zeros(n_gal)
#                 store['dflux']  = np.ones(n_gal)
#             elif params['gal_model'] == 'bulge': # Bulge only model, no disk or knot flux
#                 store['bflux']  = np.ones(n_gal)
#                 store['dflux']  = np.zeros(n_gal)
#             else: # General composite model. bflux = bulge flux fraction. dflux*(1-bflux) = disk flux fraction. Remaining flux is in form of star-knots, (1-bflux)*(1-dflux). Knot flux is capped at 50% of disk flux.
#                 r_ = np.zeros(n_gal)
#                 gal_rng.generate(r_)
#                 store['bflux']  = r_
#                 r_ = np.zeros(n_gal)
#                 gal_rng.generate(r_)
#                 store['dflux']  = old_div(r_,4.)+0.75
#             store['size']       = self.fwhm_to_hlr(phot['fwhm'][store['pind']]) # half-light radius
#             store['z']          = phot['redshift'][store['pind']] # redshift
#             for f in list(filter_dither_dict.keys()):
#                 store[f]        = phot[filter_flux_dict[f]][store['pind']] # magnitude in this filter
#             for name in store.dtype.names:
#                 print(name,np.mean(store[name]),np.min(store[name]),np.max(store[name]))

#             # Save truth file with galaxy properties
#             return self.dump_truth_gal(filename,store)

#             print('-------truth catalog built-------')

#         else:
#             raise ParamError('COSMOS profiles not currently implemented.')
#             # cosmos gal not guaranteed to work. uncomment at own risk
#             # # Cosmos real or parametric objects
#             # if self.params['gal_type'] == 1:
#             #     use_real = False
#             #     gtype = 'parametric'
#             # else:
#             #     use_real = True
#             #     gtype = 'real'

#             # # Load cosmos catalog
#             # cat = galsim.COSMOSCatalog(self.params['cat_name'], dir=self.params['cat_dir'], use_real=use_real)
#             # self.logger.info('Read in %d galaxies from catalog'%cat.nobjects)

#             # rand_ind = []
#             # for i in range(self.params['gal_n_use']):
#             #     # Select unique cosmos index list with length gal_n_use.
#             #     rand_ind.append(int(self.gal_rng()*cat.nobjects))
#             # # Make object list of unique cosmos galaxies
#             # self.obj_list = cat.makeGalaxy(rand_ind, chromatic=True, gal_type=gtype)

#     def init_star(self,params):
#         """
#         Compiles a list of stars properties to draw.
#         Not working with new structure yet.

#         Input
#         params   : parameter dict
#         """

#         # Make sure star catalog filename is well-formed and link to it
#         if isinstance(params['star_sample'],string_types):
#             # Provided a catalog of star positions and properties.
#             if 'tmpdir' in params:
#                 filename2 = get_filename(params['tmpdir'],
#                                         'truth',
#                                         params['output_truth'],
#                                         name2='truth_star',
#                                         overwrite=params['overwrite'])
#                 shutil.copy(params['star_sample'],filename2)
#                 stars = fio.FITS(filename2)[-1]
#             else:            
#                 stars = fio.FITS(params['star_sample'])[-1]
#             self.n_star = stars.read_header()['NAXIS2']
#         else:
#             return None

#         # # Cut really bright stars that take too long to draw for now
#         # mask = np.ones(len(stars),dtype=bool)
#         # for f in filter_dither_dict.keys():
#         #     mask = mask & (stars_[f]<1e5)
#         # stars = stars[mask]

#         return stars

#     def init_supernova(self,params):
#         if isinstance(params['supernovae'],string_types):
            
#             # Given a lightcurve Phot.fits file.
#             with fits.open(params['supernovae'] + "_HEAD.FITS") as sn:
#                 supernovae = sn[1].data
#                 self.n_supernova = sn[1].header['NAXIS2']
#             with fits.open(params['supernovae'] + "_PHOT.FITS") as light:
#                 lightcurves = light[1].data
#         else:
#             return None
#         return supernovae,lightcurves

#     def init_sed(self,params):
#         """
#         Loads the relevant SEDs into memory

#         Input 
#         params   : parameter dict
#         """

#         if not params['dc2']:
#             return None

#         filename = get_filename(params['out_path'],
#                                 'truth',
#                                 params['output_truth'],
#                                 name2='truth_sed',
#                                 overwrite=False, ftype='h5')

#         if 'tmpdir' in params:
#             filename2 = get_filename(params['tmpdir'],
#                                 'truth',
#                                 params['output_truth'],
#                                 name2='truth_sed',
#                                 overwrite=False, ftype='h5')
#             if not params['overwrite']:
#                 if not os.path.exists(filename2):
#                     shutil.copy(filename,filename2)
#         else:
#             filename2 = filename

#         sedfile = h5py.File(filename2,mode='r')

#         self.seds = {}
#         for s in np.unique(self.gals['sed']):
#             if s=='':
#                 continue
#             self.seds[s] = sedfile[s.lstrip().rstrip()][:]

#         for s in np.unique(self.stars['sed']):
#             if s=='':
#                 continue
#             self.seds[s] = sedfile[s.lstrip().rstrip()][:]

#         return self.seds

# class modify_image(object):
#     """
#     Class to simulate non-idealities and noise of wfirst detector images.
#     """

#     def __init__(self,params):
#         """
#         Set up noise properties of image

#         Input
#         params  : parameter dict
#         rng     : Random generator
#         """

#         self.params    = params

#     def add_effects(self,im,pointing,radec,local_wcs,rng,phot=False, ps_save=False):
#         """
#         Add detector effects for WFIRST.

#         Input:
#         im        : Postage stamp or image.
#         pointing  : Pointing object
#         radec     : World coordinate position of image
#         local_wcs : The local WCS
#         phot      : photon shooting mode

#         Preserve order:
#         1) add_background
#         2) add_poisson_noise
#         3) recip_failure
#         4) quantize
#         5) dark_current
#         6) add_persistence
#         7) nonlinearity
#         8) interpix_cap
#         9) Read noise
#         10) e_to_ADU
#         11) quantize

#         Where does persistence get added? Immediately before/after background?
#         Chien-Hao: I added persistence between dark current and nonlinearity.
#         """

#         self.rng       = rng
#         self.noise     = self.init_noise_model()

#         im, sky_image = self.add_background(im,pointing,radec,local_wcs,phot=phot) # Add background to image and save background
#         im = self.add_poisson_noise(im,sky_image,phot=phot) # Add poisson noise to image
#         im = self.recip_failure(im) # Introduce reciprocity failure to image
#         im.quantize() # At this point in the image generation process, an integer number of photons gets detected
#         im = self.dark_current(im) # Add dark current to image
#         if ps_save: #don't apply persistence for stamps
#             im = self.add_persistence(im, pointing)
#         im = self.nonlinearity(im) # Apply nonlinearity
#         im = self.interpix_cap(im) # Introduce interpixel capacitance to image.
#         im = self.add_read_noise(im)
#         im = self.e_to_ADU(im) # Convert electrons to ADU
#         im.quantize() # Finally, the analog-to-digital converter reads in an integer value.
#         # Note that the image type after this step is still a float. If we want to actually
#         # get integer values, we can do new_img = galsim.Image(im, dtype=int)
#         # Since many people are used to viewing background-subtracted images, we return a
#         # version with the background subtracted (also rounding that to an int).
#         im,sky_image = self.finalize_background_subtract(im,sky_image)
#         # im = galsim.Image(im, dtype=int)
#         # get weight map
#         if not self.params['use_background']:
#             return im,None
#         sky_image.invertSelf()

#         return im, sky_image


#     def add_effects_flat(self,im,phot=False):
#         """
#         Add detector effects for WFIRST.

#         Input:
#         im        : Postage stamp or image.
#         pointing  : Pointing object
#         radec     : World coordinate position of image
#         local_wcs : The local WCS
#         phot      : photon shooting mode

#         Preserve order:
#         1) add_background
#         2) add_poisson_noise
#         3) recip_failure
#         4) quantize
#         5) dark_current
#         6) nonlinearity
#         7) interpix_cap
#         8) Read noise
#         9) e_to_ADU
#         10) quantize

#         Where does persistence get added? Immediately before/after background?
#         """

#         # im = self.add_poisson_noise(im,sky_image,phot=phot) # Add poisson noise to image
#         im = self.recip_failure(im) # Introduce reciprocity failure to image
#         im.quantize() # At this point in the image generation process, an integer number of photons gets detected
#         im = self.dark_current(im) # Add dark current to image
#         im = self.nonlinearity(im) # Apply nonlinearity
#         im = self.interpix_cap(im) # Introduce interpixel capacitance to image.
#         im = self.add_read_noise(im)
#         im = self.e_to_ADU(im) # Convert electrons to ADU
#         im.quantize() # Finally, the analog-to-digital converter reads in an integer value.

#         return im

#     def get_eff_sky_bg(self,pointing,radec):
#         """
#         Calculate effective sky background per pixel for nominal wfirst pixel scale.

#         Input
#         pointing            : Pointing object
#         radec               : World coordinate position of image
#         """

#         sky_level = wfirst.getSkyLevel(pointing.bpass, world_pos=radec, date=pointing.date)
#         sky_level *= (1.0 + wfirst.stray_light_fraction)*wfirst.pixel_scale**2

#         return sky_level


#     def add_background(self,im,pointing,radec,local_wcs,sky_level=None,thermal_backgrounds=None,phot=False):
#         """
#         Add backgrounds to image (sky, thermal).

#         First we get the amount of zodaical light for a position corresponding to the position of
#         the object. The results are provided in units of e-/arcsec^2, using the default WFIRST
#         exposure time since we did not explicitly specify one. Then we multiply this by a factor
#         >1 to account for the amount of stray light that is expected. If we do not provide a date
#         for the observation, then it will assume that it's the vernal equinox (sun at (0,0) in
#         ecliptic coordinates) in 2025.

#         Input
#         im                  : Image
#         pointing            : Pointing object
#         radec               : World coordinate position of image
#         local_wcs           : Local WCS
#         sky_level           : The sky level. None uses current specification.
#         thermal_backgrounds : The thermal background of instrument. None uses current specification.
#         phot                : photon shooting mode
#         """

#         # If requested, dump an initial fits image to disk for diagnostics
#         if self.params['save_diff']:
#             orig = im.copy()
#             orig.write('orig.fits')

#         # If effect is turned off, return image unchanged
#         if not self.params['use_background']:
#             return im,None

#         # Build current specification sky level if sky level not given
#         if sky_level is None:
#             sky_level = wfirst.getSkyLevel(pointing.bpass, world_pos=radec, date=pointing.date)
#             sky_level *= (1.0 + wfirst.stray_light_fraction)
#         # Make a image of the sky that takes into account the spatially variable pixel scale. Note
#         # that makeSkyImage() takes a bit of time. If you do not care about the variable pixel
#         # scale, you could simply compute an approximate sky level in e-/pix by multiplying
#         # sky_level by wfirst.pixel_scale**2, and add that to final_image.

#         # Create sky image
#         sky_stamp = galsim.Image(bounds=im.bounds, wcs=local_wcs)
#         local_wcs.makeSkyImage(sky_stamp, sky_level)

#         # This image is in units of e-/pix. Finally we add the expected thermal backgrounds in this
#         # band. These are provided in e-/pix/s, so we have to multiply by the exposure time.
#         if thermal_backgrounds is None:
#             sky_stamp += wfirst.thermal_backgrounds[pointing.filter]*wfirst.exptime
#         else:
#             sky_stamp += thermal_backgrounds*wfirst.exptime

#         # Adding sky level to the image.
#         if not phot:
#             im += sky_stamp

#         # If requested, dump a post-change fits image to disk for diagnostics
#         if self.params['save_diff']:
#             prev = im.copy()
#             diff = prev-orig
#             diff.write('sky_a.fits')

#         return im,sky_stamp

#     def init_noise_model(self):
#         """
#         Generate a poisson noise model.
#         """

#         return galsim.PoissonNoise(self.rng)

#     def add_poisson_noise(self,im,sky_image,phot=False):
#         """
#         Add pre-initiated poisson noise to image.

#         Input
#         im : image
#         """

#         # If effect is turned off, return image unchanged
#         if not self.params['use_poisson_noise']:
#             return im

#         # Check if noise initiated
#         if self.noise is None:
#             self.init_noise_model()

#         # Add poisson noise to image
#         if phot:
#             sky_image_ = sky_image.copy()
#             sky_image_.addNoise(self.noise)
#             im += sky_image_
#         else:
#             im.addNoise(self.noise)

#         # If requested, dump a post-change fits image to disk for diagnostics. Both cumulative and iterative delta.
#         if self.params['save_diff']:
#             diff = im-prev
#             diff.write('noise_a.fits')
#             diff = im-orig
#             diff.write('noise_b.fits')
#             prev = im.copy()

#         return im

#     def recip_failure(self,im,exptime=wfirst.exptime,alpha=wfirst.reciprocity_alpha,base_flux=1.0):
#         """
#         Introduce reciprocity failure to image.

#         Reciprocity, in the context of photography, is the inverse relationship between the
#         incident flux (I) of a source object and the exposure time (t) required to produce a given
#         response(p) in the detector, i.e., p = I*t. However, in NIR detectors, this relation does
#         not hold always. The pixel response to a high flux is larger than its response to a low
#         flux. This flux-dependent non-linearity is known as 'reciprocity failure', and the
#         approximate amount of reciprocity failure for the WFIRST detectors is known, so we can
#         include this detector effect in our images.

#         Input
#         im        : image
#         exptime   : Exposure time
#         alpha     : Reciprocity alpha
#         base_flux : Base flux
#         """

#         # If effect is turned off, return image unchanged
#         if not self.params['use_recip_failure']:
#             return im

#         # Add reciprocity effect
#         im.addReciprocityFailure(exp_time=exptime, alpha=alpha, base_flux=base_flux)

#         # If requested, dump a post-change fits image to disk for diagnostics. Both cumulative and iterative delta.
#         if self.params['save_diff']:
#             diff = im-prev
#             diff.write('recip_a.fits')
#             diff = im-orig
#             diff.write('recip_b.fits')
#             prev = im.copy()

#         return im

#     def dark_current(self,im,dark_current=None):
#         """
#         Adding dark current to the image.

#         Even when the detector is unexposed to any radiation, the electron-hole pairs that
#         are generated within the depletion region due to finite temperature are swept by the
#         high electric field at the junction of the photodiode. This small reverse bias
#         leakage current is referred to as 'dark current'. It is specified by the average
#         number of electrons reaching the detectors per unit time and has an associated
#         Poisson noise since it is a random event.

#         Input
#         im           : image
#         dark_current : The dark current to apply
#         """

#         # If effect is turned off, return image unchanged
#         if not self.params['use_dark_current']:
#             return im

#         # If dark_current is not provided, calculate what it should be based on current specifications
#         self.dark_current_ = dark_current
#         if self.dark_current_ is None:
#             self.dark_current_ = wfirst.dark_current*wfirst.exptime

#         # Add dark current to image
#         dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(self.rng, self.dark_current_))
#         im.addNoise(dark_noise)

#         # NOTE: Sky level and dark current might appear like a constant background that can be
#         # simply subtracted. However, these contribute to the shot noise and matter for the
#         # non-linear effects that follow. Hence, these must be included at this stage of the
#         # image generation process. We subtract these backgrounds in the end.

#         # If requested, dump a post-change fits image to disk for diagnostics. Both cumulative and iterative delta.
#         if self.params['save_diff']:
#             diff = im-prev
#             diff.write('dark_a.fits')
#             diff = im-orig
#             diff.write('dark_b.fits')
#             prev = im.copy()

#         return im

#     def add_persistence(self, img, pointing):
#         """
#         Applying the persistence effect.

#         Even after reset, some charges from prior illuminations are trapped in defects of semiconductors.
#         Trapped charges are gradually released and generate the flux-dependent persistence signal.
#         Here we adopt the same fermi-linear model to describe the illumination dependence and time dependence
#         of the persistence effect for all SCAs.
#         """
#         if not self.params['use_persistence']:
#             return img

#         prev_exposures_filename = get_filename(self.params['out_path'],
#                                 'prev_exp',
#                                 'prev_exp',
#                                 var=str(pointing.sca),
#                                 ftype='pkl',
#                                 overwrite=False)
#         try:
#             with open(prev_exposures_filename, 'rb') as fp:
#                 prev_exposures = pickle.load(fp)
#         except FileNotFoundError:
#             prev_exposures = []

#         if not hasattr(prev_exposures,'__iter__'):
#             raise TypeError("prev_exposures must be a list of Image instances")
#         n_exp = len(prev_exposures)
#         for i in range(n_exp):
#             img._array += galsim.wfirst.wfirst_detectors.fermi_linear(
#             prev_exposures[i].array,
#              (0.5+i)*galsim.wfirst.exptime)*galsim.wfirst.exptime

#         prev_exposures = [img.copy()] + prev_exposures[:]
#         with open(prev_exposures_filename, 'wb') as fw:
#             pickle.dump(prev_exposures, fw)

#         return img

#     def nonlinearity(self,im,NLfunc=wfirst.NLfunc):
#         """
#         Applying a quadratic non-linearity.

#         Note that users who wish to apply some other nonlinearity function (perhaps for other NIR
#         detectors, or for CCDs) can use the more general nonlinearity routine, which uses the
#         following syntax:
#         final_image.applyNonlinearity(NLfunc=NLfunc)
#         with NLfunc being a callable function that specifies how the output image pixel values
#         should relate to the input ones.

#         Input
#         im     : Image
#         NLfunc : Nonlinearity function
#         """

#         # If effect is turned off, return image unchanged
#         if not self.params['use_nonlinearity']:
#             return im

#         # Apply the WFIRST nonlinearity routine, which knows all about the nonlinearity expected in
#         # the WFIRST detectors. Alternately, use a user-provided function.
#         im.applyNonlinearity(NLfunc=NLfunc)

#         # If requested, dump a post-change fits image to disk for diagnostics. Both cumulative and iterative delta.
#         if self.params['save_diff']:
#             diff = im-prev
#             diff.write('nl_a.fits')
#             diff = im-orig
#             diff.write('nl_b.fits')
#             prev = im.copy()

#         return im

#     def interpix_cap(self,im,kernel=wfirst.ipc_kernel):
#         """
#         Including Interpixel capacitance

#         The voltage read at a given pixel location is influenced by the charges present in the
#         neighboring pixel locations due to capacitive coupling of sense nodes. This interpixel
#         capacitance effect is modeled as a linear effect that is described as a convolution of a
#         3x3 kernel with the image. The WFIRST IPC routine knows about the kernel already, so the
#         user does not have to supply it.

#         Input
#         im      : image
#         kernel  : Interpixel capacitance kernel
#         """

#         # If effect is turned off, return image unchanged
#         if not self.params['use_interpix_cap']:
#             return im

#         # Apply interpixel capacitance
#         im.applyIPC(kernel, edge_treatment='extend', fill_value=None)

#         # If requested, dump a post-change fits image to disk for diagnostics. Both cumulative and iterative delta.
#         if self.params['save_diff']:
#             diff = im-prev
#             diff.write('ipc_a.fits')
#             diff = im-orig
#             diff.write('ipc_b.fits')
#             prev = im.copy()

#         return im

#     def add_read_noise(self,im,sigma=wfirst.read_noise):
#         """
#         Adding read noise

#         Read noise is the noise due to the on-chip amplifier that converts the charge into an
#         analog voltage.  We already applied the Poisson noise due to the sky level, so read noise
#         should just be added as Gaussian noise

#         Input
#         im    : image
#         sigma : Variance of read noise
#         """

#         if not self.params['use_read_noise']:
#             return im

#         # Create noise realisation and apply it to image
#         read_noise = galsim.GaussianNoise(self.rng, sigma=sigma)
#         im.addNoise(read_noise)

#         return im

#     def e_to_ADU(self,im):
#         """
#         We divide by the gain to convert from e- to ADU. Currently, the gain value in the WFIRST
#         module is just set to 1, since we don't know what the exact gain will be, although it is
#         expected to be approximately 1. Eventually, this may change when the camera is assembled,
#         and there may be a different value for each SCA. For now, there is just a single number,
#         which is equal to 1.

#         Input
#         im : image
#         """

#         return old_div(im,wfirst.gain)

#     def finalize_sky_im(self,im):
#         """
#         Finalize sky background for subtraction from final image. Add dark current,
#         convert to analog voltage, and quantize.

#         Input
#         im : sky image
#         """

#         if (self.params['sub_true_background'])&(self.params['use_dark_current']):
#             im = (im + round(self.dark_current_))
#         im = self.e_to_ADU(im)
#         im.quantize()

#         return im

#     def finalize_background_subtract(self,im,sky):
#         """
#         Finalize background subtraction of image.

#         Input
#         im : image
#         sky : sky image
#         """

#         # If effect is turned off, return image unchanged
#         if not self.params['use_background']:
#             return im,sky

#         sky.quantize() # Quantize sky
#         sky = self.finalize_sky_im(sky) # Finalize sky with dark current, convert to ADU, and quantize.
#         im -= sky

#         # If requested, dump a final fits image to disk for diagnostics.
#         if self.params['save_diff']:
#             im.write('final_a.fits')

#         return im,sky

# class draw_image(object):
#     """
#     This is where the management of drawing happens (basicaly all the galsim interaction).
#     The general process is that 1) a galaxy model is specified from the truth catalog, 2) rotated, sheared, and convolved with the psf, 3) its drawn into a postage samp, 4) that postage stamp is added to a persistent image of the SCA, 5) the postage stamp is finalized by going through make_image(). Objects within the SCA are iterated using the iterate_*() functions, and the final SCA image (self.im) can be completed with self.finalize_sca().
#     """

#     def __init__(self, params, pointing, modify_image, cats, logger, image_buffer=256, rank=0, comm=None):
#         """
#         Sets up some general properties, including defining the object index lists, starting the generator iterators, assigning the SEDs (single stand-ins for now but generally red to blue for bulg/disk/knots), defining SCA bounds, and creating the empty SCA image.

#         Input
#         params          : parameter dict
#         pointing        : Pointing object
#         modify_image    : modify_image object
#         cats            : init_catalots object
#         logger          : logger instance
#         gal_ind_list    : List of indices from gal truth catalog to attempt to simulate
#         star_ind_list   : List of indices from star truth catalog to attempt to simulate
#         image_buffer    : Number of pixels beyond SCA to attempt simulating objects that may overlap SCA
#         rank            : process rank
#         """

#         self.params       = params
#         self.pointing     = pointing
#         self.modify_image = modify_image
#         self.cats         = cats
#         self.stamp_size   = self.params['stamp_size']
#         self.num_sizes    = self.params['num_sizes']
#         self.gal_iter     = 0
#         self.star_iter    = 0
#         self.supernova_iter = 0
#         self.gal_done     = False
#         self.star_done    = False
#         self.supernova_done = False
#         self.lightcurves = self.cats.lightcurves
#         self.rank         = rank
#         self.rng          = galsim.BaseDeviate(self.params['random_seed'])

#         # Setup galaxy SED
#         # Need to generalize to vary sed based on input catalog
#         if not self.params['dc2']:
#             self.galaxy_sed_b = galsim.SED(self.params['sedpath_E'], wave_type='Ang', flux_type='flambda')
#             self.galaxy_sed_d = galsim.SED(self.params['sedpath_Scd'], wave_type='Ang', flux_type='flambda')
#             self.galaxy_sed_n = galsim.SED(self.params['sedpath_Im'],  wave_type='Ang', flux_type='flambda')
#             # Setup star SED
#             self.star_sed     = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')
#             self.supernova_sed = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')

#         # Galsim bounds object to specify area to simulate objects that might overlap the SCA
#         self.b0  = galsim.BoundsI(  xmin=1-old_div(int(image_buffer),2),
#                                     ymin=1-old_div(int(image_buffer),2),
#                                     xmax=wfirst.n_pix+old_div(int(image_buffer),2),
#                                     ymax=wfirst.n_pix+old_div(int(image_buffer),2))
#         # Galsim bounds object to specify area to simulate objects that would have centroids that fall on the SCA to save as postage stamps (pixels not on the SCA have weight=0)
#         self.b   = galsim.BoundsI(  xmin=1,
#                                     ymin=1,
#                                     xmax=wfirst.n_pix,
#                                     ymax=wfirst.n_pix)

#         # SCA image (empty right now)
#         if self.params['draw_sca']:
#             self.im = galsim.Image(self.b, wcs=self.pointing.WCS)
#         else:
#             self.im = None

#         # Get sky background for pointing
#         self.sky_level = wfirst.getSkyLevel(self.pointing.bpass,
#                                             world_pos=self.pointing.WCS.toWorld(
#                                                         galsim.PositionI(old_div(wfirst.n_pix,2),
#                                                                         old_div(wfirst.n_pix,2))),
#                                             date=self.pointing.date)
#         self.sky_level *= (1.0 + wfirst.stray_light_fraction)*wfirst.pixel_scale**2 # adds stray light and converts to photons/cm^2
#         self.sky_level *= self.stamp_size*self.stamp_size # Converts to photons, but uses smallest stamp size to do so - not optimal

#         if self.params['dc2']:
#             self.ax={}
#             self.bx={}
#             wavelen = np.arange(3000.,11500.+1.,1., dtype='float')
#             sb = np.zeros(len(wavelen), dtype='float')
#             sb[abs(wavelen-5000.)<1./2.] = 1.
#             self.imsim_bpass = galsim.Bandpass(galsim.LookupTable(x=wavelen,f=sb,interpolant='nearest'),'a',blue_limit=3000., red_limit=11500.).withZeropoint('AB')

#     def iterate_gal(self):
#         """
#         Iterator function to loop over all possible galaxies to draw
#         """

#         # Check if the end of the galaxy list has been reached; return exit flag (gal_done) True
#         # You'll have a bad day if you aren't checking for this flag in any external loop...
#         # self.gal_done = True
#         # return
#         if self.gal_iter == self.cats.get_gal_length():
#             self.gal_done = True
#             print('Proc '+str(self.rank)+' done with galaxies.')
#             return

#         # Reset galaxy information
#         self.gal_model = None
#         self.gal_stamp = None

#         # if self.gal_iter>1000:
#         #     self.gal_done = True
#         #     return


#         if self.gal_iter%10==0:
#             print('Progress '+str(self.rank)+': Attempting to simulate galaxy '+str(self.gal_iter)+' in SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.')

#         # Galaxy truth index and array for this galaxy
#         self.ind,self.gal = self.cats.get_gal(self.gal_iter)
#         self.gal_iter    += 1
#         self.rng          = galsim.BaseDeviate(self.params['random_seed']+self.ind+self.pointing.dither)

#         # if self.ind != 157733:
#         #     return

#         # if self.ind != 144078:
#         #     return

#         # If galaxy image position (from wcs) doesn't fall within simulate-able bounds, skip (slower)
#         # If it does, draw it
#         if self.check_position(self.gal['ra'],self.gal['dec']):
#             # print('iterate',self.gal_iter,time.time()-t0)
#             # print(process.memory_info().rss/2**30)
#             # print(process.memory_info().vms/2**30)
#             self.draw_galaxy()

#     def iterate_star(self):
#         """
#         Iterator function to loop over all possible stars to draw
#         """

#         # self.star_done = True
#         # return
#         # Don't draw stars into postage stamps
#         if not self.params['draw_sca']:
#             self.star_done = True
#             print('Proc '+str(self.rank)+' done with stars.')
#             return
#         if not self.params['draw_stars']:
#             self.star_done = True
#             print('Proc '+str(self.rank)+' not doing stars.')
#             return
#         # Check if the end of the star list has been reached; return exit flag (gal_done) True
#         # You'll have a bad day if you aren't checking for this flag in any external loop...
#         if self.star_iter == self.cats.get_star_length():
#             self.star_done = True
#             return

#         # Not participating in star parallelisation
#         if self.rank == -1:
#             self.star_done = True
#             return

#         # if self.star_iter%10==0:
#         print('Progress '+str(self.rank)+': Attempting to simulate star '+str(self.star_iter)+' in SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.')

#         # Star truth index for this galaxy
#         self.ind,self.star = self.cats.get_star(self.star_iter)
#         self.star_iter    += 1
#         self.rng        = galsim.BaseDeviate(self.params['random_seed']+self.ind+self.pointing.dither)

#         # If star image position (from wcs) doesn't fall within simulate-able bounds, skip (slower)
#         # If it does, draw it
#         if self.check_position(self.star['ra'],self.star['dec']):
#             self.draw_star()

#     def iterate_supernova(self):
#         if not self.params['draw_sca']:
#             self.supernova_done = True
#             print('Proc '+str(self.rank)+' done with supernova.')
#             return 
#         if not self.params['draw_supernova']:
#             self.supernova_done = True
#             print('Proc '+str(self.rank)+' not doing stars.')
#             return             
#         # Check if the end of the supernova list has been reached; return exit flag (supernova_done) True
#         # You'll have a bad day if you aren't checking for this flag in any external loop...
#         if self.supernova_iter == self.cats.get_supernova_length():
#             self.supernova_done = True
#             return 

#         # Not participating in star parallelisation
#         if self.rank == -1:
#             self.supernova_done = True
#             return 
            
#         self.supernova_stamp = None
#         # if self.star_iter%10==0:
#         #     print 'Progress '+str(self.rank)+': Attempting to simulate star '+str(self.star_iter)+' in SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.'
        
#         #Host galaxy variable
#         self.hostid = None
        
#         # Supernova truth index for this supernova
#         self.ind,self.supernova = self.cats.get_supernova(self.supernova_iter)
#         self.supernova_iter    += 1
#         self.rng        = galsim.BaseDeviate(self.params['random_seed']+self.ind+self.pointing.dither)

#         # If supernova image position (from wcs) doesn't fall within simulate-able bounds, skip (slower) 
#         # If it does, draw it
#         if self.check_position(self.supernova['ra'],self.supernova['dec']):
#             self.draw_supernova()

#     def check_position(self, ra, dec):
#         """
#         Create the world and image position galsim objects for obj, as well as the local WCS. Return whether object is in SCA (+half-stamp-width border).

#         Input
#         ra  : RA of object
#         dec : Dec of object
#         """

#         # Galsim world coordinate object (ra,dec)
#         self.radec = galsim.CelestialCoord(ra*galsim.radians, dec*galsim.radians)

#         # Galsim image coordinate object
#         self.xy = self.pointing.WCS.toImage(self.radec)

#         # Galsim integer image coordinate object
#         self.xyI = galsim.PositionI(int(self.xy.x),int(self.xy.y))

#         # Galsim image coordinate object holding offset from integer pixel grid
#         # troxel needs to change this
#         self.offset = self.xy-self.xyI

#         # Define the local_wcs at this world position
#         self.local_wcs = self.pointing.WCS.local(self.xy)

#         # Return whether object is in SCA (+half-stamp-width border)
#         return self.b0.includes(self.xyI)

#     def make_sed_model(self, model, sed):
#         """
#         Modifies input SED to be at appropriate redshift and magnitude, then applies it to the object model.

#         Input
#         model : Galsim object model
#         sed   : Template SED for object
#         """

#         # Apply correct flux from magnitude for filter bandpass
#         sed_ = sed.atRedshift(self.gal['z'])
#         sed_ = sed_.withMagnitude(self.gal[self.pointing.filter], self.pointing.bpass)

#         # Return model with SED applied
#         return model * sed_

#     def make_sed_model_dc2(self, model, obj, i):
#         """
#         Modifies input SED to be at appropriate redshift and magnitude, deals with dust model, then applies it to the object model.

#         Input
#         model : Galsim object model
#         i     : component index to extract truth params
#         """

#         if i==-1:
#             sedname = obj['sed'].lstrip().rstrip()
#             magnorm = obj['mag_norm']
#             Av = obj['A_v']
#             Rv = obj['R_v']
#         else:
#             if i==2:
#                 sedname = obj['sed'][1].lstrip().rstrip()
#             else:
#                 sedname = obj['sed'][i].lstrip().rstrip()
#             magnorm = obj['mag_norm'][i]
#             Av = obj['A_v'][i]
#             Rv = obj['R_v'][i]

#         sed = self.cats.seds[sedname]
#         sed_lut = galsim.LookupTable(x=sed[:,0],f=sed[:,1])
#         sed = galsim.SED(sed_lut, wave_type='nm', flux_type='flambda',redshift=0.)
#         sed_ = sed.withMagnitude(magnorm, self.imsim_bpass) # apply mag
#         if len(sed_.wave_list) not in self.ax:
#             ax,bx = setupCCM_ab(sed_.wave_list)
#             self.ax[len(sed_.wave_list)] = ax
#             self.bx[len(sed_.wave_list)] = bx
#         dust = addDust(self.ax[len(sed_.wave_list)], self.bx[len(sed_.wave_list)], A_v=Av, R_v=Rv)
#         sed_ = sed_._mul_scalar(dust) # Add dust extinction. Same function from lsst code for testing right now
#         sed_ = sed_.atRedshift(obj['z']) # redshift

#         # Return model with SED applied
#         return model * sed_

#     def galaxy_model(self):
#         """
#         Generate the intrinsic galaxy model based on truth catalog parameters
#         """


#         if self.params['dc2']:

#             # loop over components, order bulge, disk, knots
#             for i in range(3):
#                 if self.gal['size'][i] == 0:
#                     continue
#                 # If any flux, build component and apply appropriate SED
#                 if i<2:
#                     if i==0:
#                         n=4
#                     else:
#                         n=1
#                     component = galsim.Sersic(n, half_light_radius=1.*self.gal['size'][i], flux=1., trunc=10.*self.gal['size'][i])
#                 else:
#                     rng   = galsim.BaseDeviate((int(self.gal['gind'])<<10)+127) #using orig phosim unique id as random seed, which requires bit appending 127 to represent knots model
#                     component = galsim.RandomKnots(npoints=self.gal['knots'], half_light_radius=1.*self.gal['size'][i], flux=1., rng=rng)
#                 # Apply intrinsic ellipticity to the component.
#                 s         = galsim.Shear(q=1./self.gal['q'][i], beta=(90.+self.gal['pa'][i])*galsim.degrees)
#                 s         = galsim._Shear(complex(s.g1,-s.g2)) # Fix -g2
#                 component = component.shear(s)
#                 # Apply the SED
#                 component = self.make_sed_model_dc2(component, self.gal, i)

#                 if self.gal_model is None:
#                     # No model, so save the galaxy model as the component
#                     self.gal_model = component
#                 else:
#                     # Existing model, so save the galaxy model as the sum of the components
#                     self.gal_model = galsim.Add([self.gal_model, component])

#         else:

#             # Generate galaxy model
#             # Calculate flux fraction of disk portion 
#             flux = (1.-self.gal['bflux']) * self.gal['dflux']
#             if flux > 0:
#                 # If any flux, build Sersic disk galaxy (exponential) and apply appropriate SED
#                 self.gal_model = galsim.Sersic(1, half_light_radius=1.*self.gal['size'], flux=flux, trunc=10.*self.gal['size'])
#                 self.gal_model = self.make_sed_model(self.gal_model, self.galaxy_sed_d)
#                 # self.gal_model = self.gal_model.withScaledFlux(flux)

#             # Calculate flux fraction of knots portion 
#             flux = (1.-self.gal['bflux']) * (1.-self.gal['dflux'])
#             if flux > 0:
#                 # If any flux, build star forming knots model and apply appropriate SED
#                 rng   = galsim.BaseDeviate(self.params['random_seed']+self.ind)
#                 knots = galsim.RandomKnots(npoints=self.params['knots'], half_light_radius=1.*self.gal['size'], flux=flux, rng=rng) 
#                 knots = self.make_sed_model(galsim.ChromaticObject(knots), self.galaxy_sed_n)
#                 # knots = knots.withScaledFlux(flux)
#                 # Sum the disk and knots, then apply intrinsic ellipticity to the disk+knot component. Fixed intrinsic shape, but can be made variable later.
#                 self.gal_model = galsim.Add([self.gal_model, knots])
#                 self.gal_model = self.gal_model.shear(e1=self.gal['int_e1'], e2=self.gal['int_e2'])

#             # Calculate flux fraction of bulge portion 
#             flux = self.gal['bflux']
#             if flux > 0:
#                 # If any flux, build Sersic bulge galaxy (de vacaleurs) and apply appropriate SED
#                 bulge = galsim.Sersic(4, half_light_radius=1.*self.gal['size'], flux=flux, trunc=10.*self.gal['size']) 
#                 # Apply intrinsic ellipticity to the bulge component. Fixed intrinsic shape, but can be made variable later.
#                 bulge = bulge.shear(e1=self.gal['int_e1'], e2=self.gal['int_e2'])
#                 # Apply the SED
#                 bulge = self.make_sed_model(bulge, self.galaxy_sed_b)
#                 # bulge = bulge.withScaledFlux(flux)

#                 if self.gal_model is None:
#                     # No disk or knot component, so save the galaxy model as the bulge part
#                     self.gal_model = bulge
#                 else:
#                     # Disk/knot component, so save the galaxy model as the sum of two parts
#                     self.gal_model = galsim.Add([self.gal_model, bulge])


#     def galaxy(self):
#         """
#         Call galaxy_model() to get the intrinsic galaxy model, then apply properties relevant to its observation
#         """

#         # Build intrinsic galaxy model
#         self.galaxy_model()


#         # print('model1',time.time()-t0)
#         # print(process.memory_info().rss/2**30)
#         # print(process.memory_info().vms/2**30)

#         if self.params['dc2']:
#             g1 = self.gal['g1']/(1. - self.gal['k'])
#             g2 = -self.gal['g2']/(1. - self.gal['k'])
#             mu = 1./((1. - self.gal['k'])**2 - (self.gal['g1']**2 + self.gal['g2']**2))
#             # Apply a shear
#             self.gal_model = self.gal_model.lens(g1=g1,g2=g2,mu=mu)
#             # Rescale flux appropriately for wfirst
#             self.gal_model = self.gal_model * galsim.wfirst.collecting_area * galsim.wfirst.exptime
#         else:
#             # Random rotation (pairs of objects are offset by pi/2 to cancel shape noise)
#             self.gal_model = self.gal_model.rotate(self.gal['rot']*galsim.radians) 
#             # Apply a shear
#             self.gal_model = self.gal_model.shear(g1=self.gal['g1'],g2=self.gal['g2'])
#             # Rescale flux appropriately for wfirst
#             self.gal_model = self.gal_model * galsim.wfirst.collecting_area * galsim.wfirst.exptime

#         # Ignoring chromatic stuff for now for speed, so save correct flux of object
#         flux = self.gal_model.calculateFlux(self.pointing.bpass)
#         self.mag = self.gal_model.calculateMagnitude(self.pointing.bpass)
#         # print(flux,self.mag)
#         # print 'galaxy flux',flux
#         # Evaluate the model at the effective wavelength of this filter bandpass (should change to effective SED*bandpass?)
#         # This makes the object achromatic, which speeds up drawing and convolution
#         self.gal_model  = self.gal_model.evaluateAtWavelength(self.pointing.bpass.effective_wavelength)
#         # Reassign correct flux
#         self.gal_model  = self.gal_model.withFlux(flux) # reapply correct flux

#         if old_div(self.sky_level,flux) < galsim.GSParams().folding_threshold:
#             gsparams = galsim.GSParams( folding_threshold=old_div(self.sky_level,flux),
#                                         maximum_fft_size=16384 )
#         else:
#             gsparams = galsim.GSParams( maximum_fft_size=16384 )
#         gsparams = galsim.GSParams( maximum_fft_size=16384 )

#         # Convolve with PSF
#         self.gal_model = galsim.Convolve(self.gal_model.withGSParams(gsparams), self.pointing.load_psf(self.xyI), propagate_gsparams=False)

#         # Convolve with additional los motion (jitter), if any
#         if self.pointing.los_motion is not None:
#             self.gal_model = galsim.Convolve(self.gal_model, self.pointing.los_motion)

#         # chromatic stuff replaced by above lines
#         # # Draw galaxy igal into stamp.
#         # self.gal_list[igal].drawImage(self.pointing.bpass[self.params['filter']], image=gal_stamp)
#         # # Add detector effects to stamp.


#     def get_stamp_size_factor(self,obj,factor=5):
#         """
#         Select the stamp size multiple to use.

#         Input
#         obj    : Galsim object
#         factor : Factor to multiple suggested galsim stamp size by
#         """

#         return old_div(int(obj.getGoodImageSize(wfirst.pixel_scale)), self.stamp_size)
#         # return 2*np.ceil(1.*np.ceil(self.gal['size']/(np.sqrt(2*np.log(2)))*1.25)/self.stamp_size)

#     def draw_galaxy(self):
#         """
#         Draw the galaxy model into the SCA (neighbors and blending) and/or the postage stamp (isolated).
#         """

#         self.gal_stamp_too_large = False

#         # Build galaxy model that will be drawn into images
#         self.galaxy()

#         # print('draw_galaxy1',time.time()-t0)
#         # print(process.memory_info().rss/2**30)
#         # print(process.memory_info().vms/2**30)


#         stamp_size_factor = self.get_stamp_size_factor(self.gal_model)

#         # # Skip drawing some really huge objects (>twice the largest stamp size)
#         # if stamp_size_factor>2.*self.num_sizes:
#         #     return

#         # Create postage stamp bounds at position of object
#         b = galsim.BoundsI( xmin=self.xyI.x-old_div(int(stamp_size_factor*self.stamp_size),2)+1,
#                             ymin=self.xyI.y-old_div(int(stamp_size_factor*self.stamp_size),2)+1,
#                             xmax=self.xyI.x+old_div(int(stamp_size_factor*self.stamp_size),2),
#                             ymax=self.xyI.y+old_div(int(stamp_size_factor*self.stamp_size),2))

#         # If this postage stamp doesn't overlap the SCA bounds at all, no reason to draw anything
#         if not (b&self.b).isDefined():
#             return

#         # Create postage stamp for galaxy
#         gal_stamp = galsim.Image(b, wcs=self.pointing.WCS)

#         # print('draw_galaxy2',time.time()-t0)
#         # print(process.memory_info().rss/2**30)
#         # print(process.memory_info().vms/2**30)

#         # Draw galaxy model into postage stamp. This is the basis for both the postage stamp output and what gets added to the SCA image. This will obviously create biases if the postage stamp is too small - need to monitor that.
#         self.gal_model.drawImage(image=gal_stamp,offset=self.xy-b.true_center,method='phot',rng=self.rng)
#         # gal_stamp.write(str(self.ind)+'.fits')

#         # print('draw_galaxy3',time.time()-t0)
#         # print(process.memory_info().rss/2**30)
#         # print(process.memory_info().vms/2**30)

#         # Add galaxy stamp to SCA image
#         if self.params['draw_sca']:
#             self.im[b&self.b] = self.im[b&self.b] + gal_stamp[b&self.b]

#         # If object too big for stamp sizes, or not saving stamps, skip saving a stamp
#         if stamp_size_factor>=self.num_sizes:
#             print('too big stamp',stamp_size_factor,stamp_size_factor*self.stamp_size)
#             self.gal_stamp_too_large = True
#             return
#         if 'no_stamps' in self.params:
#             if self.params['no_stamps']:
#                 self.gal_stamp_too_large = True
#                 self.gal_stamp = -1
#                 return

#         # print('draw_galaxy4',time.time()-t0)
#         # print(process.memory_info().rss/2**30)
#         # print(process.memory_info().vms/2**30)

#         # Check if galaxy center falls on SCA
#         # Apply background, noise, and WFIRST detector effects
#         # Get final galaxy stamp and weight map
#         if self.b.includes(self.xyI):
#             gal_stamp, weight = self.modify_image.add_effects(gal_stamp[b&self.b],self.pointing,self.radec,self.pointing.WCS,self.rng,phot=True)

#             # Copy part of postage stamp that falls on SCA - set weight map to zero for parts outside SCA
#             self.gal_stamp = galsim.Image(b, wcs=self.pointing.WCS)
#             self.gal_stamp[b&self.b] = self.gal_stamp[b&self.b] + gal_stamp[b&self.b]
#             self.weight_stamp = galsim.Image(b, wcs=self.pointing.WCS)
#             if weight != None:
#                 self.weight_stamp[b&self.b] = self.weight_stamp[b&self.b] + weight[b&self.b]

#             # If we're saving the true PSF model, simulate an appropriate unit-flux star and draw it (oversampled) at the position of the galaxy
#             if (self.params['draw_true_psf']) and (not self.params['skip_stamps']):
#                 # self.star_model() #Star model for PSF (unit flux)
#                 # Create modified WCS jacobian for super-sampled pixelisation
#                 wcs = galsim.JacobianWCS(dudx=old_div(self.local_wcs.dudx,self.params['oversample']),
#                                          dudy=old_div(self.local_wcs.dudy,self.params['oversample']),
#                                          dvdx=old_div(self.local_wcs.dvdx,self.params['oversample']),
#                                          dvdy=old_div(self.local_wcs.dvdy,self.params['oversample']))
#                 # Create postage stamp bounds at position of object
#                 b_psf = galsim.BoundsI( xmin=self.xyI.x-old_div(int(self.params['psf_stampsize']),2)+1,
#                                     ymin=self.xyI.y-old_div(int(self.params['psf_stampsize']),2)+1,
#                                     xmax=self.xyI.x+old_div(int(self.params['psf_stampsize']),2),
#                                     ymax=self.xyI.y+old_div(int(self.params['psf_stampsize']),2))
#                 # Create postage stamp bounds at position of object
#                 b_psf2 = galsim.BoundsI( xmin=self.xyI.x-old_div(int(self.params['psf_stampsize']*self.params['oversample']),2)+1,
#                                     ymin=self.xyI.y-old_div(int(self.params['psf_stampsize']*self.params['oversample']),2)+1,
#                                     xmax=self.xyI.x+old_div(int(self.params['psf_stampsize']*self.params['oversample']),2),
#                                     ymax=self.xyI.y+old_div(int(self.params['psf_stampsize']*self.params['oversample']),2))
#                 # Create psf stamp with oversampled pixelisation
#                 self.psf_stamp = galsim.Image(b_psf, wcs=self.pointing.WCS)
#                 # print('draw_galaxy5',time.time()-t0)
#                 # print(process.memory_info().rss/2**30)
#                 # print(process.memory_info().vms/2**30)
#                 self.psf_stamp2 = galsim.Image(b_psf2, wcs=wcs)
#                 # Draw PSF into postage stamp
#                 self.st_model.drawImage(image=self.psf_stamp,wcs=self.pointing.WCS)
#                 self.st_model.drawImage(image=self.psf_stamp2,wcs=wcs,method='no_pixel')
#             # print('draw_galaxy6',time.time()-t0)
#             # print(process.memory_info().rss/2**30)
#             # print(process.memory_info().vms/2**30)

#     def star_model(self, sed = None, mag = 0.):
#         """
#         Create star model for PSF or for drawing stars into SCA

#         Input
#         sed  : The stellar SED
#         mag  : The magnitude of the star
#         """

#         # Generate star model (just a delta function) and apply SED
#         if sed is not None:
#             if self.params['dc2']:
#                 self.st_model = galsim.DeltaFunction()
#                 self.st_model = self.make_sed_model_dc2(self.st_model, self.star, -1)
#                 self.st_model = self.st_model * wfirst.collecting_area * wfirst.exptime
#             else:
#                 sed_ = sed.withMagnitude(mag, self.pointing.bpass)
#                 self.st_model = galsim.DeltaFunction() * sed_  * wfirst.collecting_area * wfirst.exptime
#             flux = self.st_model.calculateFlux(self.pointing.bpass)
#             mag = self.st_model.calculateMagnitude(self.pointing.bpass)
#             ft = old_div(self.sky_level,flux)
#             # print mag,flux,ft
#             # if ft<0.0005:
#             #     ft = 0.0005
#             if ft < galsim.GSParams().folding_threshold:
#                 gsparams = galsim.GSParams( folding_threshold=old_div(self.sky_level,flux),
#                                             maximum_fft_size=16384 )
#             else:
#                 gsparams = galsim.GSParams( maximum_fft_size=16384 )
#         else:
#             self.st_model = galsim.DeltaFunction(flux=1.)
#             flux = 1.
#             gsparams = galsim.GSParams( maximum_fft_size=16384 )

#         # Evaluate the model at the effective wavelength of this filter bandpass (should change to effective SED*bandpass?)
#         # This makes the object achromatic, which speeds up drawing and convolution
#         self.st_model = self.st_model.evaluateAtWavelength(self.pointing.bpass.effective_wavelength)
#         # Reassign correct flux
#         self.st_model  = self.st_model.withFlux(flux) # reapply correct flux

#         # Convolve with PSF
#         if mag<15:
#             psf = self.pointing.load_psf(self.xyI,star=True)
#             psf = psf.withGSParams(galsim.GSParams(folding_threshold=1e-3))
#             self.st_model = galsim.Convolve(self.st_model, psf)
#         else:
#             psf = self.pointing.load_psf(self.xyI,star=True)
#             self.st_model = galsim.Convolve(self.st_model, psf)

#         # Convolve with additional los motion (jitter), if any
#         if self.pointing.los_motion is not None:
#             self.st_model = galsim.Convolve(self.st_model, self.pointing.los_motion)

#         # old chromatic version
#         # self.psf_list[igal].drawImage(self.pointing.bpass[self.params['filter']],image=psf_stamp, wcs=local_wcs)

#         return mag

#     def draw_star(self):
#         """
#         Draw a star into the SCA
#         """

#         # Get star model with given SED and flux
#         if self.params['dc2']:
#             mag = self.star_model(sed=self.star['sed'].lstrip().rstrip())
#         else:
#             mag = self.star_model(sed=self.star_sed,mag=self.star[self.pointing.filter])

#         # Get good stamp size multiple for star
#         # stamp_size_factor = self.get_stamp_size_factor(self.st_model)#.withGSParams(gsparams))
#         stamp_size_factor = 50

#         # Create postage stamp bounds for star
#         # b = galsim.BoundsI( xmin=self.xyI.x-int(stamp_size_factor*self.stamp_size)/2,
#         #                     ymin=self.xyI.y-int(stamp_size_factor*self.stamp_size)/2,
#         #                     xmax=self.xyI.x+int(stamp_size_factor*self.stamp_size)/2,
#         #                     ymax=self.xyI.y+int(stamp_size_factor*self.stamp_size)/2 )
#         b = galsim.BoundsI( xmin=self.xyI.x-old_div(int(stamp_size_factor*self.stamp_size),2),
#                             ymin=self.xyI.y-old_div(int(stamp_size_factor*self.stamp_size),2),
#                             xmax=self.xyI.x+old_div(int(stamp_size_factor*self.stamp_size),2),
#                             ymax=self.xyI.y+old_div(int(stamp_size_factor*self.stamp_size),2) )

#         # If postage stamp doesn't overlap with SCA, don't draw anything
#         if not (b&self.b).isDefined():
#             return

#         # Create star postage stamp
#         star_stamp = galsim.Image(b, wcs=self.pointing.WCS)

#         # print(self.star[self.pointing.filter],repr(self.st_model))
#         # Draw star model into postage stamp
#         if mag<12:
#             self.st_model.drawImage(image=star_stamp,offset=self.xy-b.true_center)
#         else:
#             self.st_model.drawImage(image=star_stamp,offset=self.xy-b.true_center,method='phot',rng=self.rng,maxN=1000000)

#         # star_stamp.write('/fs/scratch/cond0083/wfirst_sim_out/images/'+str(self.ind)+'.fits.gz')

#         # Add star stamp to SCA image
#         self.im[b&self.b] += star_stamp[b&self.b]
#         # self.st_model.drawImage(image=self.im,add_to_image=True,offset=self.xy-self.im.true_center,method='phot',rng=self.rng,maxN=1000000)

#         if self.b.includes(self.xyI):
#             self.supernova_stamp = star_stamp        

#     def draw_supernova(self):
        
#         # Get star model with given SED and flux
#         index = self.supernova['ptrobs_min'] - 1
#         current_filter = self.lightcurves['flt'][index]
#         filt_index = 0
#         no_of_filters = 0
#         filters = []
#         while current_filter not in filters:
#             if current_filter == self.pointing.filter[0]:
#                 filt_index = index
#             filters.append(current_filter)
#             no_of_filters += 1
#             index += 1
#             current_filter = self.lightcurves['flt'][index]
#         current_date = self.lightcurves['mjd'][filt_index]
#         while current_date <= self.pointing.mjd and filt_index < self.supernova['ptrobs_max']:
#             filt_index += no_of_filters
#             current_date = self.lightcurves['mjd'][filt_index]
#         flux1 = 10 ** ((27.5 - self.lightcurves['sim_magobs'][filt_index - no_of_filters]) / 2.512)
#         flux2 = 10 ** ((27.5 - self.lightcurves['sim_magobs'][filt_index]) / 2.512)
#         flux = np.interp(self.pointing.mjd, [self.lightcurves['mjd'][filt_index - no_of_filters], current_date], [flux1, flux2])
#         if flux <= 0.0:
#             magnitude = 100
#         else:
#             magnitude = 27.5 - (2.512 * math.log10(flux))
#         self.ind = self.supernova['snid']
#         self.mag = magnitude
#         self.hostid = self.supernova['hostgal_objid']
            
#         gsparams = self.star_model(sed=self.supernova_sed,mag=magnitude)

#         # Get good stamp size multiple for supernova
#         # stamp_size_factor = self.get_stamp_size_factor(self.st_model)#.withGSParams(gsparams))
#         stamp_size_factor = 40

#         # Create postage stamp bounds for supernova
#         # b = galsim.BoundsI( xmin=self.xyI.x-int(stamp_size_factor*self.stamp_size)/2,
#         #                     ymin=self.xyI.y-int(stamp_size_factor*self.stamp_size)/2,
#         #                     xmax=self.xyI.x+int(stamp_size_factor*self.stamp_size)/2,
#         #                     ymax=self.xyI.y+int(stamp_size_factor*self.stamp_size)/2 )
#         b = galsim.BoundsI( xmin=self.xyI.x-old_div(int(stamp_size_factor*self.stamp_size),2),
#                             ymin=self.xyI.y-old_div(int(stamp_size_factor*self.stamp_size),2),
#                             xmax=self.xyI.x+old_div(int(stamp_size_factor*self.stamp_size),2),
#                             ymax=self.xyI.y+old_div(int(stamp_size_factor*self.stamp_size),2) )

#         # If postage stamp doesn't overlap with SCA, don't draw anything
#         if not (b&self.b).isDefined():
#             return

#         # Create star postage stamp
#         star_stamp = galsim.Image(b, wcs=self.pointing.WCS)

#         # Draw star model into postage stamp
#         self.st_model.drawImage(image=star_stamp,offset=self.offset,method='phot',rng=self.rng,maxN=1000000)

#         # star_stamp.write('/fs/scratch/cond0083/wfirst_sim_out/images/'+str(self.ind)+'.fits.gz')

#         # Add star stamp to SCA image
#         self.im[b&self.b] = self.im[b&self.b] + star_stamp[b&self.b]
#         # self.st_model.drawImage(image=self.im,add_to_image=True,offset=self.xy-self.im.true_center,method='phot',rng=self.rng,maxN=1000000)

#         if self.b.includes(self.xyI):
#             self.supernova_stamp = star_stamp

#     def retrieve_stamp(self):
#         """
#         Helper function to accumulate various information about a postage stamp and return it in dictionary form.
#         """

#         if self.gal_stamp is None:
#             return None

#         if self.gal_stamp_too_large:
#             # stamp size too big
#             return {'ind'    : self.ind, # truth index
#                     'ra'     : self.gal['ra'], # ra of galaxy
#                     'dec'    : self.gal['dec'], # dec of galaxy
#                     'x'      : self.xy.x, # SCA x position of galaxy
#                     'y'      : self.xy.y, # SCA y position of galaxy
#                     'dither' : self.pointing.dither, # dither index
#                     'mag'    : self.mag, #Calculated magnitude
#                     'stamp'  : self.get_stamp_size_factor(self.gal_model)*self.stamp_size, # Get stamp size in pixels
#                     'gal'    : None, # Galaxy image object (includes metadata like WCS)
#                     'psf'    : None, # Flattened array of PSF image
#                     # 'psf2'    : None, # Flattened array of PSF image
#                     'weight' : None } # Flattened array of weight map

#         return {'ind'    : self.ind, # truth index
#                 'ra'     : self.gal['ra'], # ra of galaxy
#                 'dec'    : self.gal['dec'], # dec of galaxy
#                 'x'      : self.xy.x, # SCA x position of galaxy
#                 'y'      : self.xy.y, # SCA y position of galaxy
#                 'dither' : self.pointing.dither, # dither index
#                 'mag'    : self.mag, #Calculated magnitude
#                 'stamp'  : self.get_stamp_size_factor(self.gal_model)*self.stamp_size, # Get stamp size in pixels
#                 'gal'    : self.gal_stamp, # Galaxy image object (includes metadata like WCS)
#                 # 'psf'    : self.psf_stamp.array.flatten(), # Flattened array of PSF image
#                 # 'psf2'   : self.psf_stamp2.array.flatten(), # Flattened array of PSF image
#                 'weight' : self.weight_stamp.array.flatten() } # Flattened array of weight map

#     def retrieve_star_stamp(self):
    
#         if self.star_stamp is None:
#             return None
        
#         return {'ind'    : self.ind, # truth index
#                 'ra'     : self.star['ra'], # ra of galaxy
#                 'dec'    : self.star['dec'], # dec of galaxy
#                 'x'      : self.xy.x, # SCA x position of galaxy
#                 'y'      : self.xy.y, # SCA y position of galaxy
#                 'dither' : self.pointing.dither, # dither index
#                 'mag'    : self.mag } #Calculated magnitude
    
#     def retrieve_supernova_stamp(self):
        
#         if self.supernova_stamp is None:
#             return None
        
#         return {'ind'    : self.ind, # truth index
#                 'ra'     : self.supernova['ra'], # ra of supernova
#                 'dec'    : self.supernova['dec'], # dec of supernova
#                 'x'      : self.xy.x, # SCA x position of supernova
#                 'y'      : self.xy.y, # SCA y position of supernova
#                 'dither' : self.pointing.dither, # dither index
#                 'mag'    : self.mag, #Calculated magnitude
#                 'hostid' : self.hostid, #Host galaxy id number
#                 'supernova'    : self.supernova_stamp } # Supernova image object (includes metadata like WCS)

#     def finalize_sca(self):
#         """
#         # Apply background, noise, and WFIRST detector effects to SCA image
#         # Get final SCA image and weight map
#         """

#         # World coordinate of SCA center
#         radec = self.pointing.WCS.toWorld(galsim.PositionI(old_div(wfirst.n_pix,2),old_div(wfirst.n_pix,2)))
#         # Apply background, noise, and WFIRST detector effects to SCA image and return final SCA image and weight map
#         return self.modify_image.add_effects(self.im,self.pointing,radec,self.pointing.WCS,self.rng,phot=True, ps_save=True)[0]

# class accumulate_output_disk(object):

#     def __init__(self, param_file, filter_, pix, comm, ignore_missing_files = False, setup = False,condor_build=False, shape=False, shape_iter = None, shape_cnt = None):

#         self.params     = yaml.load(open(param_file))
#         self.param_file = param_file
#         # Do some parsing
#         for key in list(self.params.keys()):
#             if self.params[key]=='None':
#                 self.params[key]=None
#             if self.params[key]=='none':
#                 self.params[key]=None
#             if self.params[key]=='True':
#                 self.params[key]=True
#             if self.params[key]=='False':
#                 self.params[key]=False
#         self.ditherfile = self.params['dither_file']
#         logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
#         self.logger = logging.getLogger('wfirst_sim')
#         self.pointing   = pointing(self.params,self.logger,filter_=filter_,sca=None,dither=None)
#         self.pix = pix
#         self.skip = False

#         self.comm = comm
#         status = MPI.Status()
#         if self.comm is None:
#             self.rank = 0
#             self.size = 1
#         else:
#             self.rank = self.comm.Get_rank()
#             self.size = self.comm.Get_size()

#         self.shape_iter = shape_iter
#         if self.shape_iter is None:
#             self.shape_iter = 0
#         self.shape_cnt  = shape_cnt
#         if self.shape_cnt is None:
#             self.shape_cnt = 1

#         print('mpi check',self.rank,self.size)
#         if not condor:
#             os.chdir(os.environ['TMPDIR'].replace('[','[').replace(']',']'))

#         if shape:
#             self.file_exists = True
#             if not condor:
#                 raise ParamError('Not intended to work outside condor.')
#             if ('output_meds' not in self.params) or ('psf_meds' not in self.params):
#                 raise ParamError('Must define both output_meds and psf_meds in yaml')
#             if (self.params['output_meds'] is None) or (self.params['psf_meds'] is None):
#                 raise ParamError('Must define both output_meds and psf_meds in yaml')
#             print('shape',self.shape_iter,self.shape_cnt)
#             self.load_index()
#             self.local_meds = get_filename('./',
#                     '',
#                     self.params['output_meds'],
#                     var=self.pointing.filter+'_'+str(self.pix),
#                     ftype='fits',
#                     overwrite=False)
#             self.local_meds_psf = get_filename('./',
#                     '',
#                     self.params['psf_meds'],
#                     var=self.pointing.filter+'_'+str(self.pix),
#                     ftype='fits',
#                     overwrite=False)

#             os.system( 'gunzip '+self.local_meds+'.gz')

#             if self.local_meds != self.local_meds_psf:
#                 os.system( 'gunzip '+self.local_meds_psf+'.gz')

#             return
#         else:
#             self.file_exists = False

#         if (not setup)&(not condor_build):
#             if self.rank==0:
#                 make = True
#             else:
#                 make = False

#             self.meds_filename = get_filename(self.params['out_path'],
#                                 'meds',
#                                 self.params['output_meds'],
#                                 var=self.pointing.filter+'_'+str(self.pix),
#                                 ftype='fits.gz',
#                                 overwrite=False,
#                                 make=make)
#             self.local_meds = get_filename('./',
#                                 'meds',
#                                 self.params['output_meds'],
#                                 var=self.pointing.filter+'_'+str(self.pix),
#                                 ftype='fits',
#                                 overwrite=False,
#                                 make=make)

#             self.local_meds_psf = self.local_meds
#             if 'psf_meds' in self.params:
#                 if self.params['psf_meds'] is not None:
#                     self.meds_psf = get_filename(self.params['psf_path'],
#                             'meds',
#                             self.params['psf_meds'],
#                             var=self.pointing.filter+'_'+str(self.pix),
#                             ftype='fits.gz',
#                             overwrite=False,make=False)
#                     if self.meds_psf!=self.meds_filename:
#                         self.local_meds_psf = get_filename('./',
#                                     '',
#                                     self.params['psf_meds'],
#                                     var=self.pointing.filter+'_'+str(self.pix),
#                                     ftype='fits',
#                                     overwrite=False)
#                     if not condor:
#                         if self.meds_psf!=self.meds_filename:
#                             shutil.copy(self.meds_psf,self.local_meds_psf+'.gz')
#                             os.system( 'gunzip '+self.local_meds_psf+'.gz')

#         if self.rank>0:
#             return

#         print('to before setup')
#         if setup:
#             self.accumulate_index_table()
#             return

#         if condor_build:
#             self.load_index(full=True)
#             self.condor_build()
#             return

#         self.load_index()
#         tmp = self.EmptyMEDS()
#         if tmp is None:
#             self.skip = True
#             return
#         if tmp:
#             shutil.copy(self.meds_filename,self.local_meds+'.gz')
#             os.system( 'gunzip '+self.local_meds+'.gz')
#             self.file_exists = True
#             return
#         self.accumulate_dithers()


#     def accumulate_index_table(self):

#         print('inside accumulate')

#         index_filename = get_filename(self.params['out_path'],
#                             'truth',
#                             self.params['output_meds'],
#                             var=self.pointing.filter+'_index_sorted',
#                             ftype='fits.gz',
#                             overwrite=False)

#         if (os.path.exists(index_filename)) and (not self.params['overwrite']):

#             print('break accumulate')
#             return

#         else:

#             if not setup:
#                 raise ParamError('Trying to setup index file in potentially parallel run. Run with setup first.')

#             print('good accumulate')
#             index_files = get_filenames(self.params['out_path'],
#                                         'truth',
#                                         self.params['output_meds'],
#                                         var='index'+'_'+self.pointing.filter,
#                                         ftype='fits')

#             print('good2 accumulate',index_files)
#             length = 0
#             for filename in index_files:
#                 print('length ',filename)
#                 length+=fio.FITS(filename)[-1].read_header()['NAXIS2']

#             print('tmp')

#             self.index = np.zeros(length,dtype=fio.FITS(index_files[0])[-1].read().dtype)
#             length = 0
#             for filename in index_files:
#                 print('reading ',filename)
#                 f = fio.FITS(filename)[-1].read()
#                 self.index[length:length+len(f)] = f
#                 length += len(f)

#             self.index = self.index[np.argsort(self.index, order=['ind','dither'])]

#             steps = np.where(np.roll(self.index['ind'],1)!=self.index['ind'])[0]
#             self.index_ = np.zeros(len(self.index)+len(np.unique(self.index['ind'])),dtype=self.index.dtype)
#             for name in self.index.dtype.names:
#                 if name=='dither':
#                     self.index_[name] = np.insert(self.index[name],steps,np.ones(len(steps))*-1)
#                 else:
#                     self.index_[name] = np.insert(self.index[name],steps,self.index[name][steps])

#             self.index = self.index_
#             self.index_= None
#             self.index['ra']  = np.degrees(self.index['ra'])
#             self.index['dec'] = np.degrees(self.index['dec'])
#             fio.write(index_filename,self.index,clobber=True)

#     def condor_build(self):

#         if not self.params['condor']:
#             return

#         a = """#-*-shell-script-*-

# universe     = vanilla
# Requirements = OSGVO_OS_VERSION == "7" && CVMFS_oasis_opensciencegrid_org_REVISION >= 10686 && (HAS_CVMFS_sw_lsst_eu =?= True)

# +ProjectName = "duke.lsst"
# +WantsCvmfsStash = true
# request_memory = 4G

# should_transfer_files = YES
# when_to_transfer_output = ON_EXIT_OR_EVICT
# Executable     = ../run_osg.sh
# transfer_output_files   = meds
# Initialdir     = /stash/user/troxel/wfirst_sim_%s/
# log            = %s_meds_log_$(MEDS).log
# Arguments = %s_osg.yaml H158 meds $(MEDS)
# Output         = %s_meds_$(MEDS).log
# Error          = %s_meds_$(MEDS).log


# """ % (self.params['output_meds'],self.params['output_tag'],self.params['output_tag'],self.params['output_tag'],self.params['output_tag'])

#         a2 = """#-*-shell-script-*-

# universe     = vanilla
# Requirements = OSGVO_OS_VERSION == "7" && CVMFS_oasis_opensciencegrid_org_REVISION >= 10686 && (HAS_CVMFS_sw_lsst_eu =?= True)

# +ProjectName = "duke.lsst"
# +WantsCvmfsStash = true
# request_memory = 2G

# should_transfer_files = YES
# when_to_transfer_output = ON_EXIT_OR_EVICT
# Executable     = ../run_osg.sh
# transfer_output_files   = ngmix
# Initialdir     = /stash/user/troxel/wfirst_sim_%s/
# log            = %s_shape_log_$(MEDS)_$(ITER).log
# Arguments = %s_osg.yaml H158 meds shape $(MEDS) $(ITER) 5
# Output         = %s_shape_$(MEDS)_$(ITER).log
# Error          = %s_shape_$(MEDS)_$(ITER).log


# """ % (self.params['output_meds'],self.params['output_tag'],self.params['output_tag'],self.params['output_tag'],self.params['output_tag'])

#         b = """transfer_input_files    = /home/troxel/wfirst_stack/wfirst_stack.tar.gz, \
# /home/troxel/wfirst_imsim_paper1/code/osg_runs/%s/%s_osg.yaml, \
# /home/troxel/wfirst_imsim_paper1/code/meds_pix_list.txt, \
# /stash/user/troxel/wfirst_sim_%s/run.tar""" % (self.params['output_meds'],self.params['output_tag'],self.params['output_meds'])

#         # print(self.index)
#         pix0 = self.get_index_pix()
#         # print(pix0)
#         p = np.unique(pix0)
#         p2 = np.array_split(p,10)
#         for ip2,p2_ in enumerate(p2):
#             script = a+"""
# """
#             print(p)
#             for ip,p_ in enumerate(p2_):
#                 # if ip>3:
#                 #     continue
#                 meds_psf = get_filename(self.params['psf_path'],
#                 'meds',
#                 self.params['psf_meds'],
#                 var=self.pointing.filter+'_'+str(p_),
#                 ftype='fits.gz',
#                 overwrite=False,make=False)
#                 file_list = ''
#                 stamps_used = np.unique(self.index[['dither','sca']][pix0==p_])
#                 for i in range(len(stamps_used)):
#                     if stamps_used['dither'][i]==-1:
#                         continue
#                     print(p_,i)
#                     # filename = '/stash/user/troxel/wfirst_sim_fiducial/stamps/fiducial_H158_'+str(stamps_used['dither'][i])+'/'+str(stamps_used['sca'][i])+'_0.cPickle'
#                     filename = get_filename(self.params['condor_zip_dir'],
#                                             'stamps',
#                                             self.params['output_meds'],
#                                             var=self.pointing.filter+'_'+str(stamps_used['dither'][i]),
#                                             name2=str(stamps_used['sca'][i])+'_0',
#                                             ftype='cPickle.gz',
#                                             overwrite=False,make=False)
#                     file_list+=', '+filename
#                 d = """MEDS=%s
# Queue

# """ % (str(p_))
#                 script+="""
# """+b
#                 if 'psf_meds' in self.params:
#                     if self.params['psf_meds'] is not None:
#                         if self.params['psf_meds']!=self.params['output_meds']:
#                             script+=', '+meds_psf
#                 script+=file_list+"""
# """+d

#             # print(script)
#             # print(self.params['psf_meds'])
#             f = open(self.params['output_tag']+'_meds_run_osg_'+str(ip2)+'.sh','w')
#             f.write(script)
#             f.close()

#         script = a2+"""
# """
#         meds_psf = get_filename(self.params['psf_path'],
#                             'meds',
#                             self.params['psf_meds'],
#                             var=self.pointing.filter+'_$(MEDS)',
#                             ftype='fits.gz',
#                             overwrite=False,make=False)
#         meds = get_filename(self.params['condor_zip_dir'],
#                             'meds',
#                             self.params['output_meds'],
#                             var=self.pointing.filter+'_$(MEDS)',
#                             ftype='fits.gz',
#                             overwrite=False,make=False)
#         script+="""
# """+b
#         script+=', '+meds
#         if 'psf_meds' in self.params:
#             if self.params['psf_meds'] is not None:
#                 if self.params['psf_meds']!=self.params['output_meds']:
#                     script+=', '+meds_psf

#         for ip,p_ in enumerate(p):
#             d = """MEDS=%s
# Queue ITER from seq 0 1 4 |

# """ % (str(p_))
#             script+="""
# """+d

#         f = open(self.params['output_tag']+'_meds_shape_osg.sh','w')
#         f.write(script)
#         f.close()

#     def load_index(self,full=False):

#         index_filename = get_filename(self.params['out_path'],
#                             'truth',
#                             self.params['output_meds'],
#                             var=self.pointing.filter+'_index_sorted',
#                             ftype='fits.gz',
#                             overwrite=False)

#         self.index = fio.FITS(index_filename)[-1].read()
#         if full:
#             self.index = self.index[self.index['stamp']!=0]
#         else:
#             self.index = self.index[(self.index['stamp']!=0) & (self.get_index_pix()==self.pix)]
#         # print 'debugging here'
#         # self.index = self.index[self.index['ind']<np.unique(self.index['ind'])[5]]
#         # print self.index
#         self.steps = np.where(np.roll(self.index['ind'],1)!=self.index['ind'])[0]
#         # print self.steps
#         # print 'debugging here'

#     def mask_index(self,pix):

#         return self.index[self.get_index_pix()==pix]

#     def get_index_pix(self):

#         return hp.ang2pix(self.params['nside'],old_div(np.pi,2.)-np.radians(self.index['dec']),np.radians(self.index['ra']),nest=True)

#     def EmptyMEDS(self):
#         """
#         Based on galsim.des.des_meds.WriteMEDS().
#         """

#         from galsim._pyfits import pyfits

#         if len(self.index)==0:
#             print('skipping due to no objects')
#             return None

#         if (os.path.exists(self.meds_filename+'.gz')) or (os.path.exists(self.meds_filename)):
#             if not self.params['overwrite']:
#                 print('skipping due to file exists')
#                 return True
#             os.remove(self.meds_filename+'.gz')
#             if os.path.exists(self.meds_filename):
#                 os.remove(self.meds_filename)
#         if os.path.exists(self.local_meds):
#             os.remove(self.local_meds)
#         if os.path.exists(self.local_meds+'.gz'):
#             os.remove(self.local_meds+'.gz')

#         print(self.local_meds)
#         m = fio.FITS(self.local_meds,'rw',clobber=True)

#         print('Starting empty meds pixel',self.pix)
#         indices = self.index['ind']
#         bincount = np.bincount(indices)
#         indcheck = np.where(bincount>0)[0]
#         bincount = bincount[bincount>0]
#         MAX_NCUTOUTS = np.max(bincount)
#         assert np.sum(bincount==1) == 0
#         assert np.all(indcheck==np.unique(indices))
#         assert np.all(indcheck==indices[self.steps])
#         cum_exps = len(indices)
#         # get number of objects
#         n_obj = len(indcheck)

#         # get the primary HDU
#         primary = pyfits.PrimaryHDU()

#         # second hdu is the object_data
#         # cf. https://github.com/esheldon/meds/wiki/MEDS-Format
#         dtype = [
#             ('id', 'i8'),
#             ('number', 'i8'),
#             ('box_size', 'i8'),
#             ('psf_box_size', 'i8'),
#             ('psf_box_size2', 'i8'),
#             ('ra','f8'),
#             ('dec','f8'),
#             ('ncutout', 'i8'),
#             ('file_id', 'i8', (MAX_NCUTOUTS,)),
#             ('start_row', 'i8', (MAX_NCUTOUTS,)),
#             ('psf_start_row', 'i8', (MAX_NCUTOUTS,)),
#             ('psf_start_row2', 'i8', (MAX_NCUTOUTS,)),
#             ('orig_row', 'f8', (MAX_NCUTOUTS,)),
#             ('orig_col', 'f8', (MAX_NCUTOUTS,)),
#             ('orig_start_row', 'i8', (MAX_NCUTOUTS,)),
#             ('orig_start_col', 'i8', (MAX_NCUTOUTS,)),
#             ('cutout_row', 'f8', (MAX_NCUTOUTS,)),
#             ('cutout_col', 'f8', (MAX_NCUTOUTS,)),
#             ('dudrow', 'f8', (MAX_NCUTOUTS,)),
#             ('dudcol', 'f8', (MAX_NCUTOUTS,)),
#             ('dvdrow', 'f8', (MAX_NCUTOUTS,)),
#             ('dvdcol', 'f8', (MAX_NCUTOUTS,)),
#             ('dither', 'i8', (MAX_NCUTOUTS,)),
#             ('sca', 'i8', (MAX_NCUTOUTS,)),
#         ]

#         data                 = np.zeros(n_obj,dtype)
#         data['id']           = np.arange(n_obj)
#         data['number']       = self.index['ind'][self.steps]
#         data['ra']           = self.index['ra'][self.steps]
#         data['dec']          = self.index['dec'][self.steps]
#         data['ncutout']      = bincount
#         for i in range(len(self.steps)-1):
#             data['box_size'][i] = np.min(self.index['stamp'][self.steps[i]:self.steps[i+1]])
#         data['box_size'][i+1]   = np.min(self.index['stamp'][self.steps[-1]:])
#         data['psf_box_size'] = np.ones(n_obj)*self.params['psf_stampsize']
#         data['psf_box_size2'] = np.ones(n_obj)*self.params['psf_stampsize']*self.params['oversample']
#         m.write(data,extname='object_data')

#         length = np.sum(bincount*data['box_size']**2)
#         psf_length = np.sum(bincount*data['psf_box_size']**2)
#         psf_length2 = np.sum(bincount*data['psf_box_size2']**2)
#         # print 'lengths',length,psf_length,bincount,data['box_size']

#         # third hdu is image_info
#         dtype = [
#             ('image_path', 'S256'),
#             ('image_ext', 'i8'),
#             ('weight_path', 'S256'),
#             ('weight_ext', 'i8'),
#             ('seg_path','S256'),
#             ('seg_ext','i8'),
#             ('bmask_path', 'S256'),
#             ('bmask_ext', 'i8'),
#             ('bkg_path', 'S256'),
#             ('bkg_ext', 'i8'),
#             ('image_id', 'i8'),
#             ('image_flags', 'i8'),
#             ('magzp', 'f8'),
#             ('scale', 'f8'),
#             ('position_offset', 'f8'),
#         ]

#         gstring             = 'generated_by_galsim'
#         data                = np.zeros(n_obj,dtype)
#         data['image_path']  = gstring
#         data['weight_path'] = gstring
#         data['seg_path']    = gstring
#         data['bmask_path']  = gstring
#         data['bkg_path']    = gstring
#         data['magzp']       = 30
#         m.write(data,extname='image_info')

#         # fourth hdu is metadata
#         # default values?
#         dtype = [
#             ('magzp_ref', 'f8'),
#             ('DESDATA', 'S256'),
#             ('cat_file', 'S256'),
#             ('coadd_image_id', 'S256'),
#             ('coadd_file','S256'),
#             ('coadd_hdu','i8'),
#             ('coadd_seg_hdu', 'i8'),
#             ('coadd_srclist', 'S256'),
#             ('coadd_wt_hdu', 'i8'),
#             ('coaddcat_file', 'S256'),
#             ('coaddseg_file', 'S256'),
#             ('cutout_file', 'S256'),
#             ('max_boxsize', 'S3'),
#             ('medsconv', 'S3'),
#             ('min_boxsize', 'S2'),
#             ('se_badpix_hdu', 'i8'),
#             ('se_hdu', 'i8'),
#             ('se_wt_hdu', 'i8'),
#             ('seg_hdu', 'i8'),
#             ('psf_hdu', 'i8'),
#             ('sky_hdu', 'i8'),
#             ('fake_coadd_seg', 'f8'),
#         ]

#         data                   = np.zeros(n_obj,dtype)
#         data['magzp_ref']      = 30
#         data['DESDATA']        = gstring
#         data['cat_file']       = gstring
#         data['coadd_image_id'] = gstring
#         data['coadd_file']     = gstring
#         data['coadd_hdu']      = 9999
#         data['coadd_seg_hdu']  = 9999
#         data['coadd_srclist']  = gstring
#         data['coadd_wt_hdu']   = 9999
#         data['coaddcat_file']  = gstring
#         data['coaddseg_file']  = gstring
#         data['cutout_file']    = gstring
#         data['max_boxsize']    = '-1'
#         data['medsconv']       = 'x'
#         data['min_boxsize']    = '-1'
#         data['se_badpix_hdu']  = 9999
#         data['se_hdu']         = 9999
#         data['se_wt_hdu']      = 9999
#         data['seg_hdu']        = 9999
#         data['psf_hdu']        = 9999
#         data['sky_hdu']        = 9999
#         data['fake_coadd_seg'] = 9999
#         m.write(data,extname='metadata')

#         # rest of HDUs are image vectors
#         print('Writing empty meds pixel',self.pix)
#         m.write(np.zeros(length,dtype='f8'),extname='image_cutouts')
#         m.write(np.zeros(length,dtype='f8'),extname='weight_cutouts')
#         # m.write(np.zeros(length,dtype='f8'),extname='seg_cutouts')
#         m.write(np.zeros(psf_length,dtype='f8'),extname='psf')
#         m.write(np.zeros(psf_length2,dtype='f8'),extname='psf2')
#         # m['image_cutouts'].write(np.zeros(1,dtype='f8'), start=[length])
#         # m['weight_cutouts'].write(np.zeros(1,dtype='f8'), start=[length])
#         # m['seg_cutouts'].write(np.zeros(1,dtype='f8'), start=[length])
#         # m['psf'].write(np.zeros(1,dtype='f8'), start=[psf_length])

#         m.close()
#         print('Done empty meds pixel',self.pix)

#         return False

#     def dump_meds_start_info(self,object_data,i,j):

#         object_data['start_row'][i][j] = np.sum((object_data['ncutout'][:i])*object_data['box_size'][:i]**2)+j*object_data['box_size'][i]**2
#         # change here
#         # object_data['psf_start_row'][i][j] = np.sum((object_data['ncutout'][:i])*object_data['box_size'][:i]**2)+j*object_data['box_size'][i]**2
#         object_data['psf_start_row'][i][j] = np.sum((object_data['ncutout'][:i])*object_data['psf_box_size'][:i]**2)+j*object_data['psf_box_size'][i]**2
#         object_data['psf_start_row2'][i][j] = np.sum((object_data['ncutout'][:i])*object_data['psf_box_size2'][:i]**2)+j*object_data['psf_box_size2'][i]**2
#         # print 'starts',i,j,object_data['start_row'][i][j],object_data['psf_start_row'][i][j],object_data['box_size'][i],object_data['psf_box_size'][i]

#     def dump_meds_wcs_info( self,
#                             object_data,
#                             i,
#                             j,
#                             x,
#                             y,
#                             origin_x,
#                             origin_y,
#                             dither,
#                             sca,
#                             dudx,
#                             dudy,
#                             dvdx,
#                             dvdy,
#                             wcsorigin_x=None,
#                             wcsorigin_y=None):

#         object_data['orig_row'][i][j]       = y
#         object_data['orig_col'][i][j]       = x
#         object_data['orig_start_row'][i][j] = origin_y
#         object_data['orig_start_col'][i][j] = origin_x
#         object_data['dither'][i][j]         = dither
#         object_data['sca'][i][j]            = sca
#         object_data['dudcol'][i][j]         = dudx
#         object_data['dudrow'][i][j]         = dudy
#         object_data['dvdcol'][i][j]         = dvdx
#         object_data['dvdrow'][i][j]         = dvdy
#         if wcsorigin_y is None:
#             object_data['cutout_row'][i][j]     = y-origin_y
#         else:
#             object_data['cutout_row'][i][j]     = wcsorigin_y
#         if wcsorigin_x is None:
#             object_data['cutout_col'][i][j]     = x-origin_x
#         else:
#             object_data['cutout_col'][i][j]     = wcsorigin_x

#     def dump_meds_pix_info(self,m,object_data,i,j,gal,weight,psf):#,psf2):

#         assert len(gal)==object_data['box_size'][i]**2
#         assert len(weight)==object_data['box_size'][i]**2
#         # assert len(psf)==object_data['psf_box_size'][i]**2
#         # change here
#         m['image_cutouts'].write(gal, start=object_data['start_row'][i][j])
#         m['weight_cutouts'].write(weight, start=object_data['start_row'][i][j])
#         m['psf'].write(psf, start=object_data['psf_start_row'][i][j])
#         m['psf2'].write(psf2, start=object_data['psf_start_row2'][i][j])

#     def accumulate_dithers(self):
#         """
#         Accumulate the written pickle files that contain the postage stamps for all objects, with SCA and dither ids.
#         Write stamps to MEDS file, and SCA and dither ids to truth files.
#         """


#         print('mpi check',self.rank,self.size)

#         print('Starting meds pixel',self.pix)
#         m = fio.FITS(self.local_meds,'rw')
#         object_data = m['object_data'].read()

#         stamps_used = np.unique(self.index[['dither','sca']])
#         print('number of files',stamps_used)
#         for si,s in enumerate(range(len(stamps_used))):
#             if stamps_used['dither'][s] == -1:
#                 continue

#             if condor:
#                 filename = get_filename('./',
#                                         '',
#                                         self.params['output_meds'],
#                                         var=self.pointing.filter+'_'+str(stamps_used['dither'][s]),
#                                         name2=str(stamps_used['sca'][s])+'_0',
#                                         ftype='cPickle',
#                                         overwrite=False)
#             else:
#                 filename1 = get_filename(self.params['out_path'],
#                                         'stamps',
#                                         self.params['output_meds'],
#                                         var=self.pointing.filter+'_'+str(stamps_used['dither'][s]),
#                                         name2=str(stamps_used['sca'][s])+'_0',
#                                         ftype='cPickle.gz',
#                                         overwrite=False)
#                 filename = get_filename('./',
#                                         '',
#                                         self.params['output_meds'],
#                                         var=self.pointing.filter+'_'+str(stamps_used['dither'][s]),
#                                         name2=str(stamps_used['sca'][s])+'_0',
#                                         ftype='cPickle',
#                                         overwrite=False)
#                 shutil.copy(filename1,filename+'.gz')

#             os.system('gunzip '+filename+'.gz')
#             print(stamps_used['dither'][s],stamps_used['sca'][s])

#             with io.open(filename, 'rb') as p :
#                 unpickler = pickle.Unpickler(p)
#                 while p.peek(1) :
#                     gal = unpickler.load()
#                     i = np.where(gal['ind'] == object_data['number'])[0]
#                     if len(i)==0:
#                         continue
#                     assert len(i)==1
#                     # print gal
#                     i = i[0]
#                     j = np.nonzero(object_data['dither'][i])[0]
#                     if len(j)==0:
#                         j = 0
#                     else:
#                         j = np.max(j)+1
#                     index_i = np.where((self.index['ind']==gal['ind'])&(self.index['dither']==gal['dither']))[0]
#                     assert len(index_i)==1
#                     index_i=index_i[0]

#                     if j==0:
#                         self.dump_meds_start_info(object_data,i,j)
#                         j+=1
#                     self.dump_meds_start_info(object_data,i,j)

#                     if object_data['box_size'][i] > self.index['stamp'][index_i]:
#                         pad_    = old_div((object_data['box_size'][i] - self.index['stamp'][index_i]),2)
#                         gal_    = np.pad(gal['gal'].array,(pad_,pad_),'wrap').flatten()
#                         weight_ = np.pad(gal['weight'].reshape(self.index['stamp'][index_i],self.index['stamp'][index_i]),(pad_,pad_),'wrap').flatten()
#                     elif object_data['box_size'][i] < self.index['stamp'][index_i]:
#                         pad_    = old_div((self.index['stamp'][index_i] - object_data['box_size'][i]),2)
#                         gal_    = gal['gal'].array[pad_:-pad_,pad_:-pad_].flatten()
#                         weight_ = gal['weight'].reshape(self.index['stamp'][index_i],self.index['stamp'][index_i])[pad_:-pad_,pad_:-pad_].flatten()
#                     else:
#                         gal_    = gal['gal'].array.flatten()
#                         weight_ = gal['weight']

#                     # orig_box_size = object_data['box_size'][i]
#                     # if True:
#                     #     object_data['box_size'][i] = int(orig_box_size*1.5)+int(orig_box_size*1.5)%2

#                     # box_diff = object_data['box_size'][i] - self.index['stamp'][index_i]

#                     # ====================
#                     # this is a patch, remove later
#                     # gal['x']+=0.5
#                     # gal['y']+=0.5
#                     # ===================
#                     origin_x = gal['gal'].origin.x
#                     origin_y = gal['gal'].origin.y
#                     gal['gal'].setOrigin(0,0)
#                     new_pos  = galsim.PositionD(gal['x']-origin_x,gal['y']-origin_y)
#                     wcs = gal['gal'].wcs.affine(image_pos=new_pos)
#                     self.dump_meds_wcs_info(object_data,
#                                             i,
#                                             j,
#                                             gal['x'],
#                                             gal['y'],
#                                             origin_x,
#                                             origin_y,
#                                             self.index['dither'][index_i],
#                                             self.index['sca'][index_i],
#                                             wcs.dudx,
#                                             wcs.dudy,
#                                             wcs.dvdx,
#                                             wcs.dvdy)

#                     self.dump_meds_pix_info(m,
#                                             object_data,
#                                             i,
#                                             j,
#                                             gal_,
#                                             weight_,
#                                             gal['psf'],
#                                             gal['psf2'])
#                     # print np.shape(gals[gal]['psf']),gals[gal]['psf']

#         # object_data['psf_box_size'] = object_data['box_size']
#         print('Writing meds pixel',self.pix)
#         m['object_data'].write(object_data)
#         m.close()
#         print('Done meds pixel',self.pix)

#     def finish(self,condor=False):

#         if self.rank>0:
#             return

#         print('start meds finish')
#         if not self.file_exists:
#             os.system('gzip '+self.local_meds)
#         if not condor and not self.file_exists:
#             shutil.move(self.local_meds+'.gz',self.meds_filename)
#             # if os.path.exists(self.local_meds+'.gz'):
#             #     os.remove(self.local_meds+'.gz')
#         print('done meds finish')

#     def get_cutout_psf2(self,m,m2,i,j):

#         box_size = m['psf_box_size2'][i]
#         start_row = m['psf_start_row2'][i, j]
#         row_end = start_row + box_size*box_size

#         imflat = m2['psf2'][start_row:row_end]
#         im = imflat.reshape(box_size, box_size)
#         return im

#     def get_cutout_psf(self,m,m2,i,j):

#         box_size = m['psf_box_size'][i]
#         start_row = m['psf_start_row'][i, j]
#         row_end = start_row + box_size*box_size

#         imflat = m2['psf'][start_row:row_end]
#         im = imflat.reshape(box_size, box_size)
#         return im

#     def get_exp_list(self,m,i,m2=None,size=None):

#         def get_stamp(size,box_size):
#             hlp = size*10./wfirst.pixel_scale
#             if hlp>box_size:
#                 return int(box_size)
#             if hlp<32:
#                 return 32
#             return int(2**(int(np.log2(100))+1))

#         if m2 is None:
#             m2 = m

#         obs_list=ObsList()
#         psf_list=ObsList()

#         if size is not None:
#             box_size = get_stamp(size,m['box_size'][i])

#         included = []
#         w        = []
#         # For each of these objects create an observation
#         for j in range(m['ncutout'][i]):
#             if j==0:
#                 continue
#             # if j>1:
#             #     continue
#             im = m.get_cutout(i, j, type='image')
#             im = im[:,len(im)//2-box_size//2:len(im)//2+box_size//2][len(im)//2-box_size//2:len(im)//2+box_size//2,:]
#             weight = m.get_cutout(i, j, type='weight')
#             weight = weight[:,len(weight)//2-box_size//2:len(weight)//2+box_size//2][len(weight)//2-box_size//2:len(weight)//2+box_size//2,:]

#             im_psf = self.get_cutout_psf(m, m2, i, j)
#             im_psf2 = self.get_cutout_psf2(m, m2, i, j)
#             if np.sum(im)==0.:
#                 print(self.local_meds, i, j, np.sum(im))
#                 print('no flux in image ',i,j)
#                 continue

#             jacob = m.get_jacobian(i, j)
#             gal_jacob=Jacobian(
#                 row=(m['orig_row'][i][j]-m['orig_start_row'][i][j])-m['box_size'][i]/2+box_size/2,
#                 col=(m['orig_col'][i][j]-m['orig_start_col'][i][j])-m['box_size'][i]/2+box_size/2,
#                 dvdrow=jacob['dvdrow'],
#                 dvdcol=jacob['dvdcol'],
#                 dudrow=jacob['dudrow'],
#                 dudcol=jacob['dudcol'])

#             psf_center = old_div((m['psf_box_size2'][i]-1),2.)
#             psf_jacob2=Jacobian(
#                 row=jacob['row0']*self.params['oversample'],
#                 col=jacob['col0']*self.params['oversample'],
#                 dvdrow=old_div(jacob['dvdrow'],self.params['oversample']),
#                 dvdcol=old_div(jacob['dvdcol'],self.params['oversample']),
#                 dudrow=old_div(jacob['dudrow'],self.params['oversample']),
#                 dudcol=old_div(jacob['dudcol'],self.params['oversample']))

#             # Create an obs for each cutout
#             mask = np.where(weight!=0)
#             if 1.*len(weight[mask])/np.product(np.shape(weight))<0.8:
#                 continue

#             w.append(np.mean(weight[mask]))
#             noise = old_div(np.ones_like(weight),w[-1])

#             psf_obs = Observation(im_psf, jacobian=gal_jacob, meta={'offset_pixels':None,'file_id':None})
#             psf_obs2 = Observation(im_psf2, jacobian=psf_jacob2, meta={'offset_pixels':None,'file_id':None})
#             obs = Observation(im, weight=weight, jacobian=gal_jacob, psf=psf_obs, meta={'offset_pixels':None,'file_id':None})
#             obs.set_noise(noise)

#             obs_list.append(obs)
#             psf_list.append(psf_obs2)
#             included.append(j)

#         return obs_list,psf_list,np.array(included)-1,np.array(w)

#     def get_snr(self,obs_list,res,res_full):

#         if res_full['flags']!=0:
#             return -1

#         size = res['pars'][4]
#         flux = res['flux']

#         model_ = galsim.Sersic(1, half_light_radius=1.*size, flux=flux*(1.-res['pars'][5])) + galsim.Sersic(4, half_light_radius=1.*size, flux=flux*res['pars'][5])
#         for i in range(len(obs_list)):
#             obs = obs_list[i]
#             im = obs.psf.image.copy()
#             im *= 1.0/im.sum()/len(obs_list)
#             psf_gsimage = galsim.Image(im,wcs=obs.psf.jacobian.get_galsim_wcs())
#             psf_ii = galsim.InterpolatedImage(psf_gsimage,x_interpolant='lanczos15')

#             model = galsim.Convolve(model_,psf_ii)
#             gal_stamp = galsim.Image(np.shape(obs.image)[0],np.shape(obs.image)[1], wcs=obs.jacobian.get_galsim_wcs())

#             model.drawImage(image=gal_stamp)
#             if i==0:
#                 image = gal_stamp.array*obs.weight
#             else:
#                 image += gal_stamp.array*obs.weight

#         return image.sum()

#     def measure_shape_mof(self,obs_list,T,flux=1000.0,fracdev=None,use_e=None,model='exp'):
#         # model in exp, bdf

#         pix_range = old_div(galsim.wfirst.pixel_scale,10.)
#         e_range = 0.1
#         fdev = 1.
#         def pixe_guess(n):
#             return 2.*n*np.random.random() - n

#         # possible models are 'exp','dev','bdf' galsim.wfirst.pixel_scale
#         cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.wfirst.pixel_scale, galsim.wfirst.pixel_scale)
#         gp = ngmix.priors.GPriorBA(0.3)
#         hlrp = ngmix.priors.FlatPrior(1.0e-4, 1.0e2)
#         fracdevp = ngmix.priors.Normal(0.5, 0.1, bounds=[0., 1.])
#         fluxp = ngmix.priors.FlatPrior(0, 1.0e5) # not sure what lower bound should be in general

#         # center1 + center2 + shape + hlr + fracdev + fluxes for each object
#         # guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,0.5+pixe_guess(fdev),100.])
#         if model=='bdf':
#             if fracdev is None:
#                 fracdev = pixe_guess(fdev)
#             if use_e is None:
#                 e1 = pixe_guess(e_range)
#                 e2 = pixe_guess(e_range)
#             else:
#                 e1 = use_e[0]
#                 e2 = use_e[1]
#             prior = joint_prior.PriorBDFSep(cp, gp, hlrp, fracdevp, fluxp)
#             guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),e1,e2,T,fracdev,flux])
#         elif model=='exp':
#             prior = joint_prior.PriorSimpleSep(cp, gp, hlrp, fluxp)
#             guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,500.])
#         else:
#             raise ParamError('Bad model choice.')

#         if not self.params['avg_fit']:
#             multi_obs_list=MultiBandObsList()
#             multi_obs_list.append(obs_list)

#             fitter = mof.GSMOF([multi_obs_list], model, prior)
#             # center1 + center2 + shape + hlr + fracdev + fluxes for each object
#             # guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,0.5+pixe_guess(fdev),100.])
#             fitter.go(guess)

#             res_ = fitter.get_object_result(0)
#             res_full_  = fitter.get_result()
#             if model=='exp':
#                 res_['flux'] = res_['pars'][5]
#             else:
#                 res_['flux'] = res_['pars'][6]

#             res_['s2n_r'] = self.get_snr(obs_list,res_,res_full_)

#             return res_,res_full_

#         else:

#             out = []
#             out_obj = []
#             for i in range(len(obs_list)):
#                 multi_obs_list = MultiBandObsList()
#                 tmp_obs_list = ObsList()
#                 tmp_obs_list.append(obs_list[i])
#                 multi_obs_list.append(tmp_obs_list)

#                 fitter = mof.KGSMOF([multi_obs_list], 'bdf', prior)
#                 # center1 + center2 + shape + hlr + fracdev + fluxes for each object
#                 # guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,0.5+pixe_guess(fdev),100.])
#                 guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,0.5+pixe_guess(fdev),100.])
#                 fitter.go(guess)

#                 out_obj.append(fitter.get_object_result(0))
#                 out.append(fitter.get_result())

#             return out_obj,out


#     def measure_shape_gmix(self,obs_list,T,flux=1000.0,fracdev=None,use_e=None,model='exp'):
#         # model in exp, bdf

#         pix_range = old_div(galsim.wfirst.pixel_scale,10.)
#         e_range = 0.1
#         fdev = 1.
#         def pixe_guess(n):
#             return 2.*n*np.random.random() - n

#         # possible models are 'exp','dev','bdf' galsim.wfirst.pixel_scale
#         cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.wfirst.pixel_scale, galsim.wfirst.pixel_scale)
#         gp = ngmix.priors.GPriorBA(0.3)
#         hlrp = ngmix.priors.FlatPrior(1.0e-4, 1.0e2)
#         fracdevp = ngmix.priors.Normal(0.5, 0.1, bounds=[0., 1.])
#         fluxp = ngmix.priors.FlatPrior(0, 1.0e5) # not sure what lower bound should be in general

#         # center1 + center2 + shape + hlr + fracdev + fluxes for each object
#         # guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,0.5+pixe_guess(fdev),100.])
#         if model=='bdf':
#             if fracdev is None:
#                 fracdev = pixe_guess(fdev)
#             if use_e is None:
#                 e1 = pixe_guess(e_range)
#                 e2 = pixe_guess(e_range)
#             else:
#                 e1 = use_e[0]
#                 e2 = use_e[1]
#             prior = joint_prior.PriorBDFSep(cp, gp, hlrp, fracdevp, fluxp)
#             guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),e1,e2,T,fracdev,flux])
#         elif model=='exp':
#             prior = joint_prior.PriorSimpleSep(cp, gp, hlrp, fluxp)
#             guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,500.])
#         else:
#             raise ParamError('Bad model choice.')

#             multi_obs_list=MultiBandObsList()
#             multi_obs_list.append(obs_list)

#             fitter = mof.GSMOF([multi_obs_list], model, prior)
#             # center1 + center2 + shape + hlr + fracdev + fluxes for each object
#             # guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,0.5+pixe_guess(fdev),100.])
#             fitter.go(guess)

#             res_ = fitter.get_object_result(0)
#             res_full_  = fitter.get_result()
#             if model=='exp':
#                 res_['flux'] = res_['pars'][5]
#             else:
#                 res_['flux'] = res_['pars'][6]

#             res_['s2n_r'] = self.get_snr(obs_list,res_,res_full_)

#             return res_,res_full_

#     def measure_shape_ngmix(self,obs_list,T,flux=1000.0,model='exp'):

#         pix_range = old_div(galsim.wfirst.pixel_scale,10.)
#         e_range = 0.1
#         fdev = 1.
#         def pixe_guess(n):
#             return 2.*n*np.random.random() - n

#         # possible models are 'exp','dev','bdf' galsim.wfirst.pixel_scale
#         cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.wfirst.pixel_scale, galsim.wfirst.pixel_scale)
#         gp = ngmix.priors.GPriorBA(0.3)
#         hlrp = ngmix.priors.FlatPrior(1.0e-4, 1.0e2)
#         fracdevp = ngmix.priors.TruncatedGaussian(0.5, 0.5, -0.5, 1.5)
#         fluxp = ngmix.priors.FlatPrior(0, 1.0e5) # not sure what lower bound should be in general

#         prior = joint_prior.PriorBDFSep(cp, gp, hlrp, fracdevp, fluxp)
#         # center1 + center2 + shape + hlr + fracdev + fluxes for each object
#         # guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,0.5+pixe_guess(fdev),100.])
#         guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,pixe_guess(fdev),300.])

#         if not self.params['avg_fit']:

#             guesser           = R50FluxGuesser(T,flux)
#             ntry              = 5
#             runner            = GalsimRunner(obs_list,model,guesser=guesser)
#             runner.go(ntry=ntry)
#             fitter            = runner.get_fitter()

#             res_ = fitter.get_result()
#             if model=='exp':
#                 res_['flux'] = res_['pars'][5]
#             else:
#                 res_['flux'] = res_['pars'][6]

#             return res_,res_

#         else:

#             out = []
#             out_obj = []
#             for i in range(len(obs_list)):

#                 tmp_obs_list = ObsList()
#                 tmp_obs_list.append(obs_list[i])
#                 guesser           = R50FluxGuesser(T,flux)
#                 ntry              = 5
#                 runner            = GalsimRunner(tmp_obs_list,model,guesser=guesser)
#                 runner.go(ntry=ntry)
#                 fitter            = runner.get_fitter()
#                 out.append(fitter.get_result())
#                 out_obj.append(fitter.get_result())

#             return out_obj,out

#     def make_jacobian(self,dudx,dudy,dvdx,dvdy,x,y):
#         j = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)
#         return j.withOrigin(galsim.PositionD(x,y))

#     def measure_psf_shape(self,obs_list,T_guess=0.16):
#         # doesn't work

#         def make_ngmix_prior(T, pixel_scale):

#             # centroid is 1 pixel gaussian in each direction
#             cen_prior=priors.CenPrior(0.0, 0.0, pixel_scale, pixel_scale)

#             # g is Bernstein & Armstrong prior with sigma = 0.1
#             gprior=priors.GPriorBA(0.1)

#             # T is log normal with width 0.2
#             Tprior=priors.LogNormal(T, 0.05)

#             # flux is the only uninformative prior
#             Fprior=priors.FlatPrior(-10.0, 1e3)

#             prior=joint_prior.PriorSimpleSep(cen_prior, gprior, Tprior, Fprior)
#             return prior

#         T_guess = (old_div(T_guess, 2.35482))**2 * 2.

#         cnt = dx = dy = e1 = e2 = T = flux = 0
#         for ipsf,psf in enumerate(obs_list):

#             # try:

#             obs = ngmix.Observation(image=psf.psf.image, jacobian=psf.psf.jacobian)

#             lm_pars = {'maxfev':4000}
#             wcs = self.make_jacobian(psf.psf.jacobian.dudcol,
#                                     psf.psf.jacobian.dudrow,
#                                     psf.psf.jacobian.dvdcol,
#                                     psf.psf.jacobian.dvdrow,
#                                     psf.psf.jacobian.col0,
#                                     psf.psf.jacobian.row0)
#             prior = make_ngmix_prior(T_guess, wcs.minLinearScale())
#             runner=PSFRunner(obs, 'gauss', T_guess, lm_pars, prior=prior)
#             runner.go(ntry=5)

#             flag = runner.fitter.get_result()['flags']
#             gmix = runner.fitter.get_gmix()

#             # except Exception as e:
#             #     print 'exception'
#             #     cnt+=1
#             #     continue

#             if flag != 0:
#                 print('flag',flag)
#                 cnt+=1
#                 continue

#             e1_, e2_, T_ = gmix.get_g1g2T()
#             dx_, dy_ = gmix.get_cen()
#             if (np.abs(e1_) > 0.5) or (np.abs(e2_) > 0.5) or (dx_**2 + dy_**2 > MAX_CENTROID_SHIFT**2):
#                 print('g,xy',e1_,e2_,dx_,dy_)
#                 cnt+=1
#                 continue

#             flux_ = old_div(gmix.get_flux(), wcs.pixelArea())

#             dx   += dx_
#             dy   += dy_
#             e1   += e1_
#             e2   += e2_
#             T    += T_
#             flux += flux_

#         if cnt == len(obs_list):
#             return None

#         return cnt, old_div(dx,(len(obs_list)-cnt)), old_div(dy,(len(obs_list)-cnt)), old_div(e1,(len(obs_list)-cnt)), old_div(e2,(len(obs_list)-cnt)), old_div(T,(len(obs_list)-cnt)), old_div(flux,(len(obs_list)-cnt))

#     def measure_psf_shape_moments(self,obs_list):

#         def make_psf_image(self,obs):

#             wcs = self.make_jacobian(obs.jacobian.dudcol,
#                                     obs.jacobian.dudrow,
#                                     obs.jacobian.dvdcol,
#                                     obs.jacobian.dvdrow,
#                                     obs.jacobian.col0,
#                                     obs.jacobian.row0)

#             return galsim.Image(obs.image, xmin=1, ymin=1, wcs=wcs)

#         out = np.zeros(len(obs_list),dtype=[('e1','f4')]+[('e2','f4')]+[('T','f4')]+[('dx','f4')]+[('dy','f4')]+[('flag','i2')])
#         for iobs,obs in enumerate(obs_list):

#             M = e1 = e2= 0
#             im = make_psf_image(self,obs)

#             try:
#                 shape_data = im.FindAdaptiveMom(weight=None, strict=False)
#             except:
#                 out['flag'][iobs] |= BAD_MEASUREMENT
#                 continue

#             if shape_data.moments_status != 0:
#                 out['flag'][iobs] |= BAD_MEASUREMENT
#                 continue

#             out['dx'][iobs] = shape_data.moments_centroid.x - im.true_center.x
#             out['dy'][iobs] = shape_data.moments_centroid.y - im.true_center.y
#             if out['dx'][iobs]**2 + out['dy'][iobs]**2 > MAX_CENTROID_SHIFT**2:
#                 out['flag'][iobs] |= CENTROID_SHIFT
#                 continue

#             # Account for the image wcs
#             if im.wcs.isPixelScale():
#                 out['e1'][iobs] = shape_data.observed_shape.g1
#                 out['e2'][iobs] = shape_data.observed_shape.g2
#                 out['T'][iobs]  = 2 * shape_data.moments_sigma**2 * im.scale**2
#             else:
#                 e1    = shape_data.observed_shape.e1
#                 e2    = shape_data.observed_shape.e2
#                 s     = shape_data.moments_sigma
#                 jac   = im.wcs.jacobian(im.true_center)
#                 M     = np.matrix( [[ 1 + e1, e2 ], [ e2, 1 - e1 ]] ) * s*s
#                 J     = jac.getMatrix()
#                 M     = J * M * J.T
#                 scale = np.sqrt(M/2./s/s)
#                 e1    = old_div((M[0,0] - M[1,1]), (M[0,0] + M[1,1]))
#                 e2    = old_div((2.*M[0,1]), (M[0,0] + M[1,1]))
#                 shear = galsim.Shear(e1=e1, e2=e2)
#                 out['T'][iobs]  = M[0,0] + M[1,1]
#                 out['e1'][iobs] = shear.g1
#                 out['e2'][iobs] = shear.g2

#         return out

#     def get_coadd_shape(self):


#         def get_flux(obs_list):
#             flux = 0.
#             for obs in obs_list:
#                 flux += obs.image.sum()
#             flux /= len(obs_list)
#             if flux<0:
#                 flux = 10.
#             return flux

#         #tmp
#         # self.psf_model = []
#         # for i in range(1,19):
#         #     self.pointing.sca = i
#         #     self.pointing.get_psf()
#         #     self.psf_model.append(self.pointing.PSF)
#         #tmp

#         print('mpi check 2',self.rank,self.size)

#         filename = get_filename(self.params['out_path'],
#                                 'truth',
#                                 self.params['output_truth'],
#                                 name2='truth_gal',
#                                 overwrite=False)
#         truth = fio.FITS(filename)[-1]
#         m  = meds.MEDS(self.local_meds)
#         m2 = fio.FITS(self.local_meds_psf)
#         if self.shape_iter is not None:
#             indices = np.array_split(np.arange(len(m['number'][:])),self.shape_cnt)[self.shape_iter]
#         else:
#             indices = np.arange(len(m['number'][:]))

#         print('rank in coadd_shape', self.rank)
#         coadd = {}
#         res   = np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),])#, ('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])

#         for i,ii in enumerate(indices):
#             if i%self.size!=self.rank:
#                 continue
#             if i%100==0:
#                 print('made it to object',i)
#             try_save = False

#             ind = m['number'][ii]
#             t   = truth[ind]

#             res['ind'][i]                       = ind
#             res['ra'][i]                        = t['ra']
#             res['dec'][i]                       = t['dec']
#             res['nexp_tot'][i]                  = m['ncutout'][ii]-1
#             res['stamp'][i]                     = m['box_size'][ii]
#             res['g1'][i]                        = t['g1']
#             res['g2'][i]                        = t['g2']
#             res['int_e1'][i]                    = t['int_e1']
#             res['int_e2'][i]                    = t['int_e2']
#             res['rot'][i]                       = t['rot']
#             res['size'][i]                      = t['size']
#             res['redshift'][i]                  = t['z']
#             res['mag_'+self.pointing.filter][i] = t[self.pointing.filter]
#             res['pind'][i]                      = t['pind']
#             res['bulge_flux'][i]                = t['bflux']
#             res['disk_flux'][i]                 = t['dflux']

#             obs_list,psf_list,included,w = self.get_exp_list(m,ii,m2=m2,size=t['size'])
#             if len(included)==0:
#                 continue
#             # coadd[i]            = psc.Coadder(obs_list).coadd_obs
#             # coadd[i].set_meta({'offset_pixels':None,'file_id':None})
#             if self.params['shape_code']=='mof':
#                 res_,res_full_      = self.measure_shape_mof(obs_list,t['size'],flux=get_flux(obs_list),fracdev=t['bflux'],use_e=[t['int_e1'],t['int_e2']],model=self.params['ngmix_model'])
#             elif self.params['shape_code']=='ngmix':
#                 res_,res_full_      = self.measure_shape_ngmix(obs_list,t['size'],model=self.params['ngmix_model'])
#             else:
#                 raise ParamError('unknown shape code request')
#             if res_full_['flags'] !=0:
#                 print('failed',i,ii,get_flux(obs_list))

#             wcs = self.make_jacobian(obs_list[0].jacobian.dudcol,
#                                     obs_list[0].jacobian.dudrow,
#                                     obs_list[0].jacobian.dvdcol,
#                                     obs_list[0].jacobian.dvdrow,
#                                     obs_list[0].jacobian.col0,
#                                     obs_list[0].jacobian.row0)

#             if not self.params['avg_fit']:
#                 res['nexp_used'][i]                 = len(included)
#                 res['flags'][i]                     = res_full_['flags']
#                 if res_full_['flags']==0:
#                     res['px'][i]                        = res_['pars'][0]
#                     res['py'][i]                        = res_['pars'][1]
#                     res['flux'][i]                      = res_['flux']
#                     res['snr'][i]                       = res_['s2n_r']
#                     res['e1'][i]                        = res_['pars'][2]
#                     res['e2'][i]                        = res_['pars'][3]
#                     res['cov_11'][i]                    = res_['pars_cov'][2,2]
#                     res['cov_22'][i]                    = res_['pars_cov'][3,3]
#                     res['cov_12'][i]                    = res_['pars_cov'][2,3]
#                     res['cov_21'][i]                    = res_['pars_cov'][3,2]
#                     res['hlr'][i]                       = res_['pars'][4]
#                 else:
#                     try_save = False
#             else:
#                 mask = []
#                 for flag in res_full_:
#                     if flag['flags']==0:
#                         mask.append(True)
#                     else:
#                         mask.append(False)
#                 mask = np.array(mask)
#                 res['nexp_used'][i]                 = np.sum(mask)
#                 div = 0
#                 if np.sum(mask)==0:
#                     res['flags'][i] = 999
#                 else:
#                     for j in range(len(mask)):
#                         if mask[j]:
#                             print(i,j,res_[j]['pars'][0],res_[j]['pars'][1])
#                             div                                 += w[j]
#                             res['px'][i]                        += res_[j]['pars'][0]
#                             res['py'][i]                        += res_[j]['pars'][1]
#                             res['flux'][i]                      += res_[j]['flux'] * w[j]
#                             if self.params['shape_code']=='mof':
#                                 res['snr'][i]                       = res_[j]['s2n'] * w[j]
#                             elif self.params['shape_code']=='ngmix':
#                                 res['snr'][i]                       = res_[j]['s2n_r'] * w[j]
#                             res['e1'][i]                        += res_[j]['pars'][2] * w[j]
#                             res['e2'][i]                        += res_[j]['pars'][3] * w[j]
#                             res['hlr'][i]                       += res_[j]['pars'][4] * w[j]
#                     res['px'][i]                        /= div
#                     res['py'][i]                        /= div
#                     res['flux'][i]                      /= div
#                     res['snr'][i]                       /= div
#                     res['e1'][i]                        /= div
#                     res['e2'][i]                        /= div
#                     res['hlr'][i]                       /= div

#             if try_save:
#                 mosaic = np.hstack((obs_list[i].image for i in range(len(obs_list))))
#                 psf_mosaic = np.hstack((obs_list[i].psf.image for i in range(len(obs_list))))
#                 mosaic = np.vstack((mosaic,np.hstack((obs_list[i].weight for i in range(len(obs_list))))))
#                 plt.imshow(mosaic)
#                 plt.tight_layout()
#                 plt.savefig('/users/PCON0003/cond0083/tmp_'+str(i)+'.png', bbox_inches='tight')#, dpi=400)
#                 plt.close()
#                 plt.imshow(psf_mosaic)
#                 plt.tight_layout()
#                 plt.savefig('/users/PCON0003/cond0083/tmp_psf_'+str(i)+'.png', bbox_inches='tight')#, dpi=400)
#                 plt.close()

#             out = self.measure_psf_shape_moments(psf_list)
#             mask = out['flag']==0
#             out = out[mask]
#             w = w[mask]
#             res['psf_e1'][i]        = np.average(out['e1'],weights=w)
#             res['psf_e2'][i]        = np.average(out['e2'],weights=w)
#             res['psf_T'][i]         = np.average(out['T'],weights=w)
#             if len(out)<len(obs_list):
#                 print('----------- bad psf measurement in ',i)
#             res['psf_nexp_used'][i] = len(out)

#             # obs_list = ObsList()
#             # obs_list.append(coadd[i])
#             # res_,res_full_     = self.measure_shape(obs_list,t['size'],model=self.params['ngmix_model'])

#             # res['coadd_flags'][i]                   = res_full_['flags']
#             # if res_full_['flags']==0:
#             #     res['coadd_px'][i]                  = res_['pars'][0]
#             #     res['coadd_py'][i]                  = res_['pars'][1]
#             #     res['coadd_flux'][i]                = res_['pars'][5] / wcs.pixelArea()
#             #     res['coadd_snr'][i]                 = res_['s2n']
#             #     res['coadd_e1'][i]                  = res_['pars'][2]
#             #     res['coadd_e2'][i]                  = res_['pars'][3]
#             #     res['coadd_hlr'][i]                 = res_['pars'][4]

#             # out = self.measure_psf_shape_moments([coadd[i]])
#             # if out['flag']==0:
#             #     res['coadd_psf_e1'][i]        = out['e1']
#             #     res['coadd_psf_e2'][i]        = out['e2']
#             #     res['coadd_psf_T'][i]         = out['T']
#             # else:
#             #     res['coadd_psf_e1'][i]        = -9999
#             #     res['coadd_psf_e2'][i]        = -9999
#             #     res['coadd_psf_T'][i]         = -9999

#         m.close()

#         print('done measuring',self.rank)

#         self.comm.Barrier()
#         print('after first barrier')

#         if self.rank==0:
#             for i in range(1,self.size):
#                 print('getting',i)
#                 tmp_res   = self.comm.recv(source=i)
#                 mask      = tmp_res['size']!=0
#                 res[mask] = tmp_res[mask]
#                 # coadd.update(self.comm.recv(source=i))

#             print('before barrier',self.rank)
#             self.comm.Barrier()
#             # print coadd.keys()
#             res = res[np.argsort(res['ind'])]
#             res['ra'] = np.degrees(res['ra'])
#             res['dec'] = np.degrees(res['dec'])
#             if self.shape_iter is None:
#                 ilabel = 0
#             else:
#                 ilabel = self.shape_iter
#             filename = get_filename(self.params['out_path'],
#                                 'ngmix',
#                                 self.params['output_meds'],
#                                 var=self.pointing.filter+'_'+str(self.pix)+'_'+str(ilabel),
#                                 ftype='fits',
#                                 overwrite=True)
#             fio.write(filename,res)
#             #tmp
#             # if os.path.exists(self.local_meds):
#             #     os.remove(self.local_meds)
#             #tmp


#             # m        = fio.FITS(self.local_meds,'rw')
#             # object_data = m['object_data'].read()

#             # for i in coadd:
#             #     self.dump_meds_wcs_info(object_data,
#             #                             i,
#             #                             0,
#             #                             9999,
#             #                             9999,
#             #                             9999,
#             #                             9999,
#             #                             9999,
#             #                             9999,
#             #                             coadd[i].jacobian.dudcol,
#             #                             coadd[i].jacobian.dudrow,
#             #                             coadd[i].jacobian.dvdcol,
#             #                             coadd[i].jacobian.dvdrow,
#             #                             coadd[i].jacobian.col0,
#             #                             coadd[i].jacobian.row0)

#             #     self.dump_meds_pix_info(m,
#             #                             object_data,
#             #                             i,
#             #                             0,
#             #                             coadd[i].image.flatten(),
#             #                             coadd[i].weight.flatten(),
#             #                             coadd[i].psf.image.flatten())

#             # m['object_data'].write(object_data)
#             # m.close()

#         else:

#             self.comm.send(res, dest=0)
#             res = None
#             # self.comm.send(coadd, dest=0)
#             # coadd = None
#             print('before barrier',self.rank)
#             self.comm.Barrier()

#     def cleanup(self):

#         filenames = get_filenames(self.params['out_path'],
#                                     'ngmix',
#                                     self.params['output_meds'],
#                                     var=self.pointing.filter,
#                                     ftype='fits')
#         filename = get_filename(self.params['out_path'],
#                     'ngmix',
#                     self.params['output_meds'],
#                     var=self.pointing.filter+'_combined',
#                     ftype='fits',
#                     overwrite=True)

#         length = 0
#         for f_ in filenames:
#             if length==0:
#                 tmp = fio.FITS(f_)[-1].read()
#             length += fio.FITS(f_)[-1].read_header()['NAXIS2']


#         l = 0
#         out = np.zeros(length,dtype=tmp.dtype)
#         for f_ in filenames:
#             tmp = fio.FITS(f_)[-1].read()
#             for name in tmp.dtype.names:
#                 out[name][l:l+len(tmp)] = tmp[name]
#             l+=len(tmp)

#         out = out[np.argsort(out['ind'])]

#         fio.write(filename,out)


# class wfirst_sim(object):
#     """
#     WFIRST image simulation.

#     Input:
#     param_file : File path for input yaml config file or yaml dict. Example located at: ./example.yaml.
#     """

#     def __init__(self, param_file):

#         if isinstance(param_file, string_types):
#             # Load parameter file
#             self.params     = yaml.load(open(param_file))
#             self.param_file = param_file
#             # Do some parsing
#             for key in list(self.params.keys()):
#                 if self.params[key]=='None':
#                     self.params[key]=None
#                 if self.params[key]=='none':
#                     self.params[key]=None
#                 if self.params[key]=='True':
#                     self.params[key]=True
#                 if self.params[key]=='False':
#                     self.params[key]=False
#             if 'condor' not in self.params:
#                 self.params['condor']=False

#         else:
#             # Else use existing param dict
#             self.params     = param_file

#         if 'tmpdir' in self.params:
#             os.chdir(self.params['tmpdir'])


#         # Set up some information on processes and MPI
#         if self.params['mpi']:
#             self.comm = MPI.COMM_WORLD
#             self.rank = self.comm.Get_rank()
#             self.size = self.comm.Get_size()
#             print('doing mpi')
#         else:
#             self.comm = None
#             self.rank = 0
#             self.size = 1

#         print('mpi',self.rank,self.size)

#         # Set up logger. I don't really use this, but it could be used.
#         logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
#         self.logger = logging.getLogger('wfirst_sim')

#         return

#     def setup(self,filter_,dither,sca=1,setup=False):
#         """
#         Set up initial objects.

#         Input:
#         filter_ : A filter name. 'None' to determine by dither.
#         """

#         if filter_!='None':
#             # Filter be present in filter_dither_dict{} (exists in survey strategy file).
#             if filter_ not in list(filter_dither_dict.keys()):
#                 raise ParamError('Supplied invalid filter: '+filter_)

#         # This sets up a mostly-unspecified pointing object in this filter. We will later specify a dither and SCA to complete building the pointing information.
#         if filter_=='None':
#             self.pointing = pointing(self.params,self.logger,filter_=None,sca=None,dither=None,rank=self.rank)
#         else:
#             self.pointing = pointing(self.params,self.logger,filter_=filter_,sca=None,dither=None,rank=self.rank)

#         if not setup:
#             # This updates the dither
#             self.pointing.update_dither(dither)
#             # This sets up a specific pointing for this SCA (things like WCS, PSF)
#             self.pointing.update_sca(sca)

#         self.gal_rng = galsim.UniformDeviate(self.params['random_seed'])
#         # This checks whether a truth galaxy/star catalog exist. If it doesn't exist, it is created based on specifications in the yaml file. It then sets up links to the truth catalogs on disk.
#         self.cats     = init_catalogs(self.params, self.pointing, self.gal_rng, self.rank, self.size, comm=self.comm, setup=setup)

#         if setup:
#             return False

#         if len(self.cats.gal_ind)==0:
#             print('skipping due to no objects near pointing')
#             return True

#         return False

#     def get_sca_list(self):
#         """
#         Generate list of SCAs to simulate based on input parameter file.
#         """

#         if hasattr(self.params,'sca'):
#             if self.params['sca'] is None:
#                 sca_list = np.arange(1,19)
#             elif self.params['sca'] == 'None':
#                 sca_list = np.arange(1,19)
#             elif hasattr(self.params['sca'],'__len__'):
#                 if type(self.params['sca'])==string_types:
#                     raise ParamError('Provided SCA list is not numeric.')
#                 sca_list = self.params['sca']
#             else:
#                 sca_list = [self.params['sca']]
#         else:
#             sca_list = np.arange(1,19)

#         return sca_list

#     def get_inds(self):
#         """
#         Checks things are setup, cut out objects not near SCA, and distributes objects across procs.
#         """

#         # If something went wrong and there's no pointing defined, then crash.
#         if not hasattr(self,'pointing'):
#             raise ParamError('Sim object has no pointing - need to run sim.setup() first.')
#         if self.pointing.dither is None:
#             raise ParamError('Sim pointing object has no dither assigned - need to run sim.pointing.update_dither() first.')

#         mask_sca      = self.pointing.in_sca(self.cats.gals['ra'][:],self.cats.gals['dec'][:])
#         mask_sca_star = self.pointing.in_sca(self.cats.stars['ra'][:],self.cats.stars['dec'][:])
#         mask_sca_supernova = self.pointing.in_sca(self.cats.supernovae['ra'][:],self.cats.supernovae['dec'][:])
#         self.cats.add_mask(mask_sca,star_mask=mask_sca_star,supernova_mask=mask_sca_supernova)

#     def iterate_image(self):
#         """
#         This is the main simulation. It instantiates the draw_image object, then iterates over all galaxies and stars. The output is then accumulated from other processes (if mpi is enabled), and saved to disk.
#         """
#         # Build file name path for stampe dictionary pickle
#         if 'tmpdir' in self.params:
#             filename = get_filename(self.params['tmpdir'],
#                                     'stamps',
#                                     self.params['output_meds'],
#                                     var=self.pointing.filter+'_'+str(self.pointing.dither),
#                                     name2=str(self.pointing.sca)+'_'+str(self.rank),
#                                     ftype='cPickle',
#                                     overwrite=True)
#             filename_ = get_filename(self.params['out_path'],
#                                     'stamps',
#                                     self.params['output_meds'],
#                                     var=self.pointing.filter+'_'+str(self.pointing.dither),
#                                     name2=str(self.pointing.sca)+'_'+str(self.rank),
#                                     ftype='cPickle',
#                                     overwrite=True)
#             supernova_filename = get_filename(self.params['tmpdir'],
#                                           'stamps',
#                                           self.params['output_meds'],
#                                           var=self.pointing.filter+'_'+str(self.pointing.dither),
#                                           name2=str(self.pointing.sca)+'_'+str(self.rank)+'_supernova',
#                                           ftype='cPickle',
#                                           overwrite=True)
#             supernova_filename_ = get_filename(self.params['out_path'],
#                                           'stamps',
#                                           self.params['output_meds'],
#                                           var=self.pointing.filter+'_'+str(self.pointing.dither),
#                                           name2=str(self.pointing.sca)+'_'+str(self.rank)+'_supernova',
#                                           ftype='cPickle',
#                                           overwrite=True)
#             star_filename = get_filename(self.params['tmpdir'],
#                                           'stamps',
#                                           self.params['output_meds'],
#                                           var=self.pointing.filter+'_'+str(self.pointing.dither),
#                                           name2=str(self.pointing.sca)+'_'+str(self.rank)+'_star',
#                                           ftype='cPickle',
#                                           overwrite=True)
#             star_filename_ = get_filename(self.params['out_path'],
#                                           'stamps',
#                                           self.params['output_meds'],
#                                           var=self.pointing.filter+'_'+str(self.pointing.dither),
#                                           name2=str(self.pointing.sca)+'_'+str(self.rank)+'_star',
#                                           ftype='cPickle',
#                                           overwrite=True)
#         else:
#             filename = get_filename(self.params['out_path'],
#                                     'stamps',
#                                     self.params['output_meds'],
#                                     var=self.pointing.filter+'_'+str(self.pointing.dither),
#                                     name2=str(self.pointing.sca)+'_'+str(self.rank),
#                                     ftype='cPickle',
#                                     overwrite=True)
#             filename_ = None

#             supernova_filename = get_filename(self.params['out_path'],
#                                           'stamps',
#                                           self.params['output_meds'],
#                                           var=self.pointing.filter+'_'+str(self.pointing.dither),
#                                           name2=str(self.pointing.sca)+'_'+str(self.rank)+'_supernova',
#                                           ftype='cPickle',
#                                           overwrite=True)
#             supernova_filename_ = None
            
#             star_filename = get_filename(self.params['out_path'],
#                                           'stamps',
#                                           self.params['output_meds'],
#                                           var=self.pointing.filter+'_'+str(self.pointing.dither),
#                                           name2=str(self.pointing.sca)+'_'+str(self.rank)+'_star',
#                                           ftype='cPickle',
#                                           overwrite=True)
#             star_filename_ = None

#         # Instantiate draw_image object. The input parameters, pointing object, modify_image object, truth catalog object, random number generator, logger, and galaxy & star indices are passed.
#         # Instantiation defines some parameters, iterables, and image bounds, and creates an empty SCA image.
#         self.draw_image = draw_image(self.params, self.pointing, self.modify_image, self.cats,  self.logger, rank=self.rank, comm=self.comm)

#         if self.cats.get_gal_length()!=0:#&(self.cats.get_star_length()==0):
#             # Build indexing table for MEDS making later
#             index_table = np.empty(int(self.cats.get_gal_length()),dtype=[('ind',int), ('sca',int), ('dither',int), ('x',float), ('y',float), ('ra',float), ('dec',float), ('mag',float), ('stamp',int)])
#             index_table['ind']=-999
#             # Objects to simulate
#             # Open pickler
#             with io.open(filename, 'wb') as f :
#                 i=0
#                 pickler = pickle.Pickler(f)
#                 # gals = {}
#                 # Empty storage dictionary for postage stamp information
#                 tmp,tmp_ = self.cats.get_gal_list()
#                 print('Attempting to simulate '+str(len(tmp))+' galaxies for SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.')
#                 while True:
#                     # Loop over all galaxies near pointing and attempt to simulate them.
#                     self.draw_image.iterate_gal()
#                     if self.draw_image.gal_done:
#                         break
#                     # Store postage stamp output in dictionary
#                     g_ = self.draw_image.retrieve_stamp()
#                     if g_ is not None:
#                         # gals[self.draw_image.ind] = g_
#                         if not self.params['skip_stamps']:
#                             pickler.dump(g_)
#                         index_table['ind'][i]    = g_['ind']
#                         index_table['x'][i]      = g_['x']
#                         index_table['y'][i]      = g_['y']
#                         index_table['ra'][i]     = g_['ra']
#                         index_table['dec'][i]    = g_['dec']
#                         index_table['mag'][i]    = g_['mag']
#                         if g_ is not None:
#                             index_table['stamp'][i]  = g_['stamp']
#                         else:
#                             index_table['stamp'][i]  = 0
#                         index_table['sca'][i]    = self.pointing.sca
#                         index_table['dither'][i] = self.pointing.dither
#                         i+=1
#                         g_.clear()

#         with io.open(star_filename, 'wb') as f :
#             pickler = pickle.Pickler(f)
#             tmp,tmp_ = self.cats.get_star_list()
#             if len(tmp)!=0:
#                 index_table_star = np.empty(int(self.cats.get_star_length()),dtype=[('ind',int), ('sca',int), ('dither',int), ('x',float), ('y',float), ('ra',float), ('dec',float), ('mag',float), ('stamp',int)])
#                 index_table_star['ind']=-999
#                 print('Attempting to simulate '+str(len(tmp))+' stars for SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.')
#                 i=0
#                 while True:
#                     # Loop over all stars near pointing and attempt to simulate them. Stars aren't saved in postage stamp form.
#                     self.draw_image.iterate_star()
#                     if self.draw_image.star_done:
#                         break
#                     s_ = self.draw_image.retrieve_star_stamp()
#                     if s_ is not None:
#                         pickler.dump(s_)
#                         index_table_star['ind'][i]    = s_['ind']
#                         index_table_star['x'][i]      = s_['x']
#                         index_table_star['y'][i]      = s_['y']
#                         index_table_star['ra'][i]     = s_['ra']
#                         index_table_star['dec'][i]    = s_['dec']
#                         index_table_star['mag'][i]    = s_['mag']
#                         index_table_star['sca'][i]    = self.pointing.sca
#                         index_table_star['dither'][i] = self.pointing.dither
#                         i+=1
#                         s_.clear()
        
#         with io.open(supernova_filename, 'wb') as f :
#             pickler = pickle.Pickler(f)
#             tmp,tmp_ = self.cats.get_supernova_list()
#             if len(tmp)!=0:
#                 index_table_sn = np.empty(int(self.cats.get_supernova_length()),dtype=[('ind',int), ('sca',int), ('dither',int), ('x',float), ('y',float), ('ra',float), ('dec',float), ('mag',float), ('hostid',int)])
#                 index_table_sn['ind']=-999
#                 print('Attempting to simulate '+str(len(tmp))+' supernovae for SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.')
#                 i=0
#                 while True:
#                     # Loop over all supernovae near pointing and attempt to simulate them.
#                     self.draw_image.iterate_supernova()
#                     if self.draw_image.supernova_done:
#                         break
#                     s_ = self.draw_image.retrieve_supernova_stamp()
#                     if s_ is not None:
#                         pickler.dump(s_)
#                         index_table_sn['ind'][i]    = s_['ind']
#                         index_table_sn['x'][i]      = s_['x']
#                         index_table_sn['y'][i]      = s_['y']
#                         index_table_sn['ra'][i]     = s_['ra']
#                         index_table_sn['dec'][i]    = s_['dec']
#                         index_table_sn['mag'][i]    = s_['mag']
#                         index_table_sn['sca'][i]    = self.pointing.sca
#                         index_table_sn['dither'][i] = self.pointing.dither
#                         index_table_sn['hostid'][i] = s_['hostid']
#                         i+=1
#                         s_.clear()
        
    
#         self.comm.Barrier()
#         if self.rank == 0:
#             os.system('gzip '+filename)
#             if filename_ is not None:
#                 shutil.copy(filename+'.gz',filename_+'.gz')
#             # Build file name path for SCA image
#             print(tmp_name_id)
#             filename = get_filename(self.params['out_path'],
#                                     'images',
#                                     self.params['output_meds'],
#                                     var=self.pointing.filter+'_'+str(self.pointing.dither),
#                                     name2=str(self.pointing.sca)+'_'+str(tmp_name_id),
#                                     ftype='fits.gz',
#                                     overwrite=True)

#         if self.comm is None:

#             if (self.cats.get_gal_length()==0) and (len(tmp)==0):
#                 return

#             # No mpi, so just finalize the drawing of the SCA image and write it to a fits file.
#             print('Saving SCA image to '+filename)
#             img = self.draw_image.finalize_sca()
#             write_fits(filename,img)

#         else:

#             if (self.cats.get_gal_length()==0) and (len(tmp)==0):
#                 return

#             # Send/receive all versions of SCA images across procs and sum them, then finalize and write to fits file.
#             if self.rank == 0:

#                 for i in range(1,self.size):
#                     self.draw_image.im = self.draw_image.im + self.comm.recv(source=i)
#                 print('Saving SCA image to '+filename)
#                 # self.draw_image.im.write(filename+'_raw.fits.gz')
#                 img = self.draw_image.finalize_sca()
#                 write_fits(filename,img)

#             else:

#                 self.comm.send(self.draw_image.im, dest=0)

#             # Send/receive all parts of postage stamp dictionary across procs and merge them.
#             # if self.rank == 0:

#             #     for i in range(1,self.size):
#             #         gals.update( self.comm.recv(source=i) )

#             #     # Build file name path for stampe dictionary pickle
#             #     filename = get_filename(self.params['out_path'],
#             #                             'stamps',
#             #                             self.params['output_meds'],
#             #                             var=self.pointing.filter+'_'+str(self.pointing.dither),
#             #                             name2=str(self.pointing.sca),
#             #                             ftype='cPickle',
#             #                             overwrite=True)

#             #     if gals!={}:
#             #         # Save stamp dictionary pickle
#             #         print('Saving stamp dict to '+filename)
#             #         save_obj(gals, filename )

#             # else:

#             #     self.comm.send(gals, dest=0)

#         if self.rank == 0:

#             filename = get_filename(self.params['out_path'],
#                                     'truth',
#                                     self.params['output_meds'],
#                                     var='index',
#                                     name2=self.pointing.filter+'_'+str(self.pointing.dither)+'_'+str(self.pointing.sca),
#                                     ftype='fits',
#                                     overwrite=True)
#             filename_star = get_filename(self.params['out_path'],
#                                     'truth',
#                                     self.params['output_meds'],
#                                     var='index',
#                                     name2=self.pointing.filter+'_'+str(self.pointing.dither)+'_'+str(self.pointing.sca)+'_star',
#                                     ftype='fits',
#                                     overwrite=True)
#             filename_sn = get_filename(self.params['out_path'],
#                                     'truth',
#                                     self.params['output_meds'],
#                                     var='index',
#                                     name2=self.pointing.filter+'_'+str(self.pointing.dither)+'_'+str(self.pointing.sca)+'_sn',
#                                     ftype='fits',
#                                     overwrite=True)  
#             print('before index')
#             index_table = index_table[index_table['ind']>-999]
#             index_table_star = index_table_star[index_table_star['ind']>-999]
#             index_table_sn = index_table_sn[index_table_sn['ind']>-999]
#             print('Saving index to '+filename)
#             fio.write(filename,index_table)
#             fio.write(filename_star,index_table_star)
#             fio.write(filename_sn,index_table_sn)

#     def check_file(self,sca,dither,filter_):
#         self.pointing = pointing(self.params,self.logger,filter_=None,sca=None,dither=int(dither),rank=self.rank)
#         print(sca,dither,filter_)
#         f = get_filename(self.params['out_path'],
#                                     'truth',
#                                     self.params['output_meds'],
#                                     var='index',
#                                     name2=self.pointing.filter+'_'+str(dither)+'_'+str(sca),
#                                     ftype='fits',
#                                     overwrite=False)
#         print(f)
#         return os.path.exists(f)

# def condor_cleanup(out_path):

#     import tarfile

#     tar = tarfile.open('output.tar', 'w:gz')
#     tar.add(out_path, arcname='output.tar')
#     tar.close()

# def syntax_proc():

#     print('Possible syntax for running: ')
#     print('')
#     print('To set up truth catalog (must be run before any other modes):')
#     print('    python simulate.py <yaml settings file> <filter> setup')
#     print('')
#     print('To run in image simulation mode: ')
#     print('    python simulate.py <yaml settings file> <filter> <dither id> [verify string]')
#     print('')
#     print('To set up index information for meds making mode (must be run before attempting meds making): ')
#     print('    python simulate.py <yaml settings file> <filter> meds setup')
#     print('')
#     print('To create a meds file and run shape measurement on it: ')
#     print('    python simulate.py <yaml settings file> <filter> meds <pixel id>')
#     print('')
#     print('To cleanup meds/shape run and concatenate output files: ')
#     print('    python simulate.py <yaml settings file> <filter> meds cleanup')
#     print('')
#     print('')
#     print('Value definitions: ')
#     print('yaml settings file : A yaml file with settings for the simulation run.')
#     print('filter : Filter name. Either one of the keys of filter_dither_dict or None. None will interpret the filter from the dither simulation file. A string filter name will override the appropriate filter from the dither simulation.')
#     print('dither id : An integer dither identifier. Either an index into the dither simulation file or an integer specifying a line from a provided file listing indices into the dither simulation file (useful for setting up parallel runs).')
#     print("""verify string : A string 'verify_output'. Reruns simulation mode checking for failed runs and filling in missing files. Will only recalculate necessary missing files.""")
#     print('pixel id : Healpix id for MEDS generation. Each MEDS file corresponds to a healpixel on the sky with nside defined in the yaml settings file. Can be either a healpixel index or an integer specifying a line from a provided file listing potential healpix indices (useful for setting up parallel runs).')
#     sys.exit()

# Uncomment for profiling
# pr = cProfile.Profile()

if __name__ == "__main__":
    """
    """

    # Uncomment for profiling
    # pr.enable()

    t0 = time.time()

    # process = psutil.Process(os.getpid())
    # print(process.memory_info().rss/2**30)
    # print(process.memory_info().vms/2**30)

    try:
        param_file = sys.argv[1]
        filter_ = sys.argv[2]
        dither = sys.argv[3]
    except:
        syntax_proc()

    if (param_file.lower() == 'help') or (filter_.lower()=='help') or (dither.lower()=='help'):
        syntax_proc()

    # This instantiates the simulation based on settings in input param file
    sim = wfirst_imsim.wfirst_sim(param_file)
    print(sim.params.keys())

    if sim.params['condor']==True:
        condor=True
    else:
        condor=False

    # This sets up some things like input truth catalogs and empty objects
    if dither=='setup':
        sim.setup(filter_,dither,setup=True)
    elif dither=='meds':
        condor_build = False
        if len(sys.argv)<5:
            syntax_proc()
        if sys.argv[4]=='setup':
            setup = True
            pix = -1
        elif sys.argv[4]=='condor_build':
            condor_build = True
            setup = False
            pix = -1
        elif sys.argv[4]=='cleanup':
            setup = True
            pix = -1
            m = accumulate_output_disk( param_file, filter_, pix, sim.comm, ignore_missing_files = False, setup = setup )
            m.cleanup()
        elif sys.argv[4]=='shape':
            print(sys.argv)
            if (sim.params['meds_from_file'] is not None) & (sim.params['meds_from_file'] != 'None'):
                pix = int(np.loadtxt(sim.params['meds_from_file'])[int(sys.argv[5])-1])
            else:
                pix = int(sys.argv[5])
            m = wfirst_imsim.accumulate_output_disk( param_file, filter_, pix, sim.comm, shape=True, shape_iter = int(sys.argv[6]), shape_cnt = int(sys.argv[7]))
            #m.get_coadd_shape_mcal() 
            #m.get_coadd_shape_coadd()
            m.get_coadd_shape_multiband_coadd()
            print('out of coadd_shape')
            del(m)
            sys.exit()
        else:
            setup = False
            if (sim.params['meds_from_file'] is not None) & (sim.params['meds_from_file'] != 'None'):
                pix = int(np.loadtxt(sim.params['meds_from_file'])[int(sys.argv[4])-1])
            else:
                pix = int(sys.argv[4])
        m = wfirst_imsim.accumulate_output_disk( param_file, filter_, pix, sim.comm, ignore_missing_files = False, setup = setup, condor_build = condor_build)
        if not setup:
        	m.finish(condor=sim.params['condor'])
        	sys.exit()

        if setup or condor_build:
            print('exiting')
            sys.exit()
        m.comm.Barrier()
        skip = False
        if sim.rank==0:
            for i in range(1,sim.size):
                m.comm.send(m.skip, dest=i)
            skip = m.skip
        else:
            skip = m.comm.recv(source=0)
        if not skip:
            m.comm.Barrier()
            if not condor:
                #m.get_coadd_shape_mcal()
            	#m.get_coadd_shape_coadd()
            	m.get_coadd_shape_multiband_coadd()
            print('out of coadd_shape')
            # print 'commented out finish()'
            m.finish(condor=sim.params['condor'])
            # pr.disable()
            # ps = pstats.Stats(pr).sort_stats('time')
            # ps.print_stats(200)
            del(m)

    else:

        if (sim.params['dither_from_file'] is not None) & (sim.params['dither_from_file'] != 'None'):
            dither=np.loadtxt(sim.params['dither_from_file'])[int(dither)-1] # Assumes array starts with 1
        print(sys.argv)
        sca = int(sys.argv[4])
        if 'verify_output' in sys.argv:
            if sim.check_file(sca,int(dither),filter_):
                print('exists',dither,sca)
                sys.exit()
        skip = sim.setup(filter_,int(dither),sca=sca)
        print(skip)
        #tmp_name_id = int(sys.argv[6])

        # Loop over SCAs
        print('before barrier', sca)
        sim.comm.Barrier()
        print('after barrier', sca)
        # This sets up the object that will simulate various wfirst detector effects, noise, etc. Instantiation creates a noise realisation for the image.
        sim.modify_image = wfirst_imsim.modify_image(sim.params)
        print('modified image', sca)
        # This is the main thing - iterates over galaxies for a given pointing and SCA and simulates them all
        sim.comm.Barrier()
        print(time.time()-t0)
        # print(process.memory_info().rss/2**30)
        # print(process.memory_info().vms/2**30)
        sim.iterate_image()
        sim.comm.Barrier()

        # Uncomment for profiling
        # pr.disable()
        # ps = pstats.Stats(pr).sort_stats('time')
        # ps.print_stats(50)

    # if sim.params['condor']==True:
    #     condor_cleanup(sim.params['out_path'])

# test round galaxy recovered to cover wcs errors

# same noise in different runs? same noise
# distance to edge to reject images?
