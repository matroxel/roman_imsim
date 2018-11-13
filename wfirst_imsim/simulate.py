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

import numpy as np
import healpy as hp
import sys, os
import math
import logging
import time
import yaml
import galsim as galsim
import galsim.wfirst as wfirst
import galsim.config.process as process
import galsim.des as des
import ngmix
from ngmix.observation import Observation, ObsList
from ngmix.jacobian import Jacobian
import fitsio as fio
import cPickle as pickle
from astropy.time import Time
from mpi4py import MPI
from mpi_pool import MPIPool
import cProfile, pstats

path, filename = os.path.split(__file__)
sedpath_Star   = os.path.join(galsim.meta_data.share_dir, 'SEDs', 'vega.txt')

if sys.version_info[0] == 3:
    string_types = str,
else:
    string_types = basestring,

# Chip coordinates
cptr = np.array([
0.002689724,  1.000000000,  0.181995021, -0.002070809, -1.000000000,  0.807383134,  1.000000000,  0.004769437,  1.028725015, -1.000000000, -0.000114163, -0.024579913,
0.003307633,  1.000000000,  1.203503349, -0.002719257, -1.000000000, -0.230036847,  1.000000000,  0.006091805,  1.028993582, -1.000000000, -0.000145757, -0.024586416,
0.003888409,  1.000000000,  2.205056241, -0.003335597, -1.000000000, -1.250685466,  1.000000000,  0.007389324,  1.030581048, -1.000000000, -0.000176732, -0.024624426,
0.007871078,  1.000000000, -0.101157485, -0.005906926, -1.000000000,  1.095802866,  1.000000000,  0.009147586,  2.151242511, -1.000000000, -0.004917673, -1.151541644,
0.009838715,  1.000000000,  0.926774753, -0.007965112, -1.000000000,  0.052835488,  1.000000000,  0.011913584,  2.150981875, -1.000000000, -0.006404157, -1.151413352,
0.011694346,  1.000000000,  1.935534773, -0.009927853, -1.000000000, -0.974276664,  1.000000000,  0.014630945,  2.153506744, -1.000000000, -0.007864196, -1.152784334,
0.011758070,  1.000000000, -0.527032681, -0.008410887, -1.000000000,  1.529873670,  1.000000000,  0.012002262,  3.264990040, -1.000000000, -0.008419930, -2.274065453,
0.015128555,  1.000000000,  0.510881058, -0.011918799, -1.000000000,  0.478274989,  1.000000000,  0.016194244,  3.262719942, -1.000000000, -0.011359106, -2.272508364,
0.018323436,  1.000000000,  1.530828790, -0.015281655, -1.000000000, -0.558879607,  1.000000000,  0.020320244,  3.264721809, -1.000000000, -0.014251259, -2.273955111,
-0.002689724,  1.000000000,  0.181995021,  0.002070809, -1.000000000,  0.807383134,  1.000000000, -0.000114163, -0.024579913, -1.000000000,  0.004769437,  1.028725015,
-0.003307633,  1.000000000,  1.203503349,  0.002719257, -1.000000000, -0.230036847,  1.000000000, -0.000145757, -0.024586416, -1.000000000,  0.006091805,  1.028993582,
-0.003888409,  1.000000000,  2.205056241,  0.003335597, -1.000000000, -1.250685466,  1.000000000, -0.000176732, -0.024624426, -1.000000000,  0.007389324,  1.030581048,
-0.007871078,  1.000000000, -0.101157485,  0.005906926, -1.000000000,  1.095802866,  1.000000000, -0.004917673, -1.151541644, -1.000000000,  0.009147586,  2.151242511,
-0.009838715,  1.000000000,  0.926774753,  0.007965112, -1.000000000,  0.052835488,  1.000000000, -0.006404157, -1.151413352, -1.000000000,  0.011913584,  2.150981875,
-0.011694346,  1.000000000,  1.935534773,  0.009927853, -1.000000000, -0.974276664,  1.000000000, -0.007864196, -1.152784334, -1.000000000,  0.014630945,  2.153506744,
-0.011758070,  1.000000000, -0.527032681,  0.008410887, -1.000000000,  1.529873670,  1.000000000, -0.008419930, -2.274065453, -1.000000000,  0.012002262,  3.264990040,
-0.015128555,  1.000000000,  0.510881058,  0.011918799, -1.000000000,  0.478274989,  1.000000000, -0.011359106, -2.272508364, -1.000000000,  0.016194244,  3.262719942,
-0.018323436,  1.000000000,  1.530828790,  0.015281655, -1.000000000, -0.558879607,  1.000000000, -0.014251259, -2.273955111, -1.000000000,  0.020320244,  3.264721809 ])

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


class ParamError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

def except_func(logger, proc, k, res, t):
    print proc, k
    print t
    raise res

def save_obj(obj, name ):
    """
    Helper function to save some data as a pickle to disk.
    """
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    """
    Helper function to read some data from a pickle on disk.
    """
    with open(name, 'rb') as f:
        return pickle.load(f)

def convert_dither_to_fits(ditherfile='observing_sequence_hlsonly'):
    """
    Helper function to used to convert Chris survey dither file to fits and extract HLS part.
    """

    dither = np.genfromtxt(ditherfile+'.dat',dtype=None,names = ['date','f1','f2','ra','dec','pa','program','filter','f8','f9','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21'])
    dither = dither[['date','ra','dec','pa','filter']][dither['program']==5]
    fio.write(ditherfile+'.fits',dither,clobber=True)

    return

def convert_gaia():
    """
    Helper function to convert gaia data to star truth catalog.
    """

    n=100000000
    ra=[-5,80.]
    dec=[-64,3]
    ra1 = np.random.rand(n)*(ra[1]-ra[0])/180.*np.pi+ra[0]/180.*np.pi
    d0 = (np.cos((dec[0]+90)/180.*np.pi)+1)/2.
    d1 = (np.cos((dec[1]+90)/180.*np.pi)+1)/2.
    dec1 = np.arccos(2*np.random.rand(n)*(d1-d0)+2*d0-1)
    out = np.empty(n,dtype=[('ra',float)]+[('dec',float)]+[('H158',float)]+[('J129',float)]+[('Y106',float)]+[('F184',float)])
    out['ra']=ra1
    out['dec']=dec1-np.pi/2.

    g_band     = galsim.Bandpass('/users/PCON0003/cond0083/GalSim/galsim/share/bandpasses/gaia_g.dat', wave_type='nm').withZeropoint('AB')
    star_sed   = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')

    gaia = fio.FITS('../distwf-result.fits.gz')[-1].read()['phot_g_mean_mag'][:]
    h,b = np.histogram(gaia,bins=np.linspace(3,22.5,196))
    b = (b[1:]+b[:-1])/2
    x = np.random.choice(np.arange(len(b)),len(out),p=1.*h/np.sum(h),replace=True)
    for i,filter_ in enumerate(['J129','F184','Y106','H158']):
        print filter_
        bpass = wfirst.getBandpasses(AB_zeropoint=True)[filter_]
        b_=np.zeros(len(b))
        for ind in range(len(b)):
            star_sed_  = star_sed.withMagnitude(b[ind],g_band)
            b_[ind]    = star_sed_.calculateMagnitude(bpass)
        out[filter_]   = b_[x]

    fio.write('gaia_stars.fits',out,clobber=True)

    return

def convert_galaxia():
    """
    Helper function to convert galaxia data to star truth catalog.
    """

    j_band     = galsim.Bandpass('/users/PCON0003/cond0083/GalSim/galsim/share/bandpasses/UKIRT_UKIDSS.J.dat.txt', wave_type='nm').withZeropoint('AB')
    star_sed   = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')

    g = fio.FITS('/users/PCON0003/cond0083/galaxia_stars.fits')[-1].read()
    out = np.empty(len(g),dtype=[('ra',float)]+[('dec',float)]+[('H158',float)]+[('J129',float)]+[('Y106',float)]+[('F184',float)])
    out['ra']=g['ra']
    out['dec']=g['dec']
    for i,filter_ in enumerate(['J129','F184','Y106','H158']):
        print filter_
        bpass = wfirst.getBandpasses(AB_zeropoint=True)[filter_]
        star_sed_  = star_sed.withMagnitude(23,j_band)
        factor    = star_sed_.calculateMagnitude(bpass)-23
        out[filter_] = g['J']+factor

    s = np.random.choice(np.arange(len(out)),len(out),replace=False)
    out=out[s]
    fio.write('galaxia_stars_full.fits',out,clobber=True)

    return

def create_radec_fits(ra=[25.,27.5],dec=[-27.5,-25.],n=1500000):
    """
    Helper function that just creates random positions within some ra,dec range.
    """

    ra1 = np.random.rand(n)*(ra[1]-ra[0])/180.*np.pi+ra[0]/180.*np.pi
    d0 = (np.cos((dec[0]+90)/180.*np.pi)+1)/2.
    d1 = (np.cos((dec[1]+90)/180.*np.pi)+1)/2.
    dec1 = np.arccos(2*np.random.rand(n)*(d1-d0)+2*d0-1)
    out = np.empty(n,dtype=[('ra',float)]+[('dec',float)])
    out['ra']=ra1*180./np.pi
    out['dec']=dec1*180./np.pi-90
    fio.write('ra_'+str(ra[0])+'_'+str(ra[1])+'_dec_'+str(dec[0])+'_'+str(dec[1])+'_n_'+str(n)+'.fits.gz',out,clobber=True)

def hsm(im, psf=None, wt=None):
    """
    Not used currently, but this is a helper function to run hsm via galsim.
    """

    out = np.zeros(1,dtype=[('e1','f4')]+[('e2','f4')]+[('T','f4')]+[('dx','f4')]+[('dy','f4')]+[('flag','i2')])
    try:
        if psf is not None:
            shape_data = galsim.hsm.EstimateShear(im, psf, weight=wt, strict=False)
        else:
            shape_data = im.FindAdaptiveMom(weight=wt, strict=False)
    except:
        # print(' *** Bad measurement (caught exception).  Mask this one.')
        out['flag'] |= BAD_MEASUREMENT
        return out

    if shape_data.moments_status != 0:
        # print('status = ',shape_data.moments_status)
        # print(' *** Bad measurement.  Mask this one.')
        out['flag'] |= BAD_MEASUREMENT

    out['dx'] = shape_data.moments_centroid.x - im.trueCenter().x
    out['dy'] = shape_data.moments_centroid.y - im.trueCenter().y
    if out['dx']**2 + out['dy']**2 > MAX_CENTROID_SHIFT**2:
        # print(' *** Centroid shifted by ',out['dx'],out['dy'],'.  Mask this one.')
        out['flag'] |= CENTROID_SHIFT

    # Account for the image wcs
    if im.wcs.isPixelScale():
        out['e1'] = shape_data.observed_shape.g1
        out['e2'] = shape_data.observed_shape.g2
        out['T']  = 2 * shape_data.moments_sigma**2 * im.scale**2
    else:
        e1 = shape_data.observed_shape.e1
        e2 = shape_data.observed_shape.e2
        s = shape_data.moments_sigma

        jac = im.wcs.jacobian(im.trueCenter())
        M = np.matrix( [[ 1 + e1, e2 ], [ e2, 1 - e1 ]] ) * s*s
        J = jac.getMatrix()
        M = J * M * J.T
        scale = np.sqrt(M/2./s/s)

        e1 = (M[0,0] - M[1,1]) / (M[0,0] + M[1,1])
        e2 = (2.*M[0,1]) / (M[0,0] + M[1,1])
        out['T'] = M[0,0] + M[1,1]

        shear = galsim.Shear(e1=e1, e2=e2)
        out['e1'] = shear.g1
        out['e2'] = shear.g2

    return out

def reset_rng(self, seed):
    """
    Reset the (pseudo-)random number generators.

    Input
    self  : object
    iter  : value to iterate 
    """

    self.rng     = galsim.BaseDeviate(seed)
    self.gal_rng = galsim.UniformDeviate(seed)

    return

def get_filename( out_path, path, name, var=None, name2=None, ftype='fits', overwrite=False ):
    """
    Helper function to set up a file path, and create the path if it doesn't exist.
    """

    if var is not None:
        name += '_' + var
    if name2 is not None:
        name += '_' + name2
    name += '.' + ftype

    fpath = os.path.join(out_path,path)

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(fpath):
        os.mkdir(fpath)

    filename = os.path.join(fpath,name)
    if (overwrite)&(os.path.exists(filename)):
        os.remove(filename)

    return filename

def get_filenames( out_path, path, name, var=None, name2=None, ftype='fits' ):
    """
    Helper function to set up a file path, and create the path if it doesn't exist.
    """

    if var is not None:
        name += '_' + var
    if name2 is not None:
        name += '_' + name2
    name += '*.' + ftype

    fpath = os.path.join(out_path,path)

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(fpath):
        os.mkdir(fpath)

    filename = os.path.join(fpath,name)

    return glob.glob(filename)

class pointing():
    """
    Class to manage and hold informaiton about a wfirst pointing, including WCS and PSF.
    """


    def __init__(self, params, logger, filter_=None, sca=None, dither=None, sca_pos=None, max_rad_from_boresight=0.009, chip_enlarge=0.01):
        """
        Initializes some information about a pointing.

        Input
        params                  : Parameter dict.
        logger                  : logger instance
        filter_                 : The filter name for this pointing.
        sca                     : The SCA number (1-18)
        dither                  : The index of this pointing in the survey simulation file.
        sca_pos                 : Used to simulate the PSF at a position other than the center 
                                    of the SCA.
        max_rad_from_boresight  : Distance around pointing to attempt to simulate objects.
        chip_enlarge            : Factor to enlarge chip geometry by to account for small 
                                    inaccuracies relative to precise WCS.
        """

        self.ditherfile         = params['dither_file']
        self.n_waves            = params['n_waves'] # Number of wavelenghts of PSF to simulate
        self.approximate_struts = params['approximate_struts'] # Whether to approsimate struts
        self.extra_aberrations  = params['extra_aberrations']  # Extra aberrations to include in the PSF model. See galsim documentation.
        self.logger = logger
        self.sca    = None
        self.PSF    = None
        self.WCS    = None
        self.dither = None

        if filter_ is not None:
            self.get_bpass(filter_)

        if sca is not None:
            self.update_sca(sca,sca_pos=sca_pos)

        if dither is not None:
            self.update_dither(dither)

        self.bore           = max_rad_from_boresight
        self.sbore2         = np.sin(max_rad_from_boresight/2.)
        self.chip_enlarge   = chip_enlarge

    def get_bpass(self, filter_):
        """
        Read in the WFIRST filters, setting an AB zeropoint appropriate for this telescope given its
        diameter and (since we didn't use any keyword arguments to modify this) using the typical
        exposure time for WFIRST images.  By default, this routine truncates the parts of the
        bandpasses that are near 0 at the edges, and thins them by the default amount.

        Input
        filter_ : Fiter name for this pointing.
        """

        self.filter = filter_
        self.bpass  = wfirst.getBandpasses(AB_zeropoint=True)[self.filter]

    def update_dither(self,dither):
        """
        This updates the pointing to a new dither position.

        Input
        dither     : Pointing index in the survey simulation file.
        sca        : SCA number
        """

        self.dither = dither

        d = fio.FITS(self.ditherfile)[-1][self.dither]

        # Check that nothing went wrong with the filter specification.
        # if filter_dither_dict[self.filter] != d['filter']:
        #     raise ParamError('Requested filter and dither pointing do not match.')

        self.ra     = d['ra'][0]  * np.pi / 180. # RA of pointing
        self.dec    = d['dec'][0] * np.pi / 180. # Dec of pointing
        self.pa     = d['pa'][0]  * np.pi / 180.  # Position angle of pointing
        self.sdec   = np.sin(self.dec) # Here and below - cache some geometry stuff
        self.cdec   = np.cos(self.dec)
        self.sra    = np.sin(self.ra)
        self.cra    = np.cos(self.ra)
        self.spa    = np.sin(self.pa)
        self.cpa    = np.cos(self.pa)
        self.date   = Time(d['date'][0],format='mjd').datetime # Date of pointing

    def update_sca(self,sca):
        """
        This assigns an SCA to the pointing, and evaluates the PSF and WCS.

        Input
        dither     : Pointing index in the survey simulation file.
        sca        : SCA number
        """

        self.sca    = sca
        self.get_wcs() # Get the new WCS
        self.get_psf() # Get the new PSF
        radec           = self.WCS.toWorld(galsim.PositionI(wfirst.n_pix/2,wfirst.n_pix/2))
        self.sca_sdec   = np.sin(radec.dec) # Here and below - cache some geometry stuff
        self.sca_cdec   = np.cos(radec.dec)
        self.sca_sra    = np.sin(radec.ra)
        self.sca_cra    = np.cos(radec.ra)

    def get_psf(self, sca_pos=None, high_accuracy=False):
        """
        This updates the pointing to a new SCA, replacing the stored PSF to the new SCA.

        Input
        sca_pos : Used to simulate the PSF at a position other than the center of the SCA.
        """

        self.PSF = wfirst.getPSF(self.sca,
                                self.filter,
                                SCA_pos             = sca_pos, # - in branch 919
                                approximate_struts  = self.approximate_struts, 
                                n_waves             = self.n_waves, 
                                logger              = self.logger, 
                                wavelength          = self.bpass.effective_wavelength,
                                extra_aberrations   = self.extra_aberrations,
                                high_accuracy       = high_accuracy,
                                )

        # sim.logger.info('Done PSF precomputation in %.1f seconds!'%(time.time()-t0))

    def get_wcs(self):
        """
        Get the WCS for an observation at this position. We are not supplying a date, so the routine will assume it's the vernal equinox. The output of this routine is a dict of WCS objects, one for each SCA. We then take the WCS for the SCA that we are using.
        """

        self.WCS = wfirst.getWCS(world_pos  = galsim.CelestialCoord(ra=self.ra*galsim.radians, \
                                                                    dec=self.dec*galsim.radians), 
                                PA          = self.pa*galsim.radians, 
                                date        = self.date,
                                SCAs        = self.sca,
                                PA_is_FPA   = True
                                )[self.sca]

    def in_sca(self, ra, dec):
        """
        Check if ra, dec falls on approximate SCA area.

        Input
        ra  : Right ascension of object
        dec : Declination of object
        """

        # Catch some problems, like the pointing not being defined
        if self.dither is None:
            raise ParamError('No dither defined to check ra, dec against.')

        if self.sca is None:
            raise ParamError('No sca defined to check ra, dec against.')

        # # Discard any object greater than some dec from pointing
        # if np.abs(dec-self.dec)>self.bore:
        #     return False

        # Position of the object in boresight coordinates
        mX  = -self.sdec   * np.cos(dec) * np.cos(self.ra-ra) + self.cdec * np.sin(dec)
        mY  =  np.cos(dec) * np.sin(self.ra-ra)

        xi  = -(self.spa * mX + self.cpa * mY) / 0.0021801102 # Image plane position in chips
        yi  =  (self.cpa * mX - self.spa * mY) / 0.0021801102

        # Check if object falls on SCA
        if hasattr(ra,'__len__'):
            return   np.where((cptr[0+12*(self.sca-1)]*xi+cptr[1+12*(self.sca-1)]*yi  \
                                <cptr[2+12*(self.sca-1)]+self.chip_enlarge)       \
                            & (cptr[3+12*(self.sca-1)]*xi+cptr[4+12*(self.sca-1)]*yi  \
                                <cptr[5+12*(self.sca-1)]+self.chip_enlarge)       \
                            & (cptr[6+12*(self.sca-1)]*xi+cptr[7+12*(self.sca-1)]*yi  \
                                <cptr[8+12*(self.sca-1)]+self.chip_enlarge)       \
                            & (cptr[9+12*(self.sca-1)]*xi+cptr[10+12*(self.sca-1)]*yi \
                                <cptr[11+12*(self.sca-1)]+self.chip_enlarge))[0]

        if    (cptr[0+12*(self.sca-1)]*xi+cptr[1+12*(self.sca-1)]*yi  \
                <cptr[2+12*(self.sca-1)]+self.chip_enlarge)       \
            & (cptr[3+12*(self.sca-1)]*xi+cptr[4+12*(self.sca-1)]*yi  \
                <cptr[5+12*(self.sca-1)]+self.chip_enlarge)       \
            & (cptr[6+12*(self.sca-1)]*xi+cptr[7+12*(self.sca-1)]*yi  \
                <cptr[8+12*(self.sca-1)]+self.chip_enlarge)       \
            & (cptr[9+12*(self.sca-1)]*xi+cptr[10+12*(self.sca-1)]*yi \
                <cptr[11+12*(self.sca-1)]+self.chip_enlarge):

            return True

        return False


    def near_pointing(self, ra, dec, sca=False):
        """
        Returns objects close to pointing, using usual orthodromic distance.

        Input
        ra  : Right ascension array of objects
        dec : Declination array of objects
        """

        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)

        if sca:
            d2 = (x - self.sca_cdec*self.sca_cra)**2 + (y - self.sca_cdec*self.sca_sra)**2 + (z - self.sca_sdec)**2
        else:
            d2 = (x - self.cdec*self.cra)**2 + (y - self.cdec*self.sra)**2 + (z - self.sdec)**2

        return np.where(np.sqrt(d2)/2.<=self.sbore2)[0]

class init_catalogs():
    """
    Build truth catalogs if they don't exist from input galaxy and star catalogs.
    """


    def __init__(self, params, pointing, gal_rng, rank, size, comm=None, setup=False):
        """
        Initiate the catalogs

        Input
        params   : Parameter dictionary
        pointing : Pointing object
        gal_rng  : Random generator [0,1]
        rank     : Process rank
        comm     : MPI comm object
        """

        if rank == 0:
            # Set up file path. Check if output truth file path exists or if explicitly remaking galaxy properties
            filename = get_filename(params['out_path'],
                                    'truth',
                                    params['output_truth'],
                                    name2='truth_gal',
                                    overwrite=params['overwrite'])

            # Link to galaxy truth catalog on disk 
            self.gals  = self.init_galaxy(filename,params,pointing,gal_rng,setup)
            # Link to star truth catalog on disk 
            self.stars = self.init_star(params)

            if setup:
                comm.Barrier()
                return

            if comm is not None:
                # Pass gal_ind to other procs
                self.gal_ind  = pointing.near_pointing( self.gals['ra'][:], self.gals['dec'][:] )
                self.gals = self.gals[self.gal_ind]
                print len(self.gal_ind)
                for i in range(1,size):
                    comm.send(self.gal_ind,  dest=i)
                    comm.send(self.gals,  dest=i)

                self.star_ind = pointing.near_pointing( self.stars['ra'][:], self.stars['dec'][:] )
                self.stars = self.stars[self.star_ind]
                # Pass star_ind to other procs
                for i in range(1,size):
                    comm.send(self.star_ind,  dest=i)
                    comm.send(self.stars,  dest=i)

        else:
            if setup:
                comm.Barrier()
                return

            # Get gals
            self.gal_ind = comm.recv(source=0)
            self.gals = comm.recv(source=0)

            # Get stars
            self.star_ind = comm.recv(source=0)
            self.stars = comm.recv(source=0)


    def add_mask(self,gal_mask,star_mask=None):

        if gal_mask.dtype == bool:
            self.gal_mask = np.where(gal_mask)[0]
        else:
            self.gal_mask = gal_mask

        if star_mask is None:
            self.star_mask = []
        elif star_mask.dtype == bool:
            self.star_mask = np.where(star_mask)[0]
        else:
            self.star_mask = star_mask

    def get_gal_length(self):

        return len(self.gal_mask)

    def get_star_length(self):

        return len(self.star_mask)

    def get_gal_list(self):

        return self.gal_ind[self.gal_mask],self.gals[self.gal_mask]

    def get_star_list(self):

        return self.star_ind[self.star_mask],self.stars[self.star_mask]

    def get_gal(self,ind):

        return self.gal_ind[self.gal_mask[ind]],self.gals[self.gal_mask[ind]]

    def get_star(self,ind):

        return self.star_ind[self.star_mask[ind]],self.stars[self.star_mask[ind]]

    def dump_truth_gal(self,filename,store):
        """
        Write galaxy truth catalog to disk.

        Input
        filename    : Fits filename
        store       : Galaxy truth catalog
        """

        fio.write(filename,store,clobber=True)

        return fio.FITS(filename)[-1]

    def load_truth_gal(self,filename):
        """
        Load galaxy truth catalog from disk.

        Input
        filename    : Fits filename
        """

        store = fio.FITS(filename)[-1]

        return store

    def fwhm_to_hlr(self,fwhm):
        """
        Convert full-width half-maximum to half-light radius in units of arcseconds.

        Input
        fwhm : full-width half-maximum
        """

        radius = fwhm * 0.06 / 2. # 1 pix = 0.06 arcsec, factor 2 to convert to hlr

        return radius

    def init_galaxy(self,filename,params,pointing,gal_rng,setup):
        """
        Does the work to return a random, unique object property list (truth catalog). 

        Input
        filname  : Filename of galaxy truth catalog.
        params   : Parameter dict
        pointing : pointing object
        gal_rng  : Random generator [0,1]
        """

        # Make sure galaxy distribution filename is well-formed and link to it
        if isinstance(params['gal_dist'],string_types):
            # Provided an ra,dec catalog of object positions.
            radec_file = fio.FITS(params['gal_dist'])[-1]
        else:
            raise ParamError('Bad gal_dist filename.')

        # This is a placeholder option to allow different galaxy simulatin methods later if necessary
        if params['gal_type'] == 0:
            # Analytic profile - sersic disk


            if not setup:
                if os.path.exists(filename):
                    # Truth file exists and no instruction to overwrite it, so load existing truth file with galaxy properties
                    return self.load_truth_gal(filename)
                else:
                    raise ParamError('No truth file to load.')

            if (not params['overwrite']) and (os.path.exists(filename)):
                print 'Reusing existing truth file.'
                return None

            print '-----building truth catalog------'
            # Read in file with photometry/size/redshift distribution similar to WFIRST galaxies
            phot       = fio.FITS(params['gal_sample'])[-1].read(columns=['fwhm','redshift',filter_flux_dict['J129'],filter_flux_dict['F184'],filter_flux_dict['Y106'],filter_flux_dict['H158']])
            pind_list_ = np.ones(len(phot)).astype(bool) # storage list for original index of photometry catalog
            pind_list_ = pind_list_&(phot[filter_flux_dict['J129']]<99)&(phot[filter_flux_dict['J129']]>0) # remove bad mags
            pind_list_ = pind_list_&(phot[filter_flux_dict['F184']]<99)&(phot[filter_flux_dict['F184']]>0) # remove bad mags
            pind_list_ = pind_list_&(phot[filter_flux_dict['Y106']]<99)&(phot[filter_flux_dict['Y106']]>0) # remove bad mags
            pind_list_ = pind_list_&(phot[filter_flux_dict['H158']]<99)&(phot[filter_flux_dict['H158']]>0) # remove bad mags
            pind_list_ = pind_list_&(phot['redshift']>0)&(phot['redshift']<5) # remove bad redshifts
            pind_list_ = np.where(pind_list_)[0]

            n_gal = radec_file.read_header()['NAXIS2']

            # Create minimal storage array for galaxy properties
            store = np.ones(n_gal, dtype=[('gind','i4')]
                                        +[('ra',float)]
                                        +[('dec',float)]
                                        +[('g1','f4')]
                                        +[('g2','f4')]
                                        +[('rot','f4')]
                                        +[('size','f4')]
                                        +[('z','f4')]
                                        +[('J129','f4')]
                                        +[('F184','f4')]
                                        +[('Y106','f4')]
                                        +[('H158','f4')]
                                        +[('pind','i4')]
                                        +[('bflux','f4')]
                                        +[('dflux','f4')])
            store['gind']       = np.arange(n_gal) # Index array into original galaxy position catalog
            store['ra']         = radec_file['ra'][:]*np.pi/180. # Right ascension
            store['dec']        = radec_file['dec'][:]*np.pi/180. # Declination
            r_ = np.zeros(n_gal)
            gal_rng.generate(r_)
            store['pind']       = pind_list_[(r_*len(pind_list_)).astype(int)] # Index array into original galaxy photometry catalog
            r_ = np.zeros(int(n_gal/2)+n_gal%2)
            gal_rng.generate(r_)
            store['rot'][0::2]  = r_*2.*np.pi # Random rotation (every pair of objects is rotated 90 deg to cancel shape noise)
            store['rot'][1::2]  = store['rot'][0:n_gal-n_gal%2:2]+np.pi
            store['rot'][store['rot']>2.*np.pi]-=2.*np.pi
            r_ = np.zeros(n_gal)
            gal_rng.generate(r_)
            r_ = (r_*len(params['shear_list'])).astype(int)
            store['g1']         = np.array(params['shear_list'])[r_,0] # Shears to apply to galaxy
            store['g2']         = np.array(params['shear_list'])[r_,1]
            if params['gal_model'] == 'disk': # Disk only model, no bulge or knot flux 
                store['bflux']  = np.zeros(n_gal)
                store['dflux']  = np.ones(n_gal)
            elif params['gal_model'] == 'bulge': # Bulge only model, no disk or knot flux
                store['bflux']  = np.ones(n_gal)
                store['dflux']  = np.zeros(n_gal)
            else: # General composite model. bflux = bulge flux fraction. dflux*(1-bflux) = disk flux fraction. Remaining flux is in form of star-knots, (1-bflux)*(1-dflux). Knot flux is capped at 50% of disk flux.
                r_ = np.zeros(n_gal)
                gal_rng.generate(r_)
                store['bflux']  = r_
                r_ = np.zeros(n_gal)
                gal_rng.generate(r_)
                store['dflux']  = r_/4.+0.75
            store['size']       = self.fwhm_to_hlr(phot['fwhm'][store['pind']]) # half-light radius
            store['z']          = phot['redshift'][store['pind']] # redshift
            for f in filter_dither_dict.keys():
                store[f]        = phot[filter_flux_dict[f]][store['pind']] # magnitude in this filter
            for name in store.dtype.names:
                print name,np.mean(store[name]),np.min(store[name]),np.max(store[name])

            # Save truth file with galaxy properties
            return self.dump_truth_gal(filename,store)

            print '-------truth catalog built-------'

        else:
            raise ParamError('COSMOS profiles not currently implemented.')            
            # cosmos gal not guaranteed to work. uncomment at own risk 
            # # Cosmos real or parametric objects
            # if self.params['gal_type'] == 1:
            #     use_real = False
            #     gtype = 'parametric'
            # else:
            #     use_real = True
            #     gtype = 'real'

            # # Load cosmos catalog
            # cat = galsim.COSMOSCatalog(self.params['cat_name'], dir=self.params['cat_dir'], use_real=use_real)
            # self.logger.info('Read in %d galaxies from catalog'%cat.nobjects)

            # rand_ind = []
            # for i in range(self.params['gal_n_use']):
            #     # Select unique cosmos index list with length gal_n_use.
            #     rand_ind.append(int(self.gal_rng()*cat.nobjects))
            # # Make object list of unique cosmos galaxies
            # self.obj_list = cat.makeGalaxy(rand_ind, chromatic=True, gal_type=gtype)

    def init_star(self,params):
        """
        Compiles a list of stars properties to draw. 
        Not working with new structure yet.

        Input 
        params   : parameter dict
        """

        # Make sure star catalog filename is well-formed and link to it
        if isinstance(params['star_sample'],string_types):
            # Provided a catalog of star positions and properties.
            stars = fio.FITS(params['star_sample'])[-1]
            self.n_star = stars.read_header()['NAXIS2']
        else:
            return None

        # # Cut really bright stars that take too long to draw for now
        # mask = np.ones(len(stars),dtype=bool)
        # for f in filter_dither_dict.keys():
        #     mask = mask & (stars_[f]<1e5)
        # stars = stars[mask]

        return stars

class modify_image():
    """
    Class to simulate non-idealities and noise of wfirst detector images.
    """

    def __init__(self,params,rng):
        """
        Set up noise properties of image

        Input
        params  : parameter dict
        rng     : Random generator
        """

        self.params    = params
        self.rng       = rng
        self.noise     = self.init_noise_model()

    def add_effects(self,im,pointing,radec,local_wcs,phot=False):
        """
        Add detector effects for WFIRST.

        Input:
        im        : Postage stamp or image.
        pointing  : Pointing object
        radec     : World coordinate position of image
        local_wcs : The local WCS
        phot      : photon shooting mode

        Preserve order:
        1) add_background
        2) add_poisson_noise
        3) recip_failure 
        4) quantize
        5) dark_current
        6) nonlinearity
        7) interpix_cap
        8) Read noise
        9) e_to_ADU
        10) quantize

        Where does persistence get added? Immediately before/after background?
        """

        im, sky_image = self.add_background(im,pointing,radec,local_wcs,phot=phot) # Add background to image and save background
        im = self.add_poisson_noise(im,sky_image,phot=phot) # Add poisson noise to image
        im = self.recip_failure(im) # Introduce reciprocity failure to image
        im.quantize() # At this point in the image generation process, an integer number of photons gets detected
        im = self.dark_current(im) # Add dark current to image
        im = self.nonlinearity(im) # Apply nonlinearity
        im = self.interpix_cap(im) # Introduce interpixel capacitance to image.
        im = self.add_read_noise(im)
        im = self.e_to_ADU(im) # Convert electrons to ADU
        im.quantize() # Finally, the analog-to-digital converter reads in an integer value.
        # Note that the image type after this step is still a float. If we want to actually
        # get integer values, we can do new_img = galsim.Image(im, dtype=int)
        # Since many people are used to viewing background-subtracted images, we return a
        # version with the background subtracted (also rounding that to an int).
        im,sky_image = self.finalize_background_subtract(im,sky_image)
        # im = galsim.Image(im, dtype=int)
        # get weight map
        sky_image.invertSelf()

        return im, sky_image


    def add_effects_flat(self,im,phot=False):
        """
        Add detector effects for WFIRST.

        Input:
        im        : Postage stamp or image.
        pointing  : Pointing object
        radec     : World coordinate position of image
        local_wcs : The local WCS
        phot      : photon shooting mode

        Preserve order:
        1) add_background
        2) add_poisson_noise
        3) recip_failure 
        4) quantize
        5) dark_current
        6) nonlinearity
        7) interpix_cap
        8) Read noise
        9) e_to_ADU
        10) quantize

        Where does persistence get added? Immediately before/after background?
        """

        # im = self.add_poisson_noise(im,sky_image,phot=phot) # Add poisson noise to image
        im = self.recip_failure(im) # Introduce reciprocity failure to image
        im.quantize() # At this point in the image generation process, an integer number of photons gets detected
        im = self.dark_current(im) # Add dark current to image
        im = self.nonlinearity(im) # Apply nonlinearity
        im = self.interpix_cap(im) # Introduce interpixel capacitance to image.
        im = self.add_read_noise(im)
        im = self.e_to_ADU(im) # Convert electrons to ADU
        im.quantize() # Finally, the analog-to-digital converter reads in an integer value.

        return im

    def get_eff_sky_bg(self,pointing,radec):
        """
        Calculate effective sky background per pixel for nominal wfirst pixel scale.

        Input
        pointing            : Pointing object
        radec               : World coordinate position of image        
        """

        sky_level = wfirst.getSkyLevel(pointing.bpass, world_pos=radec, date=pointing.date)
        sky_level *= (1.0 + wfirst.stray_light_fraction)*wfirst.pixel_scale**2

        return sky_level


    def add_background(self,im,pointing,radec,local_wcs,sky_level=None,thermal_backgrounds=None,phot=False):
        """
        Add backgrounds to image (sky, thermal).

        First we get the amount of zodaical light for a position corresponding to the position of 
        the object. The results are provided in units of e-/arcsec^2, using the default WFIRST
        exposure time since we did not explicitly specify one. Then we multiply this by a factor
        >1 to account for the amount of stray light that is expected. If we do not provide a date
        for the observation, then it will assume that it's the vernal equinox (sun at (0,0) in
        ecliptic coordinates) in 2025.

        Input
        im                  : Image
        pointing            : Pointing object
        radec               : World coordinate position of image
        local_wcs           : Local WCS
        sky_level           : The sky level. None uses current specification.
        thermal_backgrounds : The thermal background of instrument. None uses current specification.
        phot                : photon shooting mode
        """

        # If requested, dump an initial fits image to disk for diagnostics
        if self.params['save_diff']:
            orig = im.copy()
            orig.write('orig.fits')

        # If effect is turned off, return image unchanged
        if not self.params['use_background']:
            return im,None

        # Build current specification sky level if sky level not given
        if sky_level is None:
            sky_level = wfirst.getSkyLevel(pointing.bpass, world_pos=radec, date=pointing.date)
            sky_level *= (1.0 + wfirst.stray_light_fraction)
        # Make a image of the sky that takes into account the spatially variable pixel scale. Note
        # that makeSkyImage() takes a bit of time. If you do not care about the variable pixel
        # scale, you could simply compute an approximate sky level in e-/pix by multiplying
        # sky_level by wfirst.pixel_scale**2, and add that to final_image.

        # Create sky image
        sky_stamp = galsim.Image(bounds=im.bounds, wcs=local_wcs)
        local_wcs.makeSkyImage(sky_stamp, sky_level)

        # This image is in units of e-/pix. Finally we add the expected thermal backgrounds in this
        # band. These are provided in e-/pix/s, so we have to multiply by the exposure time.
        if thermal_backgrounds is None:
            sky_stamp += wfirst.thermal_backgrounds[pointing.filter]*wfirst.exptime
        else:
            sky_stamp += thermal_backgrounds*wfirst.exptime

        # Adding sky level to the image.
        if not phot:
            im += sky_stamp

        # If requested, dump a post-change fits image to disk for diagnostics
        if self.params['save_diff']:
            prev = im.copy()
            diff = prev-orig
            diff.write('sky_a.fits')
        
        return im,sky_stamp

    def init_noise_model(self):
        """
        Generate a poisson noise model.
        """

        return galsim.PoissonNoise(self.rng)

    def add_poisson_noise(self,im,sky_image,phot=False):
        """
        Add pre-initiated poisson noise to image.

        Input
        im : image
        """

        # If effect is turned off, return image unchanged
        if not self.params['use_poisson_noise']:
            return im

        # Check if noise initiated
        if self.noise is None:
            self.init_noise_model()

        # Add poisson noise to image
        if phot:
            sky_image_ = sky_image.copy()
            sky_image_.addNoise(self.noise)
            im += sky_image_
        else:
            im.addNoise(self.noise)

        # If requested, dump a post-change fits image to disk for diagnostics. Both cumulative and iterative delta.
        if self.params['save_diff']:
            diff = im-prev
            diff.write('noise_a.fits')
            diff = im-orig
            diff.write('noise_b.fits')
            prev = im.copy()

        return im

    def recip_failure(self,im,exptime=wfirst.exptime,alpha=wfirst.reciprocity_alpha,base_flux=1.0):
        """
        Introduce reciprocity failure to image.

        Reciprocity, in the context of photography, is the inverse relationship between the
        incident flux (I) of a source object and the exposure time (t) required to produce a given
        response(p) in the detector, i.e., p = I*t. However, in NIR detectors, this relation does
        not hold always. The pixel response to a high flux is larger than its response to a low
        flux. This flux-dependent non-linearity is known as 'reciprocity failure', and the
        approximate amount of reciprocity failure for the WFIRST detectors is known, so we can
        include this detector effect in our images.

        Input
        im        : image
        exptime   : Exposure time
        alpha     : Reciprocity alpha
        base_flux : Base flux
        """

        # If effect is turned off, return image unchanged
        if not self.params['use_recip_failure']:
            return im

        # Add reciprocity effect
        im.addReciprocityFailure(exp_time=exptime, alpha=alpha, base_flux=base_flux)

        # If requested, dump a post-change fits image to disk for diagnostics. Both cumulative and iterative delta.
        if self.params['save_diff']:
            diff = im-prev
            diff.write('recip_a.fits')
            diff = im-orig
            diff.write('recip_b.fits')
            prev = im.copy()

        return im

    def dark_current(self,im,dark_current=None):
        """
        Adding dark current to the image.

        Even when the detector is unexposed to any radiation, the electron-hole pairs that
        are generated within the depletion region due to finite temperature are swept by the
        high electric field at the junction of the photodiode. This small reverse bias
        leakage current is referred to as 'dark current'. It is specified by the average
        number of electrons reaching the detectors per unit time and has an associated
        Poisson noise since it is a random event.

        Input
        im           : image
        dark_current : The dark current to apply
        """

        # If effect is turned off, return image unchanged
        if not self.params['use_dark_current']:
            return im

        # If dark_current is not provided, calculate what it should be based on current specifications
        self.dark_current_ = dark_current
        if self.dark_current_ is None:
            self.dark_current_ = wfirst.dark_current*wfirst.exptime

        # Add dark current to image
        dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(self.rng, self.dark_current_))
        im.addNoise(dark_noise)

        # NOTE: Sky level and dark current might appear like a constant background that can be
        # simply subtracted. However, these contribute to the shot noise and matter for the
        # non-linear effects that follow. Hence, these must be included at this stage of the
        # image generation process. We subtract these backgrounds in the end.

        # If requested, dump a post-change fits image to disk for diagnostics. Both cumulative and iterative delta.
        if self.params['save_diff']:
            diff = im-prev
            diff.write('dark_a.fits')
            diff = im-orig
            diff.write('dark_b.fits')
            prev = im.copy()

        return im

    def nonlinearity(self,im,NLfunc=wfirst.NLfunc):
        """
        Applying a quadratic non-linearity.

        Note that users who wish to apply some other nonlinearity function (perhaps for other NIR
        detectors, or for CCDs) can use the more general nonlinearity routine, which uses the
        following syntax:
        final_image.applyNonlinearity(NLfunc=NLfunc)
        with NLfunc being a callable function that specifies how the output image pixel values
        should relate to the input ones.

        Input
        im     : Image
        NLfunc : Nonlinearity function
        """

        # If effect is turned off, return image unchanged
        if not self.params['use_nonlinearity']:
            return im

        # Apply the WFIRST nonlinearity routine, which knows all about the nonlinearity expected in
        # the WFIRST detectors. Alternately, use a user-provided function.
        im.applyNonlinearity(NLfunc=NLfunc)

        # If requested, dump a post-change fits image to disk for diagnostics. Both cumulative and iterative delta.
        if self.params['save_diff']:
            diff = im-prev
            diff.write('nl_a.fits')
            diff = im-orig
            diff.write('nl_b.fits')
            prev = im.copy()

        return im

    def interpix_cap(self,im,kernel=wfirst.ipc_kernel):
        """
        Including Interpixel capacitance

        The voltage read at a given pixel location is influenced by the charges present in the
        neighboring pixel locations due to capacitive coupling of sense nodes. This interpixel
        capacitance effect is modeled as a linear effect that is described as a convolution of a
        3x3 kernel with the image. The WFIRST IPC routine knows about the kernel already, so the
        user does not have to supply it.

        Input
        im      : image
        kernel  : Interpixel capacitance kernel
        """

        # If effect is turned off, return image unchanged
        if not self.params['use_interpix_cap']:
            return im

        # Apply interpixel capacitance
        im.applyIPC(kernel, edge_treatment='extend', fill_value=None)

        # If requested, dump a post-change fits image to disk for diagnostics. Both cumulative and iterative delta.
        if self.params['save_diff']:
            diff = im-prev
            diff.write('ipc_a.fits')
            diff = im-orig
            diff.write('ipc_b.fits')
            prev = im.copy()

        return im

    def add_read_noise(self,im,sigma=wfirst.read_noise):
        """
        Adding read noise

        Read noise is the noise due to the on-chip amplifier that converts the charge into an
        analog voltage.  We already applied the Poisson noise due to the sky level, so read noise
        should just be added as Gaussian noise

        Input
        im    : image
        sigma : Variance of read noise
        """

        if not self.params['use_read_noise']:
            return im

        # Create noise realisation and apply it to image
        read_noise = galsim.GaussianNoise(self.rng, sigma=sigma)
        im.addNoise(read_noise)

        return im

    def e_to_ADU(self,im):
        """
        We divide by the gain to convert from e- to ADU. Currently, the gain value in the WFIRST
        module is just set to 1, since we don't know what the exact gain will be, although it is
        expected to be approximately 1. Eventually, this may change when the camera is assembled,
        and there may be a different value for each SCA. For now, there is just a single number,
        which is equal to 1.

        Input 
        im : image
        """

        return im/wfirst.gain

    def finalize_sky_im(self,im):
        """
        Finalize sky background for subtraction from final image. Add dark current, 
        convert to analog voltage, and quantize.

        Input 
        im : sky image
        """

        if (self.params['sub_true_background'])&(self.params['use_dark_current']):
            im = (im + round(self.dark_current_))
        im = self.e_to_ADU(im)
        im.quantize()

        return im

    def finalize_background_subtract(self,im,sky):
        """
        Finalize background subtraction of image.

        Input 
        im : image
        sky : sky image
        """

        # If effect is turned off, return image unchanged
        if not self.params['use_background']:
            return im,sky

        sky.quantize() # Quantize sky
        sky = self.finalize_sky_im(sky) # Finalize sky with dark current, convert to ADU, and quantize.
        im -= sky

        # If requested, dump a final fits image to disk for diagnostics. 
        if self.params['save_diff']:
            im.write('final_a.fits')

        return im,sky

class draw_image():
    """
    This is where the management of drawing happens (basicaly all the galsim interaction).
    The general process is that 1) a galaxy model is specified from the truth catalog, 2) rotated, sheared, and convolved with the psf, 3) its drawn into a postage samp, 4) that postage stamp is added to a persistent image of the SCA, 5) the postage stamp is finalized by going through make_image(). Objects within the SCA are iterated using the iterate_*() functions, and the final SCA image (self.im) can be completed with self.finalize_sca().
    """

    def __init__(self, params, pointing, modify_image, cats, logger, image_buffer=256, rank=0):
        """
        Sets up some general properties, including defining the object index lists, starting the generator iterators, assigning the SEDs (single stand-ins for now but generally red to blue for bulg/disk/knots), defining SCA bounds, and creating the empty SCA image.

        Input
        params          : parameter dict
        pointing        : Pointing object
        modify_image    : modify_image object
        cats            : init_catalots object
        logger          : logger instance
        gal_ind_list    : List of indices from gal truth catalog to attempt to simulate 
        star_ind_list   : List of indices from star truth catalog to attempt to simulate 
        image_buffer    : Number of pixels beyond SCA to attempt simulating objects that may overlap SCA
        rank            : process rank
        """

        self.params       = params
        self.pointing     = pointing
        self.modify_image = modify_image
        self.cats         = cats
        self.stamp_size   = self.params['stamp_size']
        self.num_sizes    = self.params['num_sizes']
        self.gal_iter   = 0
        self.star_iter  = 0
        self.gal_done   = False
        self.star_done  = False
        self.rank       = rank
        # Initialize (pseudo-)random number generators.
        reset_rng(self,self.params['random_seed']+self.rank)

        # Setup galaxy SED
        # Need to generalize to vary sed based on input catalog
        self.galaxy_sed_b = galsim.SED(self.params['sedpath_E'], wave_type='Ang', flux_type='flambda')
        self.galaxy_sed_d = galsim.SED(self.params['sedpath_Scd'], wave_type='Ang', flux_type='flambda')
        self.galaxy_sed_n = galsim.SED(self.params['sedpath_Im'],  wave_type='Ang', flux_type='flambda')
        # Setup star SED
        self.star_sed     = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')

        # Galsim bounds object to specify area to simulate objects that might overlap the SCA
        self.b0  = galsim.BoundsI(  xmin=1-int(image_buffer)/2,
                                    ymin=1-int(image_buffer)/2,
                                    xmax=wfirst.n_pix+int(image_buffer)/2,
                                    ymax=wfirst.n_pix+int(image_buffer)/2)
        # Galsim bounds object to specify area to simulate objects that would have centroids that fall on the SCA to save as postage stamps (pixels not on the SCA have weight=0)
        self.b   = galsim.BoundsI(  xmin=1,
                                    ymin=1,
                                    xmax=wfirst.n_pix,
                                    ymax=wfirst.n_pix)

        # SCA image (empty right now)
        if self.params['draw_sca']:
            self.im = galsim.Image(self.b, wcs=self.pointing.WCS)
        else: 
            self.im = None

        # Get sky background for pointing
        self.sky_level = wfirst.getSkyLevel(self.pointing.bpass, 
                                            world_pos=self.pointing.WCS.toWorld(
                                                        galsim.PositionI(wfirst.n_pix/2,
                                                                        wfirst.n_pix/2)), 
                                            date=self.pointing.date)
        self.sky_level *= (1.0 + wfirst.stray_light_fraction)*wfirst.pixel_scale**2 # adds stray light and converts to photons/cm^2
        self.sky_level *= self.stamp_size*self.stamp_size # Converts to photons, but uses smallest stamp size to do so - not optimal

    def iterate_gal(self):
        """
        Iterator function to loop over all possible galaxies to draw
        """

        # Check if the end of the galaxy list has been reached; return exit flag (gal_done) True
        # You'll have a bad day if you aren't checking for this flag in any external loop...
        # self.gal_done = True
        # return
        if self.gal_iter == self.cats.get_gal_length():
            self.gal_done = True
            print 'Proc '+str(self.rank)+' done with galaxies.'
            return 

        # Reset galaxy information
        self.gal_model = None
        self.gal_stamp = None

        # if self.gal_iter>1000:
        #     self.gal_done = True
        #     return             

        if self.gal_iter%100==0:
            print 'Progress '+str(self.rank)+': Attempting to simulate galaxy '+str(self.gal_iter)+' in SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.'

        # Galaxy truth index and array for this galaxy
        self.ind,self.gal = self.cats.get_gal(self.gal_iter)
        self.gal_iter    += 1
        # if self.ind != 157733:
        #     return

        # if self.ind != 144078:
        #     return

        # If galaxy image position (from wcs) doesn't fall within simulate-able bounds, skip (slower) 
        # If it does, draw it
        if self.check_position(self.gal['ra'],self.gal['dec']):
            self.draw_galaxy()

    def iterate_star(self):
        """
        Iterator function to loop over all possible stars to draw
        """

        # self.star_done = True
        # return 
        # Don't draw stars into postage stamps
        if not self.params['draw_sca']:
            self.star_done = True
            print 'Proc '+str(self.rank)+' done with stars.'
            return 
        if not self.params['draw_stars']:
            self.star_done = True
            print 'Proc '+str(self.rank)+' not doing stars.'
            return             
        # Check if the end of the star list has been reached; return exit flag (gal_done) True
        # You'll have a bad day if you aren't checking for this flag in any external loop...
        if self.star_iter == self.cats.get_star_length():
            self.star_done = True
            return 

        # Not participating in star parallelisation
        if self.rank == -1:
            self.star_done = True
            return 

        if self.star_iter%10==0:
            print 'Progress '+str(self.rank)+': Attempting to simulate star '+str(self.star_iter)+' in SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.'

        # Star truth index for this galaxy
        self.ind,self.star = self.cats.get_star(self.star_iter)
        self.star_iter    += 1

        # If star image position (from wcs) doesn't fall within simulate-able bounds, skip (slower) 
        # If it does, draw it
        if self.check_position(self.star['ra'],self.star['dec']):
            self.draw_star()

    def check_position(self, ra, dec):
        """
        Create the world and image position galsim objects for obj, as well as the local WCS. Return whether object is in SCA (+half-stamp-width border).

        Input
        ra  : RA of object
        dec : Dec of object
        """

        # Galsim world coordinate object (ra,dec)
        self.radec = galsim.CelestialCoord(ra*galsim.radians, dec*galsim.radians)

        # Galsim image coordinate object 
        self.xy = self.pointing.WCS.toImage(self.radec)

        # Galsim integer image coordinate object 
        self.xyI = galsim.PositionI(int(self.xy.x),int(self.xy.y))

        # Galsim image coordinate object holding offset from integer pixel grid 
        self.offset = self.xy-self.xyI

        # Define the local_wcs at this world position
        self.local_wcs = self.pointing.WCS.local(self.xy)

        # Return whether object is in SCA (+half-stamp-width border)
        return self.b0.includes(self.xyI)

    def make_sed_model(self, model, sed):
        """
        Modifies input SED to be at appropriate redshift and magnitude, then applies it to the object model.

        Input
        model : Galsim object model
        sed   : Template SED for object
        flux  : flux fraction in this sed
        """

        # Apply correct flux from magnitude for filter bandpass
        sed_ = sed.atRedshift(self.gal['z'])
        sed_ = sed_.withMagnitude(self.gal['H158'], wfirst.getBandpasses(AB_zeropoint=True)['H158'])

        # Return model with SED applied
        return model * sed_

    def galaxy_model(self):
        """
        Generate the intrinsic galaxy model based on truth catalog parameters
        """

        # Generate galaxy model
        # Calculate flux fraction of disk portion 
        flux = (1.-self.gal['bflux']) * self.gal['dflux']
        if flux > 0:
            # If any flux, build Sersic disk galaxy (exponential) and apply appropriate SED
            self.gal_model = galsim.Sersic(1, half_light_radius=1.*self.gal['size'], flux=flux, trunc=10.*self.gal['size'])
            self.gal_model = self.make_sed_model(self.gal_model, self.galaxy_sed_d)
            # self.gal_model = self.gal_model.withScaledFlux(flux)

        # Calculate flux fraction of knots portion 
        flux = (1.-self.gal['bflux']) * (1.-self.gal['dflux'])
        if flux > 0:
            # If any flux, build star forming knots model and apply appropriate SED
            knots = galsim.RandomWalk(npoints=self.params['knots'], half_light_radius=1.*self.gal['size'], flux=flux, rng=self.rng) 
            knots = self.make_sed_model(knots, self.galaxy_sed_n)
            # knots = knots.withScaledFlux(flux)
            # Sum the disk and knots, then apply intrinsic ellipticity to the disk+knot component. Fixed intrinsic shape, but can be made variable later.
            self.gal_model = galsim.Add([self.gal_model, knots])
            self.gal_model = self.gal_model.shear(e1=0.25, e2=0.25)
 
        # Calculate flux fraction of bulge portion 
        flux = self.gal['bflux']
        if flux > 0:
            # If any flux, build Sersic bulge galaxy (de vacaleurs) and apply appropriate SED
            bulge = galsim.Sersic(4, half_light_radius=1.*self.gal['size'], flux=flux, trunc=10.*self.gal['size']) 
            # Apply intrinsic ellipticity to the bulge component. Fixed intrinsic shape, but can be made variable later.
            bulge = bulge.shear(e1=0.25, e2=0.25)
            # Apply the SED
            bulge = self.make_sed_model(bulge, self.galaxy_sed_b)
            # bulge = bulge.withScaledFlux(flux)

            if self.gal_model is None:
                # No disk or knot component, so save the galaxy model as the bulge part
                self.gal_model = bulge
            else:
                # Disk/knot component, so save the galaxy model as the sum of two parts
                self.gal_model = galsim.Add([self.gal_model, bulge])


    def galaxy(self):
        """
        Call galaxy_model() to get the intrinsic galaxy model, then apply properties relevant to its observation
        """

        # Build intrinsic galaxy model
        self.galaxy_model()

        # Random rotation (pairs of objects are offset by pi/2 to cancel shape noise)
        self.gal_model = self.gal_model.rotate(self.gal['rot']*galsim.radians) 
        # Apply a shear
        self.gal_model = self.gal_model.shear(g1=self.gal['g1'],g2=self.gal['g1'])
        # Rescale flux appropriately for wfirst
        self.gal_model = self.gal_model * galsim.wfirst.collecting_area * galsim.wfirst.exptime

        # Ignoring chromatic stuff for now for speed, so save correct flux of object
        flux = self.gal_model.calculateFlux(self.pointing.bpass)
        self.mag = self.gal_model.calculateMagnitude(self.pointing.bpass)
        # print 'galaxy flux',flux
        # Evaluate the model at the effective wavelength of this filter bandpass (should change to effective SED*bandpass?)
        # This makes the object achromatic, which speeds up drawing and convolution
        self.gal_model  = self.gal_model.evaluateAtWavelength(self.pointing.bpass.effective_wavelength)
        # Reassign correct flux
        self.gal_model  = self.gal_model.withFlux(flux) # reapply correct flux
        
        if self.sky_level/flux < galsim.GSParams().folding_threshold:
            gsparams = galsim.GSParams( folding_threshold=self.sky_level/flux,
                                        maximum_fft_size=16384 )
        else:
            gsparams = galsim.GSParams( maximum_fft_size=16384 )

        # Convolve with PSF
        self.gal_model = galsim.Convolve(self.gal_model.withGSParams(gsparams), self.pointing.PSF, propagate_gsparams=False)
 
        # Convolve with additional los motion (jitter), if any
        if 'los_motion' in self.params:
            los = galsim.Gaussian(fwhm=2.*np.sqrt(2.*np.log(2.))*self.params['los_motion'])
            los = los.shear(g1=self.params['los_motion_e1'],g2=self.params['los_motion_e1']) # assymetric jitter noise
            self.gal_model = galsim.Convolve(self.gal_model, los)

        # chromatic stuff replaced by above lines
        # # Draw galaxy igal into stamp.
        # self.gal_list[igal].drawImage(self.pointing.bpass[self.params['filter']], image=gal_stamp)
        # # Add detector effects to stamp.

    def star_model(self, sed = None, mag = 0.):
        """
        Create star model for PSF or for drawing stars into SCA

        Input
        sed  : The stellar SED
        mag  : The magnitude of the star
        """

        # Generate star model (just a delta function) and apply SED
        if sed is not None:
            sed_ = sed.withMagnitude(mag, self.pointing.bpass)
            self.st_model = galsim.DeltaFunction() * sed_  * wfirst.collecting_area * wfirst.exptime
            flux = self.st_model.calculateFlux(self.pointing.bpass)
            ft = self.sky_level/flux
            # print mag,flux,ft
            # if ft<0.0005:
            #     ft = 0.0005
            if ft < galsim.GSParams().folding_threshold:
                gsparams = galsim.GSParams( folding_threshold=self.sky_level/flux,
                                            maximum_fft_size=16384 )
            else:
                gsparams = galsim.GSParams( maximum_fft_size=16384 )
        else:
            self.st_model = galsim.DeltaFunction(flux=1.)
            gsparams = galsim.GSParams( maximum_fft_size=16384 )

        # Evaluate the model at the effective wavelength of this filter bandpass (should change to effective SED*bandpass?)
        # This makes the object achromatic, which speeds up drawing and convolution
        self.st_model = self.st_model.evaluateAtWavelength(self.pointing.bpass.effective_wavelength)

        # Convolve with PSF
        if mag!=0.:
            self.st_model = galsim.Convolve(self.st_model, self.pointing.PSF, gsparams=gsparams, propagate_gsparams=False)
        else:
            self.st_model = galsim.Convolve(self.st_model, self.pointing.PSF)

        # Convolve with additional los motion (jitter), if any
        if 'los_motion' in self.params:
            los = galsim.Gaussian(fwhm=2.*np.sqrt(2.*np.log(2.))*self.params['los_motion'])
            los = los.shear(g1=0.3,g2=0.) # assymetric jitter noise
            self.st_model = galsim.Convolve(self.st_model, los)

        if mag!=0.:
            return gsparams

        # old chromatic version
        # self.psf_list[igal].drawImage(self.pointing.bpass[self.params['filter']],image=psf_stamp, wcs=local_wcs)

    def get_stamp_size_factor(self,obj,factor=5):
        """
        Select the stamp size multiple to use.

        Input
        obj    : Galsim object
        factor : Factor to multiple suggested galsim stamp size by
        """

        return int(obj.getGoodImageSize(wfirst.pixel_scale)) / self.stamp_size
        # return int(self.gal['size']/wfirst.pixel_scale * factor) / self.stamp_size + 1

    def draw_galaxy(self):
        """
        Draw the galaxy model into the SCA (neighbors and blending) and/or the postage stamp (isolated).
        """

        self.gal_stamp_too_large = False

        # Build galaxy model that will be drawn into images
        self.galaxy()

        stamp_size_factor = self.get_stamp_size_factor(self.gal_model)

        # # Skip drawing some really huge objects (>twice the largest stamp size)
        # if stamp_size_factor>2.*self.num_sizes:
        #     return

        # Create postage stamp bounds at position of object
        b = galsim.BoundsI( xmin=self.xyI.x-int(stamp_size_factor*self.stamp_size)/2,
                            ymin=self.xyI.y-int(stamp_size_factor*self.stamp_size)/2,
                            xmax=self.xyI.x+int(stamp_size_factor*self.stamp_size)/2,
                            ymax=self.xyI.y+int(stamp_size_factor*self.stamp_size)/2)

        # If this postage stamp doesn't overlap the SCA bounds at all, no reason to draw anything
        if not (b&self.b).isDefined():
            return

        # Create postage stamp for galaxy
        gal_stamp = galsim.Image(b, wcs=self.pointing.WCS)

        # Draw galaxy model into postage stamp. This is the basis for both the postage stamp output and what gets added to the SCA image. This will obviously create biases if the postage stamp is too small - need to monitor that.
        self.gal_model.drawImage(image=gal_stamp,offset=self.offset,method='phot',rng=self.rng)
        # gal_stamp.write(str(self.ind)+'.fits')

        # Add galaxy stamp to SCA image
        if self.params['draw_sca']:
            self.im[b&self.b] = self.im[b&self.b] + gal_stamp[b&self.b]

        # If object too big for stamp sizes, skip saving a stamp
        if stamp_size_factor>=self.num_sizes:
            print 'too big stamp',stamp_size_factor,stamp_size_factor*self.stamp_size
            self.gal_stamp_too_large = True
            return

        # Check if galaxy center falls on SCA
        # Apply background, noise, and WFIRST detector effects
        # Get final galaxy stamp and weight map
        if self.b.includes(self.xyI):
            gal_stamp, weight = self.modify_image.add_effects(gal_stamp[b&self.b],self.pointing,self.radec,self.pointing.WCS,phot=True)

            # Copy part of postage stamp that falls on SCA - set weight map to zero for parts outside SCA
            self.gal_stamp = galsim.Image(b, wcs=self.pointing.WCS)
            self.gal_stamp[b&self.b] = self.gal_stamp[b&self.b] + gal_stamp[b&self.b]
            self.weight_stamp = galsim.Image(b, wcs=self.pointing.WCS)
            self.weight_stamp[b&self.b] = self.weight_stamp[b&self.b] + weight[b&self.b]

            # If we're saving the true PSF model, simulate an appropriate unit-flux star and draw it (oversampled) at the position of the galaxy
            if self.params['draw_true_psf']:
                self.star_model() #Star model for PSF (unit flux)
                # Create modified WCS jacobian for super-sampled pixelisation
                wcs = galsim.JacobianWCS(dudx=self.local_wcs.dudx/self.params['oversample'],
                                         dudy=self.local_wcs.dudy/self.params['oversample'],
                                         dvdx=self.local_wcs.dvdx/self.params['oversample'],
                                         dvdy=self.local_wcs.dvdy/self.params['oversample'])
                # Create psf stamp with oversampled pixelisation
                self.psf_stamp = galsim.Image(self.params['psf_stampsize']*self.params['oversample'], self.params['psf_stampsize']*self.params['oversample'], wcs=wcs)
                # Draw PSF into postage stamp
                self.st_model.drawImage(image=self.psf_stamp,wcs=wcs)

    def draw_star(self):
        """
        Draw a star into the SCA
        """

        # Get star model with given SED and flux
        gsparams = self.star_model(sed=self.star_sed,mag=self.star[self.pointing.filter])

        # Get good stamp size multiple for star
        # stamp_size_factor = self.get_stamp_size_factor(self.st_model)#.withGSParams(gsparams))
        stamp_size_factor = 40

        # Create postage stamp bounds for star
        b = galsim.BoundsI( xmin=self.xyI.x-int(stamp_size_factor*self.stamp_size)/2,
                            ymin=self.xyI.y-int(stamp_size_factor*self.stamp_size)/2,
                            xmax=self.xyI.x+int(stamp_size_factor*self.stamp_size)/2,
                            ymax=self.xyI.y+int(stamp_size_factor*self.stamp_size)/2 )

        # If postage stamp doesn't overlap with SCA, don't draw anything
        if not (b&self.b).isDefined():
            return

        # Create star postage stamp
        star_stamp = galsim.Image(b, wcs=self.pointing.WCS)

        # Draw star model into postage stamp
        self.st_model.drawImage(image=star_stamp,offset=self.offset,method='phot',rng=self.rng,maxN=1000000)

        # star_stamp.write('/fs/scratch/cond0083/wfirst_sim_out/images/'+str(self.ind)+'.fits.gz')

        # Add star stamp to SCA image
        self.im[b&self.b] = self.im[b&self.b] + star_stamp[b&self.b]
        # self.st_model.drawImage(image=self.im,add_to_image=True,offset=self.xy-self.im.true_center,method='phot',rng=self.rng,maxN=1000000)

    def retrieve_stamp(self):
        """
        Helper function to accumulate various information about a postage stamp and return it in dictionary form.
        """

        if self.gal_stamp is None:
            return None

        if self.gal_stamp_too_large:
            # stamp size too big
            return {'ind'    : self.ind, # truth index
                    'ra'     : self.gal['ra'], # ra of galaxy
                    'dec'    : self.gal['dec'], # dec of galaxy
                    'x'      : self.xy.x, # SCA x position of galaxy
                    'y'      : self.xy.y, # SCA y position of galaxy
                    'dither' : self.pointing.dither, # dither index
                    'mag'    : self.mag, #Calculated magnitude
                    'stamp'  : self.get_stamp_size_factor(self.gal_model)*self.stamp_size, # Get stamp size in pixels
                    'gal'    : None, # Galaxy image object (includes metadata like WCS)
                    'psf'    : None, # Flattened array of PSF image
                    'weight' : None } # Flattened array of weight map

        return {'ind'    : self.ind, # truth index
                'ra'     : self.gal['ra'], # ra of galaxy
                'dec'    : self.gal['dec'], # dec of galaxy
                'x'      : self.xy.x, # SCA x position of galaxy
                'y'      : self.xy.y, # SCA y position of galaxy
                'dither' : self.pointing.dither, # dither index
                'mag'    : self.mag, #Calculated magnitude
                'stamp'  : self.get_stamp_size_factor(self.gal_model)*self.stamp_size, # Get stamp size in pixels
                'gal'    : self.gal_stamp, # Galaxy image object (includes metadata like WCS)
                'psf'    : self.psf_stamp.array.flatten(), # Flattened array of PSF image
                'weight' : self.weight_stamp.array.flatten() } # Flattened array of weight map

    def finalize_sca(self):
        """
        # Apply background, noise, and WFIRST detector effects to SCA image
        # Get final SCA image and weight map
        """

        # World coordinate of SCA center
        radec = self.pointing.WCS.toWorld(galsim.PositionI(wfirst.n_pix/2,wfirst.n_pix/2))
        # Apply background, noise, and WFIRST detector effects to SCA image and return final SCA image and weight map
        return self.modify_image.add_effects(self.im,self.pointing,radec,self.pointing.WCS,phot=True)[0]

class accumulate_output():

    def __init__(self, param_file, filter_, pix, ignore_missing_files = False, setup = False):

        self.params     = yaml.load(open(param_file))
        self.param_file = param_file
        self.ditherfile = self.params['dither_file']
        self.pix        = pix
        logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
        self.logger = logging.getLogger('wfirst_sim')
        self.pointing   = pointing(self.params,self.logger,filter_=filter_,sca=None,dither=None)

        self.accumulate_index_table(setup)

        if not setup:

            self.EmptyMEDS()

            self.accumulate_dithers()

    def accumulate_index_table(self):

        index_filename = get_filename(self.params['out_path'],
                            'truth',
                            self.params['output_meds'],
                            var=self.pointing.filter+'_index_sorted',
                            ftype='fits',
                            overwrite=False)

        if (os.path.exists(index_filename)) and (not self.params['overwrite']):

            self.index = fio.FITS(index_filename).read()

        elif (not os.path.exists(index_filename)) and (not setup):

            raise ParamError('Index file not setup.')

        else:

            if not setup:
                raise ParamError('Trying to setup index file in potentially parallel run. Run with setup first.')

            index_files = get_filenames(self.params['out_path'],
                                        'truth',
                                        self.params['output_meds']+'_'+self.pointing.filter,
                                        var='index',
                                        ftype='fits')

            length = 0
            for filename in index_files:
                length+=fio.FITS(filename).read_header()['NAXIS2']

            self.index = np.zeros(length,dtype=fio.FITS(filename).read().dtype)
            length = 0
            for filename in index_files:
                f = fio.FITS(filename).read()
                self.index[length:length+len(f)] = f

            self.index = self.index[np.argsort(self.index, order=['ind','dither'])]

            steps = np.where(np.roll(self.index['ind'],1)!=self.index['ind'])[0]
            self.index_ = np.zeros(len(self.index)+len(np.unique(self.index['ind'])),dtype=self.index.dtype)
            for name in self.index.dtype.names:
                if name=='dither':
                    self.index_[name] = np.insert(self.index[name],steps,np.ones(len(steps))*-1)
                else:
                    self.index_[name] = np.insert(self.index[name],steps,self.index[name][steps])

            self.index = self.index_
            self.index_= None
            fio.write(index_filename,self.index,clobber=True)

        if setup:
            return

        self.index = self.index[(self.index['stamp']!=0) & (self.get_index_pix()==self.pix)]
        self.steps = np.where(np.roll(self.index['ind'],1)!=self.index['ind'])[0]

    def get_index_pix(self):

        return hp.ang2pix(self.params['nside'],np.pi/2.-np.radians(self.index['dec']),np.radians(self.index['ra']),nest=True)

    def EmptyMEDS(self):
        """
        Based on galsim.des.des_meds.WriteMEDS().
        """

        from galsim._pyfits import pyfits

        indices = self.index['ind']
        bincount = np.bincount(indices)
        MAX_NCUTOUTS = np.argmax(bincount)
        assert np.sum(bincount==1) == 0
        cum_exps = len(indices)

        # get number of objects
        n_obj = len(np.unique(indices))

        # get the primary HDU
        primary = pyfits.PrimaryHDU()

        # second hdu is the object_data
        # cf. https://github.com/esheldon/meds/wiki/MEDS-Format
        cols = []
        tmp  = [[0]*MAX_NCUTOUTS]*n_obj
        cols.append( pyfits.Column(name='id',             format='K', array=np.arange(n_obj)            ) )
        cols.append( pyfits.Column(name='number',         format='K', array=self.index['ind'][self.steps]                        ) )
        cols.append( pyfits.Column(name='ra',             format='D', array=self.index['ra'][self.steps]           ) )
        cols.append( pyfits.Column(name='dec',            format='D', array=self.index['dec'][self.steps]          ) )
        cols.append( pyfits.Column(name='ncutout',        format='K', array=bincount                ) )
        cols.append( pyfits.Column(name='box_size',       format='K', array=self.index['stamp'][self.steps]    ) )
        cols.append( pyfits.Column(name='psf_box_size',   format='K', array=np.ones(n_obj)*self.params['psf_stampsize']*self.params['oversample'] ) )
        cols.append( pyfits.Column(name='file_id',        format='%dK' % MAX_NCUTOUTS, array=[1]*n_obj  ) )
        cols.append( pyfits.Column(name='start_row',      format='%dK' % MAX_NCUTOUTS, array=tmp        ) )
        cols.append( pyfits.Column(name='orig_row',       format='%dD' % MAX_NCUTOUTS, array=tmp        ) )
        cols.append( pyfits.Column(name='orig_col',       format='%dD' % MAX_NCUTOUTS, array=tmp        ) )
        cols.append( pyfits.Column(name='orig_start_row', format='%dK' % MAX_NCUTOUTS, array=tmp        ) )
        cols.append( pyfits.Column(name='orig_start_col', format='%dK' % MAX_NCUTOUTS, array=tmp        ) )
        cols.append( pyfits.Column(name='cutout_row',     format='%dD' % MAX_NCUTOUTS, array=tmp        ) )
        cols.append( pyfits.Column(name='cutout_col',     format='%dD' % MAX_NCUTOUTS, array=tmp        ) )
        cols.append( pyfits.Column(name='dudrow',         format='%dD' % MAX_NCUTOUTS, array=tmp        ) )
        cols.append( pyfits.Column(name='dudcol',         format='%dD' % MAX_NCUTOUTS, array=tmp        ) )
        cols.append( pyfits.Column(name='dvdrow',         format='%dD' % MAX_NCUTOUTS, array=tmp        ) )
        cols.append( pyfits.Column(name='dvdcol',         format='%dD' % MAX_NCUTOUTS, array=tmp        ) )
        cols.append( pyfits.Column(name='psf_start_row',  format='%dK' % MAX_NCUTOUTS, array=tmp        ) )
        cols.append( pyfits.Column(name='dither',         format='%dK' % MAX_NCUTOUTS, array=tmp        ) )
        cols.append( pyfits.Column(name='sca',            format='%dK' % MAX_NCUTOUTS, array=tmp        ) )

        # Depending on the version of pyfits, one of these should work:
        try:
            object_data = pyfits.BinTableHDU.from_columns(cols)
            object_data.name = 'object_data'
        except AttributeError:  # pragma: no cover
            object_data = pyfits.new_table(pyfits.ColDefs(cols))
            object_data.update_ext_name('object_data')

        # third hdu is image_info
        cols = []
        gstring = 'generated_by_galsim'
        cols.append( pyfits.Column(name='image_path',  format='A256',   array=np.repeat(gstring,images) ) )
        cols.append( pyfits.Column(name='image_ext',   format='I',      array=np.zeros(images)          ) )
        cols.append( pyfits.Column(name='weight_path', format='A256',   array=np.repeat(gstring,images) ) )
        cols.append( pyfits.Column(name='weight_ext',  format='I',      array=np.zeros(images)          ) )
        cols.append( pyfits.Column(name='seg_path',    format='A256',   array=np.repeat(gstring,images) ) )
        cols.append( pyfits.Column(name='seg_ext',     format='I',      array=np.zeros(images)          ) )
        cols.append( pyfits.Column(name='bmask_path',  format='A256',   array=np.repeat(gstring,images) ) )
        cols.append( pyfits.Column(name='bmask_ext',   format='I',      array=np.zeros(images)          ) )
        cols.append( pyfits.Column(name='bkg_path',    format='A256',   array=np.repeat(gstring,images) ) )
        cols.append( pyfits.Column(name='bkg_ext',     format='I',      array=np.zeros(images)          ) )
        cols.append( pyfits.Column(name='image_id',    format='K',      array=np.ones(images)*-1        ) )
        cols.append( pyfits.Column(name='image_flags', format='K',      array=np.zeros(images)          ) )
        cols.append( pyfits.Column(name='magzp',       format='E',      array=np.ones(images)*30        ) )
        cols.append( pyfits.Column(name='scale',       format='E',      array=np.zeros(images)          ) )
        # TODO: Not sure if this is right!
        cols.append( pyfits.Column(name='position_offset', format='D',  array=np.zeros(images)          ) )
        try:
            image_info = pyfits.BinTableHDU.from_columns(cols)
            image_info.name = 'image_info'
        except AttributeError:  # pragma: no cover
            image_info = pyfits.new_table(pyfits.ColDefs(cols))
            image_info.update_ext_name('image_info')

        # fourth hdu is metadata
        # default values?
        cols = []
        cols.append( pyfits.Column(name='magzp_ref',     format='E',    array=[30.]                   ))
        cols.append( pyfits.Column(name='DESDATA',       format='A256', array=['generated_by_galsim'] ))
        cols.append( pyfits.Column(name='cat_file',      format='A256', array=['generated_by_galsim'] ))
        cols.append( pyfits.Column(name='coadd_image_id',format='A256', array=['generated_by_galsim'] ))
        cols.append( pyfits.Column(name='coadd_file',    format='A256', array=['generated_by_galsim'] ))
        cols.append( pyfits.Column(name='coadd_hdu',     format='K',    array=[9999]                  ))
        cols.append( pyfits.Column(name='coadd_seg_hdu', format='K',    array=[9999]                  ))
        cols.append( pyfits.Column(name='coadd_srclist', format='A256', array=['generated_by_galsim'] ))
        cols.append( pyfits.Column(name='coadd_wt_hdu',  format='K',    array=[9999]                  ))
        cols.append( pyfits.Column(name='coaddcat_file', format='A256', array=['generated_by_galsim'] ))
        cols.append( pyfits.Column(name='coaddseg_file', format='A256', array=['generated_by_galsim'] ))
        cols.append( pyfits.Column(name='cutout_file',   format='A256', array=['generated_by_galsim'] ))
        cols.append( pyfits.Column(name='max_boxsize',   format='A3',   array=['-1']                  ))
        cols.append( pyfits.Column(name='medsconf',      format='A3',   array=['x']                   ))
        cols.append( pyfits.Column(name='min_boxsize',   format='A2',   array=['-1']                  ))
        cols.append( pyfits.Column(name='se_badpix_hdu', format='K',    array=[9999]                  ))
        cols.append( pyfits.Column(name='se_hdu',        format='K',    array=[9999]                  ))
        cols.append( pyfits.Column(name='se_wt_hdu',     format='K',    array=[9999]                  ))
        cols.append( pyfits.Column(name='seg_hdu',       format='K',    array=[9999]                  ))
        cols.append( pyfits.Column(name='psf_hdu',       format='K',    array=[9999]                  ))
        cols.append( pyfits.Column(name='sky_hdu',       format='K',    array=[9999]                  ))
        cols.append( pyfits.Column(name='fake_coadd_seg',format='K',    array=[9999]                  ))
        try:
            metadata = pyfits.BinTableHDU.from_columns(cols)
            metadata.name = 'metadata'
        except AttributeError:  # pragma: no cover
            metadata = pyfits.new_table(pyfits.ColDefs(cols))
            metadata.update_ext_name('metadata')

        # rest of HDUs are image vectors
        length = np.cumsum(self.index['stamp']**2)
        image_cutouts   = pyfits.ImageHDU( np.zeros(length) , name='image_cutouts'  )
        weight_cutouts  = pyfits.ImageHDU( np.zeros(length) , name='weight_cutouts' )
        seg_cutouts     = pyfits.ImageHDU( np.zeros(length) , name='seg_cutouts'    )
        psf_cutouts     = pyfits.ImageHDU( np.zeros(cum_exps*(self.params['psf_stampsize']*self.params['oversample'])**2) , name='psf'      )

        # write all
        hdu_list = pyfits.HDUList([
            primary,
            object_data,
            image_info,
            metadata,
            image_cutouts,
            weight_cutouts,
            seg_cutouts,
            psf_cutouts
        ])

        self.meds_filename = get_filename(self.params['out_path'],
                            'meds',
                            self.params['output_meds'],
                            var=self.pointing.filter+'_'+str(self.pix),
                            ftype='fits',
                            overwrite=True)

        galsim.fits.writeFile(filename, hdu_list)

        return

    def dump_meds_start_info(self,object_data,i,j):

        object_data['start_row'][i][j] = np.sum(object_data['ncutout'][:i]*object_data['box_size'][:i])+j*object_data['box_size'][i]
        object_data['psf_start_row'][i][j] = np.sum(object_data['ncutout'][:i]*object_data['psf_box_size'][:i])+j*object_data['psf_box_size'][i]

    def dump_meds_wcs_info(self,
                            object_data,
                            i,
                            j,
                            x,
                            y,
                            origin_x,
                            origin_y,
                            dither,
                            sca,
                            dudx,
                            dudy,
                            dvdx,
                            dvdy,
                            wcsorigin_x,
                            wcsorigin_y):

        object_data['orig_row'][i][j]       = y
        object_data['orig_col'][i][j]       = x
        object_data['orig_start_row'][i][j] = origin_y
        object_data['orig_start_col'][i][j] = origin_x
        object_data['dither'][i][j]         = dither
        object_data['sca'][i][j]            = sca
        object_data['dudcol'][i][j]         = dudx
        object_data['dudrow'][i][j]         = dudy
        object_data['dvdcol'][i][j]         = dvdx
        object_data['dvdrow'][i][j]         = dvdy
        object_data['cutout_row'][i][j]     = wcsorigin_y
        object_data['cutout_col'][i][j]     = wcsorigin_x

    def dump_meds_pix_info(self,meds,i,j,gal,weight,psf):

        meds['image_cutouts'].write(gal, start=object_data['start_row'][i][j])
        meds['weight_cutouts'].write(weight, start=object_data['start_row'][i][j])
        meds['psf'].write(psf, start=object_data['psf_start_row'][i][j])

    def accumulate_dithers(self):
        """
        Accumulate the written pickle files that contain the postage stamps for all objects, with SCA and dither ids.
        Write stamps to MEDS file, and SCA and dither ids to truth files. 
        """

        meds = fio.FITS(self.meds_filename,'rw')
        object_data = meds['object_data'].read()

        stamps_used = np.unqiue(self.index[['dither','sca']])
        for s in range(len(stamps_used)):
            filename = get_filename(self.params['out_path'],
                                    'stamps',
                                    self.params['output_meds'],
                                    var=self.pointing.filter+'_'+str(stamps_used['dither'][s]),
                                    name2=str(stamps_used['sca'][s]),
                                    ftype='cPickle',
                                    overwrite=False)
            gals = load_obj(filename)

            start_exps = 0
            for gal in gals:
                i = np.where(gal['ind'] == object_data['number']) 
                if len(i)==0:
                    continue
                assert len(i)==1
                i = i[0]
                j = np.argmax(object_data['dither'][i])
                if j==0:
                    j=1
                index_i = np.where((self.index['ind']==gal['ind'])&(self.index['dither']==gal['dither']))[0]
                assert len(index_i)==1
                index_i=index_i[0]

                self.dump_meds_start_info(object_data,i,j)

                origin_x = gal['gal'].origin.x
                origin_y = gal['gal'].origin.y
                gal['gal'].setOrigin(0,0)
                wcs = gal['gal'].wcs.affine(image_pos=gal['gal'].trueCenter())
                self.dump_meds_wcs_info(object_data,
                                        i,
                                        j,
                                        gal['x'],
                                        gal['y'],
                                        origin_x,
                                        origin_y,
                                        self.index['dither'][index_i],
                                        self.index['sca'][index_i],
                                        wcs.dudx,
                                        wcs.dudy,
                                        wcs.dvdx,
                                        wcs.dvdy,
                                        wcs.origin.x,
                                        wcs.origin.y)

                if object_data['box_size'][i] != self.index['stamp'][index_i]:
                    print 'stamp size mismatch'
                    return

                self.dump_meds_pix_info(meds,
                                        i,
                                        j,
                                        gal['gal'].array.flatten(),
                                        gal['weight'].array.flatten(),
                                        gal['psf'].array.flatten())

                if j+1==object_data['ncutout'][i]:
                    get_coadd(i,object_data)

        meds['object_data'].write(object_data)
        meds.close()

        return

    def get_coadd(self,i,object_data,meds):

        obs_list=ObsList()
        # For each of these objects create an observation
        for j in range(object_data['ncutout'][i]):
            if j==0:
                continue
            start = object_data['start_row'][i][j]
            image=meds['image_cutouts'][start:start+object_data['box_size'][i]**2]
            weight=meds['weight_cutouts'][start:start+object_data['box_size'][i]**2]
            gal_jacob=Jacobian(
                row=object_data['cutout_row'][i][j],
                col=object_data['cutout_col'][i][j],
                dvdrow=object_data['dvdrow'][i][j],
                dvdcol=object_data['dvdcol'][i][j], 
                dudrow=object_data['dudrow'][i][j],
                dudcol=object_data['dudcol'][i][j])
            psf_image=meds['psf'][start:start+object_data['psf_box_size'][i]**2]
            psf_center = (object_data['psf_box_size'][i]-1)/2.
            psf_jacob=Jacobian(
                row=psf_center,
                col=psf_center,
                dvdrow=object_data['dvdrow'][i][j]/self.params['oversample'],
                dvdcol=object_data['dvdcol'][i][j]/self.params['oversample'], 
                dudrow=object_data['dudrow'][i][j]/self.params['oversample'],
                dudcol=object_data['dudcol'][i][j]/self.params['oversample'])
            # Create an obs for each cutout
            psf_obs = Observation(psf_image, jacobian=psf_jacob, meta={'offset_pixels':None})
            obs = Observation(
                image, weight=weight, jacobian=gal_jacob, psf=psf_obs, meta={'offset_pixels':None})
            obs.noise = 1./weight
            # Append the obs to the ObsList
            obs_list.append(obs)

        coadd = psc.Coadder(obs_list).coadd_obs

        j=0
        self.dump_meds_start_info(object_data,i,j)

        self.dump_meds_wcs_info(object_data,
                                i,
                                j,
                                9999,
                                9999,
                                9999,
                                9999,
                                9999,
                                9999,
                                coadd.jacobian.dudcol,
                                coadd.jacobian.dudrow,
                                coadd.jacobian.dvdcol,
                                coadd.jacobian.dvdrow,
                                coadd.jacobian.origin.col,
                                coadd.jacobian.origin.row)

        self.dump_meds_pix_info(meds,
                                i,
                                j,
                                coadd.image.flatten(),
                                coadd.image.array.flatten(),
                                coadd.psf.image.array.flatten())

class wfirst_sim(object):
    """
    WFIRST image simulation.

    Input:
    param_file : File path for input yaml config file or yaml dict. Example located at: ./example.yaml.
    """

    def __init__(self, param_file):

        if isinstance(param_file, string_types):
            # Load parameter file
            self.params     = yaml.load(open(param_file))
            self.param_file = param_file
            # Do some parsing
            for key in self.params.keys():
                if self.params[key]=='None':
                    self.params[key]=None
                if self.params[key]=='none':
                    self.params[key]=None
                if self.params[key]=='True':
                    self.params[key]=True
                if self.params[key]=='False':
                    self.params[key]=False
        else:
            # Else use existing param dict
            self.params     = param_file

        # Set up some information on processes and MPI
        if self.params['mpi']:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

        # Set up logger. I don't really use this, but it could be used.
        logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
        self.logger = logging.getLogger('wfirst_sim')

        # Initialize (pseudo-)random number generators.
        reset_rng(self,self.params['random_seed'])

        return

    def setup(self,filter_,dither,setup=False):
        """
        Set up initial objects. 

        Input:
        filter_ : A filter name. 
        """

        # Filter be present in filter_dither_dict{} (exists in survey strategy file).
        if filter_ not in filter_dither_dict.keys():
            raise ParamError('Supplied invalid filter: '+filter_)

        # This sets up a mostly-unspecified pointing object in this filter. We will later specify a dither and SCA to complete building the pointing information.
        self.pointing = pointing(self.params,self.logger,filter_=filter_,sca=None,dither=None)

        if not setup:
            # This updates the dither
            self.pointing.update_dither(dither)

        # This checks whether a truth galaxy/star catalog exist. If it doesn't exist, it is created based on specifications in the yaml file. It then sets up links to the truth catalogs on disk.
        self.cats     = init_catalogs(self.params, self.pointing, self.gal_rng, self.rank, self.size, comm=self.comm, setup=setup)


    def get_sca_list(self):
        """
        Generate list of SCAs to simulate based on input parameter file.
        """

        if hasattr(self.params,'sca'):
            if self.params['sca'] is None:
                sca_list = np.arange(1,19)
            elif hasattr(self.params['sca'],'__len__'):
                if type(self.params['sca'])==string_types:
                    raise ParamError('Provided SCA list is not numeric.')
                sca_list = self.params['sca']
            else:
                sca_list = [self.params['sca']]
        else: 
            sca_list = np.arange(1,19)

        return sca_list

    def get_inds(self):
        """
        Checks things are setup, cut out objects not near SCA, and distributes objects across procs.
        """

        # If something went wrong and there's no pointing defined, then crash. 
        if not hasattr(self,'pointing'):
            raise ParamError('Sim object has no pointing - need to run sim.setup() first.')
        if self.pointing.dither is None:
            raise ParamError('Sim pointing object has no dither assigned - need to run sim.pointing.update_dither() first.')

        # List of indices into truth input catalogs that potentially correspond to this pointing.
        # If mpi is enabled, these will be distributed uniformly between processes
        self.cats.gal_ind  = self.cats.gal_ind[self.rank::self.size]
        self.cats.gals     = self.cats.gals[self.rank::self.size]
        self.cats.star_ind = self.cats.star_ind[self.rank::self.params['starproc']]
        self.cats.stars    = self.cats.stars[self.rank::self.params['starproc']]

        mask_sca      = self.pointing.in_sca(self.cats.gals['ra'][:],self.cats.gals['dec'][:])
        mask_sca_star = self.pointing.in_sca(self.cats.stars['ra'][:],self.cats.stars['dec'][:])
        self.cats.add_mask(mask_sca,star_mask=mask_sca_star)

    def iterate_image(self):
        """
        This is the main simulation. It instantiates the draw_image object, then iterates over all galaxies and stars. The output is then accumulated from other processes (if mpi is enabled), and saved to disk.
        """

        # No objects to simulate
        if (len(self.cats.gal_ind)==0)&(len(self.cats.star_ind)==0):
            return

        # Instantiate draw_image object. The input parameters, pointing object, modify_image object, truth catalog object, random number generator, logger, and galaxy & star indices are passed.
        # Instantiation defines some parameters, iterables, and image bounds, and creates an empty SCA image.
        self.draw_image = draw_image(self.params, self.pointing, self.modify_image, self.cats,  self.logger, rank=self.rank)

        # Empty storage dictionary for postage stamp information
        gals = {}
        if self.rank==0:
            tmp,tmp_ = self.cats.get_gal_list()
            print 'Attempting to simulate '+str(len(tmp))+' galaxies for SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.'
        while True:
            # Loop over all galaxies near pointing and attempt to simulate them.
            self.draw_image.iterate_gal()
            if self.draw_image.gal_done:
                break
            # Store postage stamp output in dictionary
            g_ = self.draw_image.retrieve_stamp()
            if g_ is not None:
                gals[self.draw_image.ind] = g_

        tmp,tmp_ = self.cats.get_star_list()
        print 'Attempting to simulate '+str(len(tmp))+' stars for SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.'
        if self.rank>=self.params['starproc']:
            self.draw_image.rank=-1
        while True:
            # Loop over all stars near pointing and attempt to simulate them. Stars aren't saved in postage stamp form.
            self.draw_image.iterate_star()
            if self.draw_image.star_done:
                break

        if self.rank == 0:
            # Build file name path for SCA image
            filename = get_filename(self.params['out_path'],
                                    'images',
                                    self.params['output_meds'],
                                    var=self.pointing.filter+'_'+str(self.pointing.dither),
                                    name2=str(self.pointing.sca),
                                    ftype='fits.gz',
                                    overwrite=True)

        if self.comm is None:

            # No mpi, so just finalize the drawing of the SCA image and write it to a fits file.
            print 'Saving SCA image to '+filename
            self.draw_image.finalize_sca().write(filename)

        else:

            # Send/receive all versions of SCA images across procs and sum them, then finalize and write to fits file.
            if self.rank == 0:

                for i in range(1,self.size):
                    self.draw_image.im = self.draw_image.im + self.comm.recv(source=i)
                print 'Saving SCA image to '+filename
                # self.draw_image.im.write(filename+'_raw.fits.gz')
                self.draw_image.finalize_sca().write(filename)

            else:

                self.comm.send(self.draw_image.im, dest=0)

            # Send/receive all parts of postage stamp dictionary across procs and merge them.
            if self.rank == 0:

                for i in range(1,self.size):
                    gals.update( self.comm.recv(source=i) )

            else:

                self.comm.send(gals, dest=0)

        if self.rank == 0:
            # Build file name path for stampe dictionary pickle
            filename = get_filename(self.params['out_path'],
                                    'stamps',
                                    self.params['output_meds'],
                                    var=self.pointing.filter+'_'+str(self.pointing.dither),
                                    name2=str(self.pointing.sca),
                                    ftype='cPickle',
                                    overwrite=True)

            # Save stamp dictionary pickle
            print 'Saving stamp dict to '+filename
            save_obj(gals, filename )

            # Build indexing table for MEDS making later
            index_table = np.zeros(len(gals),dtype=[('ind',int), ('sca',int), ('dither',int), ('x',float), ('y',float), ('ra',float), ('dec',float), ('mag',float), ('stamp',int)])
            i=0
            for gal in gals:
                index_table['ind'][i]    = gals[gal]['ind']
                index_table['x'][i]      = gals[gal]['x']
                index_table['y'][i]      = gals[gal]['y']
                index_table['ra'][i]     = gals[gal]['ra']
                index_table['dec'][i]    = gals[gal]['dec']
                index_table['mag'][i]    = gals[gal]['mag']
                if gals[gal]['gal'] is not None:
                    index_table['stamp'][i]  = gals[gal]['stamp']
                else:
                    index_table['stamp'][i]  = 0
                index_table['sca'][i]    = self.pointing.sca
                index_table['dither'][i] = self.pointing.dither
                i+=1

            filename = get_filename(self.params['out_path'],
                                    'truth',
                                    self.params['output_meds'],
                                    var='index',
                                    name2=self.pointing.filter+'_'+str(self.pointing.dither)+'_'+str(self.pointing.sca),
                                    ftype='fits',
                                    overwrite=True)

            print 'Saving index to '+filename
            fio.write(filename,index_table)


# Uncomment for profiling
# pr = cProfile.Profile()

if __name__ == "__main__":
    """
    """

    # Uncomment for profiling
    # pr.enable()

    param_file = sys.argv[1]
    filter_ = sys.argv[2]
    dither = sys.argv[3]

    # This instantiates the simulation based on settings in input param file
    sim = wfirst_sim(param_file)
    # This sets up some things like input truth catalogs and empty objects
    if dither=='setup':
        sim.setup(filter_,dither,setup=True)
        sys.exit()
    elif dither=='meds':
        if len(sys.argv)!=5:
            print 'bad input format',sys.argv[0]
            sys.exit()
        if sys.argv[4]=='setup':
            setup = True
        else:
            setup = False
        meds = accumulate_output( param_file, filter_, int(sys.argv[4]), ignore_missing_files = False, setup = setup )
        sys.exit()
    else:
        sim.setup(filter_,int(dither))

    # Loop over SCAs
    for sca in sim.get_sca_list():
        # This sets up a specific pointing for this SCA (things like WCS, PSF)
        sim.pointing.update_sca(sca)
        # Select objects within some radius of pointing to attemp to simulate
        sim.get_inds()
        # This sets up the object that will simulate various wfirst detector effects, noise, etc. Instantiation creates a noise realisation for the image.
        sim.modify_image = modify_image(sim.params,sim.rng)
        # This is the main thing - iterates over galaxies for a given pointing and SCA and simulates them all
        sim.iterate_image()
        break

    # Uncomment for profiling
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('time')
    # ps.print_stats(50)
