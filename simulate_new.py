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
sedpath_Sbc    = os.path.join(galsim.meta_data.share_dir, 'SEDs', 'CWW_Sbc_ext.sed')
sedpath_Scd    = os.path.join(galsim.meta_data.share_dir, 'SEDs', 'CWW_Scd_ext.sed')
sedpath_Im     = os.path.join(galsim.meta_data.share_dir, 'SEDs', 'CWW_Im_ext.sed')
sedpath_Star   = os.path.join(galsim.meta_data.share_dir, 'SEDs', 'vega.txt')
g_band          = os.path.join(galsim.meta_data.share_dir, 'bandpasses', 'LSST_g.dat')

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

def convert_gaia_to_fits(gaiacsv='../2017-09-14-19-58-07-4430',ralims=[0,360],declims=[-90,90]):
    """
    Helper function to convert gaia data to star truth catalog.
    """

    # Need to replace with true gaia g bandpass
    g_band     = os.path.join(galsim.meta_data.share_dir, 'bandpasses', 'LSST_g.dat')
    g_band     = galsim.Bandpass(g_band, wave_type='nm').withZeropoint('AB')
    star_sed   = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')

    gaia = np.genfromtxt(gaiacsv+'.csv',dtype=None,delimiter=',',names = ['id','flux','ra','dec'],skip_header=1)
    gaia = gaia[(gaia['ra']>ralims[0])&(gaia['ra']<ralims[1])]
    gaia = gaia[(gaia['dec']>declims[0])&(gaia['dec']<declims[1])]
    out  = np.zeros(len(gaia),dtype=[('id','i4')]+[('J129','f4')]+[('F184','f4')]+[('Y106','f4')]+[('H158','f4')]+[('ra',float)]+[('dec',float)])
    out['id']  = gaia['id']
    out['ra']  = gaia['ra']
    out['dec'] = gaia['dec']
    for filter_ in ['J129','F184','Y106','H158']:
        print filter_
        bpass = wfirst.getBandpasses(AB_zeropoint=True)[filter_]
        for ind in range(len(gaia)):
            if ind%1000==0:
                print ind
            star_sed_         = star_sed.withFlux(gaia['flux'][ind],g_band)
            out[filter_][ind] = star_sed_.calculateFlux(bpass)

    fio.write('gaia_stars.fits',out,clobber=True)

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

    if os.path.exists(fpath):
        if not overwrite:
            print 'Output directory already exists. Set output_exists to True to use existing output directory at your own peril.'
            return None
    else:
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        if not os.path.exists(fpath):
            os.mkdir(fpath)

    return os.path.join(fpath,name)

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

    def update_dither(self,dither,sca):
        """
        This updates the pointing to a new dither position, replacing the stored WCS to the new WCS.

        Input
        dither     : Pointing index in the survey simulation file.
        sca        : SCA number
        """

        self.dither = dither
        self.sca    = sca

        d = fio.FITS(self.ditherfile)[-1][self.dither]

        # Check that nothing went wrong with the filter specification.
        if filter_dither_dict[self.filter] != d['filter']:
            raise ParamError('Requested filter and dither pointing do not match.')

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
        self.get_wcs() # Get the new WCS
        self.get_psf() # Get the new PSF

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

        # Discard any object greater than some dec from pointing
        if np.abs(dec-self.dec)>self.bore:
            return False

        # Position of the object in boresight coordinates
        mX  = -self.sdec   * np.cos(dec) * np.cos(self.ra-ra) + self.cdec * np.sin(dec)
        mY  =  np.cos(dec) * np.sin(self.ra-ra)

        xi  = -(self.spa * mX + self.cpa * mY) / 0.0021801102 # Image plane position in chips
        yi  =  (self.cpa * mX - self.spa * mY) / 0.0021801102

        # Check if object falls on SCA
        if        (cptr[0+12*(self.sca-1)]*xi+cptr[1+12*(self.sca-1)]*yi  \
                    <cptr[2+12*(self.sca-1)]+self.chip_enlarge)       \
                & (cptr[3+12*(self.sca-1)]*xi+cptr[4+12*(self.sca-1)]*yi  \
                    <cptr[5+12*(self.sca-1)]+self.chip_enlarge)       \
                & (cptr[6+12*(self.sca-1)]*xi+cptr[7+12*(self.sca-1)]*yi  \
                    <cptr[8+12*(self.sca-1)]+self.chip_enlarge)       \
                & (cptr[9+12*(self.sca-1)]*xi+cptr[10+12*(self.sca-1)]*yi \
                    <cptr[11+12*(self.sca-1)]+self.chip_enlarge):

            return True

        return False


    def near_pointing(self, ra, dec):
        """
        Returns objects close to pointing, using usual orthodromic distance.

        Input
        ra  : Right ascension array of objects
        dec : Declination array of objects
        """

        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)

        d2 = (x - self.cdec*self.cra)**2 + (y - self.cdec*self.sra)**2 + (z - self.sdec)**2

        return np.where(np.sqrt(d2)/2.<=self.sbore2)[0]

class init_catalogs():
    """
    Build truth catalogs if they don't exist from input galaxy and star catalogs.
    """


    def __init__(self, params, pointing, gal_rng, rank, size, comm=None):
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
                                    params['output_meds'],
                                    var=pointing.filter,
                                    name2='truth_gal',
                                    overwrite=params['overwrite'])
            # Link to galaxy truth catalog on disk 
            self.gals  = self.init_galaxy(filename,params,pointing.filter,gal_rng)
            # Link to star truth catalog on disk 
            self.stars = self.init_star(params,pointing.filter)

            # Send signal to other procs (if they exist) to let them know file is created
            for i in range(1,size):
                comm.send(None, dest=i)

        else:

            # Block until file is created
            comm.recv(source=0)

            # Set up file path. Check if output truth file path exists or if explicitly remaking galaxy properties
            filename = get_filename(params['out_path'],
                                    'truth',
                                    params['output_meds'],
                                    var=pointing.filter,
                                    name2='truth_gal',
                                    overwrite=True)
            # Link to galaxy truth catalog on disk 
            self.gals  = self.init_galaxy(filename,params,pointing.filter,gal_rng,load=True)
            # Link to star truth catalog on disk 
            self.stars = self.init_star(params,pointing.filter)

    def dump_truth_gal(self,filename,store):
        """
        Write galaxy truth catalog to disk.

        Input
        filename    : Fits filename
        store       : Galaxy truth catalog
        """

        fio.write(filename,store,clobber=True)

        return fio.FITS(filename)[-1]

    def load_truth_gal(self,filename,n_gal):
        """
        Load galaxy truth catalog from disk.

        Input
        filename    : Fits filename
        n_gal       : Length of file - used to verify nothing went wrong
        """

        store = fio.FITS(filename)[-1]

        if store.read_header()['NAXIS2']!=n_gal:
            raise ParamError('Lengths of truth array and expected number of galaxies do not match.')

        return store

    def fwhm_to_hlr(self,fwhm):
        """
        Convert full-width half-maximum to half-light radius in units of arcseconds.

        Input
        fwhm : full-width half-maximum
        """

        radius = fwhm * 0.06 / 2. # 1 pix = 0.06 arcsec, factor 2 to convert to hlr

        return radius

    def init_galaxy(self,filename,params,filter_,gal_rng,load=True):
        """
        Does the work to return a random, unique object property list (truth catalog). 

        Input
        filname : Filename of galaxy truth catalog.
        params  : Parameter dict
        filter_ : Filter name
        gal_rng : Random generator [0,1]
        load    : Force loading of file
        """

        # Make sure galaxy distribution filename is well-formed and link to it
        if isinstance(params['gal_dist'],string_types):
            # Provided an ra,dec catalog of object positions.
            radec_file = fio.FITS(params['gal_dist'])[-1]
            n_gal = radec_file.read_header()['NAXIS2'] # Number of objects in catalog
        else:
            raise ParamError('Bad gal_dist filename.')

        # This is a placeholder option to allow different galaxy simulatin methods later if necessary
        if params['gal_type'] == 0:
            # Analytic profile - sersic disk

            if load:

                # Truth file exists and no instruction to overwrite it, so load existing truth file with galaxy properties
                return self.load_truth_gal(filename,n_gal)

            else:

                # Read in file with photometry/size/redshift distribution similar to WFIRST galaxies
                phot       = fio.FITS(params['gal_sample'])[-1].read(columns=['fwhm','redshift',filter_flux_dict[filter_]])
                pind_list_ = np.ones(len(phot)).astype(bool) # storage list for original index of photometry catalog
                pind_list_ = pind_list_&(phot[filter_flux_dict[filter_]]<99)&(phot[filter_flux_dict[filter_]]>0) # remove bad mags
                pind_list_ = pind_list_&(phot['redshift']>0)&(phot['redshift']<5) # remove bad redshifts
                pind_list_ = np.where(pind_list_)[0]

                # Create minimal storage array for galaxy properties
                store = np.ones(n_gal, dtype=[('gind','i4')]
                                            +[('ra',float)]
                                            +[('dec',float)]
                                            +[('g1','f4')]
                                            +[('g2','f4')]
                                            +[('rot','f4')]
                                            +[('size','f4')]
                                            +[('z','f4')]
                                            +[('mag','f4')]
                                            +[('pind','i4')]
                                            +[('bflux','f4')]
                                            +[('dflux','f4')])
                store['ra']         = radec_file.read(columns='ra')*np.pi/180. # Right ascension
                store['dec']        = radec_file.read(columns='dec')*np.pi/180. # Declination
                store['gind']       = np.arange(n_gal) # Index array into original galaxy position catalog
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
                store['mag']        = phot[filter_flux_dict[filter_]][store['pind']] # magnitude in this filter

                # Save truth file with galaxy properties
                return self.dump_truth_gal(filename,store)

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

    def init_star(self,params,filter_):
        """
        Compiles a list of stars properties to draw. 
        Not working with new structure yet.

        Input 
        params  : parameter dict
        filter_ : Filter name
        """

        # Make sure star catalog filename is well-formed and link to it
        if isinstance(params['star_sample'],string_types):
            # Provided a catalog of star positions and properties.
            fits = fio.FITS(params['star_sample'])[-1]
            self.n_star = fits.read_header()['NAXIS2']
        else:
            return None

        stars_ = fits.read(columns=['ra','dec',filter_])

        # Create minimal storage array for galaxy properties
        stars         = np.ones(len(stars_), dtype=[('flux','f4')]+[('ra',float)]+[('dec',float)])
        stars['ra']   = stars_['ra']*np.pi/180.
        stars['dec']  = stars_['dec']*np.pi/180.
        stars['flux'] = stars_[filter_]

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

    def __init__(self, params, pointing, modify_image, cats, rng, logger, gal_ind_list=None, star_ind_list=None, stamp_size=32, num_sizes=7, image_buffer=256):
        """
        Sets up some general properties, including defining the object index lists, starting the generator iterators, assigning the SEDs (single stand-ins for now but generally red to blue for bulg/disk/knots), defining SCA bounds, and creating the empty SCA image.

        Input
        params          : parameter dict
        pointing        : Pointing object
        modify_image    : modify_image object
        cats            : init_catalots object
        rng             : Random generator
        logger          : logger instance
        gal_ind_list    : List of indices from gal truth catalog to attempt to simulate 
        star_ind_list   : List of indices from star truth catalog to attempt to simulate 
        stamp_size      : Base stamp size
        num_sizes       : Number of box sizes (will be of size np.arange(stamp_size)*stamp_size)
        image_buffer    : Number of pixels beyond SCA to attempt simulating objects that may overlap SCA
        """

        self.params       = params
        self.pointing     = pointing
        self.modify_image = modify_image
        self.cats         = cats
        self.stamp_size   = stamp_size
        self.num_sizes    = num_sizes
        if gal_ind_list is not None:
            self.gal_ind_list = gal_ind_list
        else:
            self.gal_ind_list = np.arange(self.cats.gals.read_header()['NAXIS2'])
        if star_ind_list is not None:
            self.star_ind_list = star_ind_list
        else:
            self.star_ind_list = np.arange(self.cats.stars.read_header()['NAXIS2'])
        self.gal_iter   = 0
        self.star_iter  = 0
        self.rng        = rng
        self.gal_done   = False
        self.star_done  = False

        # Setup galaxy SED
        # Need to generalize to vary sed based on input catalog
        self.galaxy_sed_b = galsim.SED(sedpath_Sbc, wave_type='Ang', flux_type='flambda')
        self.galaxy_sed_d = galsim.SED(sedpath_Scd, wave_type='Ang', flux_type='flambda')
        self.galaxy_sed_n = galsim.SED(sedpath_Im,  wave_type='Ang', flux_type='flambda')
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


    def iterate_gal(self):
        """
        Iterator function to loop over all possible galaxies to draw
        """

        # Check if the end of the galaxy list has been reached; return exit flag (gal_done) True
        # You'll have a bad day if you aren't checking for this flag in any external loop...
        # self.gal_done = True
        # return
        if self.gal_iter == len(self.gal_ind_list):
            self.gal_done = True
            return 

        # if self.gal_iter>1000:
        #     self.gal_done = True
        #     return             

        if self.gal_iter%1000==0:
            print 'Progress: Attempting to simulate galaxy '+str(self.gal_iter)+' in SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.'

        # Galaxy truth index for this galaxy
        self.ind       = self.gal_ind_list[self.gal_iter]
        self.gal_iter += 1

        # Galaxy truth array for this galaxy
        self.gal       = self.cats.gals[self.ind]

        # Reset galaxy information
        self.gal_model = None
        self.gal_stamp = None

        # If galaxy doesn't actually fall within rough simulate-able bounds, return (faster)
        if not self.pointing.in_sca(self.gal['ra'][0],self.gal['dec'][0]):
            return 

        # If galaxy image position (from wcs) doesn't fall within simulate-able bounds, skip (slower) 
        # If it does, draw it
        if self.check_position(self.gal['ra'][0],self.gal['dec'][0]):
            self.draw_galaxy()

    def iterate_star(self):
        """
        Iterator function to loop over all possible stars to draw
        """

        self.star_done = True
        return 
        # Don't draw stars into postage stamps
        if not self.params['draw_sca']:
            self.star_done = True
            return 
        # Check if the end of the star list has been reached; return exit flag (gal_done) True
        # You'll have a bad day if you aren't checking for this flag in any external loop...
        if self.star_iter == len(self.star_ind_list):
            self.star_done = True
            return 

        if self.star_iter%1000==0:
            print 'Progress: Attempting to simulate star '+str(self.star_iter)+' in SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.'

        # Star truth index for this galaxy
        self.ind       = self.star_ind_list[self.star_iter]
        self.star_iter += 1

        # Star truth array for this galaxy
        self.star      = self.cats.stars[self.ind]

        # If star doesn't actually fall within rough simulate-able bounds, return (faster)
        if not self.pointing.in_sca(self.star['ra'],self.star['dec']):
            return 

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
        """

        # Redshift SED
        sed_       = sed.atRedshift(self.gal['z'][0])
        
        # Apply correct flux from magnitude for filter bandpass
        sed_       = sed_.withMagnitude(self.gal['mag'][0], self.pointing.bpass) 

        # Return model with SED applied
        return model * sed_

    def galaxy_model(self):
        """
        Generate the intrinsic galaxy model based on truth catalog parameters
        """

        # Generate galaxy model
        # Calculate flux fraction of disk portion 
        flux = (1.-self.gal['bflux'][0]) * self.gal['dflux'][0]
        if flux > 0:
            # If any flux, build Sersic disk galaxy (exponential) and apply appropriate SED
            self.gal_model = galsim.Sersic(1, half_light_radius=1.*self.gal['size'][0], flux=flux, trunc=5.*self.gal['size'][0])
            self.gal_model = self.make_sed_model(self.gal_model, self.galaxy_sed_d)

        # Calculate flux fraction of knots portion 
        flux = (1.-self.gal['bflux'][0]) * (1.-self.gal['dflux'][0])
        if flux > 0:
            # If any flux, build star forming knots model and apply appropriate SED
            knots = galsim.RandomWalk(npoints=self.params['knots'], half_light_radius=1.*self.gal['size'][0], flux=flux, rng=self.rng) 
            knots = self.make_sed_model(knots, self.galaxy_sed_n)
            # Sum the disk and knots, then apply intrinsic ellipticity to the disk+knot component. Fixed intrinsic shape, but can be made variable later.
            self.gal_model = galsim.Add([self.gal_model, knots])
            self.gal_model = self.gal_model.shear(e1=0.25, e2=0.25)
 
        # Calculate flux fraction of bulge portion 
        flux = self.gal['bflux']
        if flux > 0:
            # If any flux, build Sersic bulge galaxy (de vacaleurs) and apply appropriate SED
            bulge = galsim.Sersic(4, half_light_radius=1.*self.gal['size'][0], flux=flux, trunc=5.*self.gal['size'][0]) 
            # Apply intrinsic ellipticity to the bulge component. Fixed intrinsic shape, but can be made variable later.
            bulge = bulge.shear(e1=0.25, e2=0.25)
            # Apply the SED
            bulge = self.make_sed_model(bulge, self.galaxy_sed_b)

            if self.gal_model is None:
                # No disk or knot component, so save the galaxy model as the bulge part
                self.gal_model = bulge
            else:
                # Disk/knot component, so save the galaxy model as the sum of two parts
                self.gal_model = galsim.Add([self.gal_model, bulge])

        # Random rotation (pairs of objects are offset by pi/2 to cancel shape noise)
        self.gal_model = self.gal_model.rotate(self.gal['rot'][0]*galsim.radians) 


    def galaxy(self):
        """
        Call galaxy_model() to get the intrinsic galaxy model, then apply properties relevant to its observation
        """

        # Build intrinsic galaxy model
        self.galaxy_model()

        # Apply a shear
        self.gal_model = self.gal_model.shear(g1=self.gal['g1'][0],g2=self.gal['g1'][0]) 
        # Rescale flux appropriately for wfirst
        self.gal_model = self.gal_model * galsim.wfirst.collecting_area * galsim.wfirst.exptime

        # Ignoring chromatic stuff for now for speed, so save correct flux of object
        flux = self.gal_model.calculateFlux(self.pointing.bpass)
        # Evaluate the model at the effective wavelength of this filter bandpass (should change to effective SED*bandpass?)
        # This makes the object achromatic, which speeds up drawing and convolution
        self.gal_model  = self.gal_model.evaluateAtWavelength(self.pointing.bpass.effective_wavelength)
        # Reassign correct flux
        self.gal_model  = self.gal_model.withFlux(flux) # reapply correct flux
        
        sky_level = wfirst.getSkyLevel( self.pointing.bpass, 
                                        world_pos=self.radec, 
                                        date=self.pointing.date)
        sky_level *= (1.0 + wfirst.stray_light_fraction)*wfirst.pixel_scale**2
        if sky_level/flux < galsim.GSParams().folding_threshold:
            gsparams = galsim.GSParams( folding_threshold=sky_level/flux,
                                        maximum_fft_size=16384 )
        else:
            gsparams = galsim.GSParams( maximum_fft_size=16384 )

        # Convolve with PSF
        self.gal_model = galsim.Convolve(self.gal_model, self.pointing.PSF, gsparams=gsparams) 
 
        # Convolve with additional los motion (jitter), if any
        if 'los_motion' in self.params:
            los = galsim.Gaussian(fwhm=2.*np.sqrt(2.*np.log(2.))*self.params['los_motion'])
            los = los.shear(g1=0.3,g2=0.) # assymetric jitter noise
            self.gal_model = galsim.Convolve(self.gal_model, los)

        # chromatic stuff replaced by above lines
        # # Draw galaxy igal into stamp.
        # self.gal_list[igal].drawImage(self.pointing.bpass[self.params['filter']], image=gal_stamp)
        # # Add detector effects to stamp.

    def star_model(self, sed = None, flux = 1.):
        """
        Create star model for PSF or for drawing stars into SCA

        Input
        sed  : The stellar SED
        flux : The flux of the star
        """

        # Generate star model (just a delta function) and apply SED
        if sed is not None:
            sed_ = sed.withFlux(flux, self.pointing.bpass)
            self.st_model = galsim.DeltaFunction() * sed_
        else:
            self.st_model = galsim.DeltaFunction(flux=flux)

        sky_level = wfirst.getSkyLevel(self.pointing.bpass, 
                                        world_pos=self.radec, 
                                        date=self.pointing.date)
        sky_level *= (1.0 + wfirst.stray_light_fraction)*wfirst.pixel_scale**2

        if sky_level/flux < galsim.GSParams().folding_threshold:
            gsparams = galsim.GSParams( folding_threshold=sky_level/flux,
                                        maximum_fft_size=16384 )
        else:
            gsparams = galsim.GSParams( maximum_fft_size=16384 )

        # Evaluate the model at the effective wavelength of this filter bandpass (should change to effective SED*bandpass?)
        # This makes the object achromatic, which speeds up drawing and convolution
        self.st_model = self.st_model.evaluateAtWavelength(self.pointing.bpass.effective_wavelength)

        # Convolve with PSF
        # if flux!=1.:
        #     self.st_model = galsim.Convolve(self.st_model, self.pointing.PSF, galsim.Pixel(wfirst.pixel_scale), gsparams=big_fft_params)
        # else:
        self.st_model = galsim.Convolve(self.st_model, self.pointing.PSF, gsparams=gsparams)

        # Convolve with additional los motion (jitter), if any
        if 'los_motion' in self.params:
            los = galsim.Gaussian(fwhm=2.*np.sqrt(2.*np.log(2.))*self.params['los_motion'])
            los = los.shear(g1=0.3,g2=0.) # assymetric jitter noise
            self.st_model = galsim.Convolve(self.st_model, los)

        # old chromatic version
        # self.psf_list[igal].drawImage(self.pointing.bpass[self.params['filter']],image=psf_stamp, wcs=local_wcs)

    def get_stamp_size(self,obj,factor=5):
        """
        Select the stamp size multiple to use.

        Input
        obj    : Galsim object
        factor : Factor to multiple suggested galsim stamp size by
        """

        # return int(obj.getGoodImageSize(wfirst.pixel_scale) * factor) / self.stamp_size
        return int(self.gal['size'][0]/wfirst.pixel_scale * factor) / self.stamp_size + 1

    def draw_galaxy(self):
        """
        Draw the galaxy model into the SCA (neighbors and blending) and/or the postage stamp (isolated).
        """

        # Build galaxy model that will be drawn into images
        self.galaxy()

        stamp_size = self.get_stamp_size(self.gal_model)

        # # Skip drawing some really huge objects (>twice the largest stamp size)
        # if stamp_size>2.*self.num_sizes:
        #     return

        # Create postage stamp bounds at position of object
        b = galsim.BoundsI( xmin=self.xyI.x-int(stamp_size*self.stamp_size)/2,
                            ymin=self.xyI.y-int(stamp_size*self.stamp_size)/2,
                            xmax=self.xyI.x+int(stamp_size*self.stamp_size)/2,
                            ymax=self.xyI.y+int(stamp_size*self.stamp_size)/2)

        # If this postage stamp doesn't overlap the SCA bounds at all, no reason to draw anything
        if not (b&self.b).isDefined():
            return

        # Create postage stamp for galaxy
        gal_stamp = galsim.Image(b, wcs=self.pointing.WCS)

        # Draw galaxy model into postage stamp. This is the basis for both the postage stamp output and what gets added to the SCA image. This will obviously create biases if the postage stamp is too small - need to monitor that.
        self.gal_model.drawImage(image=gal_stamp,offset=self.offset,method='phot')
        # gal_stamp.write(str(self.ind)+'.fits')

        # Add galaxy stamp to SCA image
        if self.params['draw_sca']:
            self.im[b&self.b] = self.im[b&self.b] + gal_stamp[b&self.b]

        # If object too big for stamp sizes, skip saving a stamp
        if stamp_size>=self.num_sizes:
            print 'too big stamp',stamp_size,stamp_size*self.stamp_size
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
        self.star_model(sed=self.star_sed,flux=self.star['flux']*galsim.wfirst.collecting_area/7000.*galsim.wfirst.exptime)

        # Get good stamp size multiple for star
        stamp_size = self.get_stamp_size(self.st_model)
        print 'stamp',stamp_size

        # Create postage stamp bounds for star
        b = galsim.BoundsI( xmin=self.xyI.x-int(stamp_size*self.stamp_size)/2,
                            ymin=self.xyI.y-int(stamp_size*self.stamp_size)/2,
                            xmax=self.xyI.x+int(stamp_size*self.stamp_size)/2,
                            ymax=self.xyI.y+int(stamp_size*self.stamp_size)/2 )
        print b

        # If postage stamp doesn't overlap with SCA, don't draw anything
        if not (b&self.b).isDefined():
            return

        print b&self.b

        # Create star postage stamp
        star_stamp = galsim.Image(b, wcs=self.pointing.WCS)
        print galsim.GSParams(),self.st_model.gsparams

        # Draw star model into postage stamp
        self.st_model.drawImage(image=star_stamp,offset=self.offset)

        star_stamp.write('/fs/scratch/cond0083/wfirst_sim_out/images/'+str(self.ind)+'.fits.gz')

        # Add star stamp to SCA image
        self.im[b&self.b] = self.im[b&self.b] + star_stamp[b&self.b]
        # self.st_model.drawImage(image=self.im,add_to_image=True,offset=self.xy-self.im.true_center,method='no_pixel')

    def retrieve_stamp(self):
        """
        Helper function to accumulate various information about a postage stamp and return it in dictionary form.
        """

        if self.gal_stamp is None:
            return None

        return {'ra'     : self.gal['ra'][0], # ra of galaxy
                'dec'    : self.gal['dec'][0], # dec of galaxy
                'x'      : self.xy.x, # SCA x position of galaxy
                'y'      : self.xy.y, # SCA y position of galaxy
                'stamp'  : self.get_stamp_size(self.gal_model)*self.stamp_size, # Get stamp size in pixels
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
        return self.modify_image.add_effects(self.im,self.pointing,radec,self.pointing.WCS)[0]

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
        self.reset_rng()

        return

    def reset_rng(self):
        """
        Reset the (pseudo-)random number generators.
        """

        self.rng     = galsim.BaseDeviate(self.params['random_seed'])
        self.gal_rng = galsim.UniformDeviate(self.params['random_seed'])

        return

    def setup(self,filter_):
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
        # This checks whether a truth galaxy/star catalog exist. If it doesn't exist, it is created based on specifications in the yaml file. It then sets up links to the truth catalogs on disk.
        self.cats     = init_catalogs(self.params, self.pointing, self.gal_rng, sim.rank, sim.size, comm=self.comm)

    def get_inds(self):
        """
        Selects all objects within some radius of the pointing to attempt to simulate.
        """

        # If something went wrong and there's no pointing defined, then crash. 
        if not hasattr(self,'pointing'):
            raise ParamError('Sim object has no pointing - need to run sim.setup() first.')
        if self.pointing.dither is None:
            raise ParamError('Sim pointing object has no dither assigned - need to run sim.pointing.update_dither() first.')

        # List of indices into truth input catalogs that potentially correspond to this pointing.
        # If mpi is enabled, these will be distributed uniformly between processes
        # That's only useful if the input catalog is unordered in position on the sky
        self.gal_ind  = sim.pointing.near_pointing(self.cats.gals['ra'][:], self.cats.gals['dec'][:])[self.rank::self.size]
        self.star_ind = sim.pointing.near_pointing(self.cats.stars['ra'][:], self.cats.stars['dec'][:])[self.rank::self.size]

    def iterate_image(self):
        """
        This is the main simulation. It instantiates the draw_image object, then iterates over all galaxies and stars. The output is then accumulated from other processes (if mpi is enabled), and saved to disk.
        """

        # No galaxies to simulate
        if len(self.gal_ind)==0:
            return

        # Instantiate draw_image object. The input parameters, pointing object, modify_image object, truth catalog object, random number generator, logger, and galaxy & star indices are passed.
        # Instantiation defines some parameters, iterables, and image bounds, and creates an empty SCA image.
        self.draw_image = draw_image(self.params, self.pointing, self.modify_image, self.cats, self.rng, self.logger, gal_ind_list=self.gal_ind, star_ind_list=self.star_ind)

        # Empty storage dictionary for postage stamp information
        gals = {}
        print 'Attempting to simulate '+str(len(self.gal_ind))+' galaxies for SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.'
        while True:
            # Loop over all galaxies near pointing and attempt to simulate them.
            self.draw_image.iterate_gal()
            if self.draw_image.gal_done:
                break
            # Store postage stamp output in dictionary
            g_ = self.draw_image.retrieve_stamp()
            if g_ is not None:
                gals[self.draw_image.ind] = g_

        print 'Attempting to simulate '+str(len(self.star_ind))+' stars for SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.'
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



    # Need to integrate this into writing of fits files as a call after the last exposure has been run to place in coadd (0) position. -troxel
    def get_coadd(self,chunk,index,):

        meds_data = meds.MEDS(self.meds_filename(chunk))
        num=meds_data['number'][index]
        ncutout=meds_data['ncutout'][index]
        obs_list=ObsList()
        # For each of these objects create an observation
        for cutout_index in range(ncutout):
            image=meds_data.get_cutout(index, cutout_index)
            weight=meds_data.get_cweight_cutout(index, cutout_index)
            meds_jacob=meds_data.get_jacobian(index, cutout_index)
            gal_jacob=Jacobian(
                row=meds_jacob['row0'],col=meds_jacob['col0'],
                dvdrow=meds_jacob['dvdrow'],
                dvdcol=meds_jacob['dvdcol'], dudrow=meds_jacob['dudrow'],
                dudcol=meds_jacob['dudcol'])
            psf_image=meds_data.get_psf(index, cutout_index)
            psf_jacob=Jacobian(
                row=31.5,col=31.5,dvdrow=meds_jacob['dvdrow'],
                dvdcol=meds_jacob['dvdcol'], dudrow=meds_jacob['dudrow'],
                dudcol=meds_jacob['dudcol'])
            # Create an obs for each cutout
            psf_obs = Observation(psf_image, jacobian=psf_jacob, meta={'offset_pixels':None})
            obs = Observation(
                image, weight=weight, jacobian=gal_jacob, psf=psf_obs, meta={'offset_pixels':None})
            obs.noise = 1./weight
            # Append the obs to the ObsList
            obs_list.append(obs)

        return psc.Coadder(obs_list).coadd_obs

# Uncomment for profiling
# pr = cProfile.Profile()

if __name__ == "__main__":
    """
    """

    # Uncomment for profiling
    # pr.enable()

    param_file = sys.argv[1]
    filter_ = sys.argv[2]
    dither = int(sys.argv[3])

    # This instantiates the simulation based on settings in input param file
    sim = wfirst_sim(param_file)
    # This sets up some things like input truth catalogs and empty objects
    sim.setup(filter_)

    # Loop over SCAs
    for sca in np.arange(1,19):
        # This sets up a specific pointing (things like WCS, PSF)
        sim.pointing.update_dither(dither,sca)
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

