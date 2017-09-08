"""
A modular implementation of galaxy and star image simulations for WFIRST 
requirements building. Built from the WFIRST GalSim module. An example 
config file is provided as example.yaml.

Built from galsim demo13...
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
import sys, os
import math
import logging
import time
import yaml
import galsim as galsim
import galsim.wfirst as wfirst
import galsim.config.process as process
import galsim.des as des
import fitsio as fio
import os
from astropy.time import Time

path, filename = os.path.split(__file__)
sedpath = os.path.join(galsim.meta_data.share_dir, 'SEDs', 'CWW_Sbc_ext.sed')
py3 = sys.version_info[0] == 3
if py3:
    string_types = str,
else:
    string_types = basestring,

class ParamError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

MAX_RAD_FROM_BORESIGHT = 0.009

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

t0=time.time()

def convert_dither_to_fits(ditherfile='observing_sequence_hlsonly'):

    dither = np.genfromtxt(ditherfile+'.dat',dtype=None,names = ['date','f1','f2','ra','dec','pa','program','filter','f8','f9','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21'])
    dither=dither[['date','ra','dec','pa','filter']][dither['program']==5]
    fio.write(ditherfile+'.fits',dither,clobber=True)

    return

def create_radec_fits(ra=[25.,27.5],dec=[-27.5,-25.],n=1500000):

    ra1 = np.random.rand(n)*(ra[1]-ra[0])/180.*np.pi+ra[0]/180.*np.pi
    d0 = (np.cos((dec[0]+90)/180.*np.pi)+1)/2.
    d1 = (np.cos((dec[1]+90)/180.*np.pi)+1)/2.
    dec1 = np.arccos(2*np.random.rand(n)*(d1-d0)+2*d0-1)
    out = np.empty(n,dtype=[('ra',float)]+[('dec',float)])
    out['ra']=ra1*180./np.pi
    out['dec']=dec1*180./np.pi-90
    fio.write('ra_'+str(ra[0])+'_'+str(ra[1])+'_dec_'+str(dec[0])+'_'+str(dec[1])+'_n_'+str(n)+'.fits.gz',out,clobber=True)


def radec_to_chip(obsRA, obsDec, obsPA, ptRA, ptDec):
    """
    Converted from Chris' c code. Used here to limit ra, dec catalog to objects that fall in each pointing.
    """

    AFTA_SCA_Coords = np.array([
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

    sort  = np.argsort(ptDec)
    ptRA  = ptRA[sort]
    ptDec = ptDec[sort]
    # Crude cut of some objects more than some encircling radius away from the boresight - creates a fast dec slice. Probably not worth doing better than this.
    begin = np.searchsorted(ptDec, obsDec-MAX_RAD_FROM_BORESIGHT)
    end   = np.searchsorted(ptDec, obsDec+MAX_RAD_FROM_BORESIGHT)

    # Position of the object in boresight coordinates
    mX  = -np.sin(obsDec)*np.cos(ptDec[begin:end])*np.cos(obsRA-ptRA[begin:end]) + np.cos(obsDec)*np.sin(ptDec[begin:end])
    mY  = np.cos(ptDec[begin:end])*np.sin(obsRA-ptRA[begin:end])

    xi  = -(np.sin(obsPA)*mX + np.cos(obsPA)*mY) / 0.0021801102 # Image plane position in chips
    yi  =  (np.cos(obsPA)*mX - np.sin(obsPA)*mY) / 0.0021801102
    SCA = np.zeros(end-begin)
    for i in range(18):
        cptr = AFTA_SCA_Coords
        mask = (cptr[0+12*i]*xi+cptr[1+12*i]*yi<cptr[2+12*i]) \
                & (cptr[3+12*i]*xi+cptr[4+12*i]*yi<cptr[5+12*i]) \
                & (cptr[6+12*i]*xi+cptr[7+12*i]*yi<cptr[8+12*i]) \
                & (cptr[9+12*i]*xi+cptr[10+12*i]*yi<cptr[11+12*i])
        SCA[mask] = i+1

    return np.pad(SCA,(begin,len(ptDec)-end),'constant',constant_values=(0, 0))[np.argsort(sort)] # Pad SCA array with zeros and resort to original indexing

class pointing(object): # need to pass date probably...
    """
    A class object to store information about a pointing. This includes the WCS, bandpasses, and PSF for each SCA.

    Optional input:
    ra          : Right ascension of pointing in degrees [default: 90.]
    dec         : Declination of pointing in degrees [default: -10.]
    PA          : Pointing angle of pointing in degrees [default: None]
                  None indicates to use the ideal orientation relative to sun - see GalSim documentation.
    PA_is_FPA   : Pointing angle is of focal plane (not telescope) [default: True]
    SCA         : A single SCA number (1-18) to initiate wcs, PSF information for. [default: None]
                  None indicates to use all SCAs.
    logger      : A GalSim logger instance [default: None]
                  None indicates to instantiate a new logger.
    """

    def __init__(self, params, ra=90., dec=-10., PA=None, filter_=None, date=None, PA_is_FPA=True, SCA=None, logger=None):
        """
        Intiitate pointing class object. Store pointing parameters, bandpasses, SCAs, 
        and instantiate wcs and PSF for those SCAs.
        """

        self.ra         = ra  * galsim.degrees
        self.dec        = dec * galsim.degrees
        self.PA         = PA  * galsim.degrees
        self.PA_is_FPA  = PA_is_FPA
        self.date       = date
        self.bpass      = wfirst.getBandpasses()

        if SCA is None:
            self.SCA    = np.arange(18,dtype=int)+1
        else:
            self.SCA    = [SCA]

        if logger is None:
            logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
            # In non-script code, use getself.logger(__name__) at module scope instead.
            self.logger = logging.getself.logger('wfirst_pointing')
        else:
            self.logger = logger

        self.get_wcs()
        self.init_psf(approximate_struts=params['approximate_struts'], n_waves=params['n_waves'],filter_=filter_)

        return


    def get_wcs(self):
        """
        Instantiate wcs solution for the requested SCAs.
        """

        # Convert pointing position to CelestialCoord object.
        pointing_pos = galsim.CelestialCoord(ra=self.ra, dec=self.dec)

        # Get the WCS for an observation at this position. We are not supplying a date, so the routine
        # will assume it's the vernal equinox. The output of this routine is a dict of WCS objects, one 
        # for each SCA. We then take the WCS for the SCA that we are using.
        self.WCS = wfirst.getWCS(world_pos=pointing_pos, PA=self.PA, date=self.date, SCAs=self.SCA, PA_is_FPA=self.PA_is_FPA)

        # We also record the center position for these SCAs. We'll tell it to give us a CelestialCoord
        # corresponding to (X, Y) = (wfirst.n_pix/2, wfirst.n_pix/2).
        self.SCA_centpos = {}
        for SCA in self.SCA:
            self.SCA_centpos[SCA] = self.WCS[SCA].toWorld(galsim.PositionD(wfirst.n_pix/2, wfirst.n_pix/2))

        return

    def init_psf(self, approximate_struts=False, n_waves=None,filter_=None):
        """
        Instantiate PSF for the requested SCAs.

        Input:
        approximate_struts  : Whether to approximate the effect of the struts. [default: False]
        n_waves             : Number of wavelengths to use for setting up interpolation of the 
                              chromatic PSF objects. [default: None]

        Set True, some integer (e.g., 10), respectively, to speed up and produce an approximate PSF.
        """

        # Here we carry out the initial steps that are necessary to get a fully chromatic PSF.  We use
        # the getPSF() routine in the WFIRST module, which knows all about the telescope parameters
        # (diameter, bandpasses, obscuration, etc.).

        # self.logger.info('Doing expensive pre-computation of PSF.')
        t0 = time.time()
        # self.logger.setLevel(logging.DEBUG)

        if filter_ is None:
            self.PSF = wfirst.getPSF(SCAs=self.SCA, approximate_struts=approximate_struts, n_waves=n_waves, logger=self.logger)
        else:
            self.PSF = wfirst.getPSF(SCAs=self.SCA, approximate_struts=approximate_struts, n_waves=n_waves, logger=self.logger, wavelength=filter_.effective_wavelength)

        # self.logger.setLevel(logging.INFO)
        self.logger.info('Done PSF precomputation in %.1f seconds!'%(time.time()-t0))

        return

def fwhm_to_hlr(fwhm):

    radius = fwhm*0.06/2. # 1 pix = 0.06 arcsec, factor 2 to convert to hlr

    return radius

class wfirst_sim(object):
    """
    WFIRST image simulation.

    Input:
    param_file : File path for input yaml config file. Example located at: ./example.yaml.
    """

    def __init__(self,param_file):

        # Load parameter file
        self.params = yaml.load(open(param_file))
        self.param_file = param_file
        # Do some parsing
        for key in self.params.keys():
            if self.params[key]=='None':
                self.params[key]=None

        self.filter = self.params['use_filters']

        # Instantiate GalSim logger
        logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
        # In non-script code, use getself.logger(__name__) at module scope instead.
        self.logger = logging.getLogger('wfirst_sim')

        # Initialize (pseudo-)random number generator.
        self.reset_rng()

        # Where to find and output data.
        path, filename = os.path.split(__file__)
        self.out_path = os.path.abspath(os.path.join(path, self.params['out_path']))

        # Make output directory if not already present.
        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)

        # Set total number of unique objects
        if self.params['n_gal'] is not None:
            # Specify number of unique objects in params file. 
            # If you provide a file for ra,dec positions of objects, uses n_gal random positions from file.
            # If you provide a number density in the focal plane, uses n_gal random positions in each SCA.
            self.n_gal = self.params['n_gal']
        else:
            if isinstance(self.params['gal_dist'],string_types):
                self.n_gal = fio.FITS(self.params['gal_dist'])[-1].read_header()['NAXIS2']

        # Check that various params make sense
        if self.params['gal_n_use']>self.n_gal:
            raise ParamError('gal_n_use should be <= n_gal.')


        # Read in the WFIRST filters, setting an AB zeropoint appropriate for this telescope given its
        # diameter and (since we didn't use any keyword arguments to modify this) using the typical
        # exposure time for WFIRST images.  By default, this routine truncates the parts of the
        # bandpasses that are near 0 at the edges, and thins them by the default amount.
        self.filters = wfirst.getBandpasses(AB_zeropoint=True)
        self.logger.debug('Read in WFIRST imaging filters.')

        return

    def reset_rng(self):
        """
        Reset the (pseudo-)random number generators.
        """

        self.rng = galsim.BaseDeviate(self.params['random_seed'])
        self.gal_rng = galsim.UniformDeviate(self.params['random_seed'])

        return

    def init_galaxy(self):
        """
        Does the heavy work to return a unique object list with gal_n_use objects. 
        gal_n_use should be <= self.n_gal, and allows you to lower the 
        overhead of creating unique objects. Really only impactful when using real 
        cosmos objects. Reads in and stores ra,dec coordinates from file.

        output:
        self.obj_list
        self.pind_list
        self.radec
        self.gind_list
        """

        self.logger.info('Pre-processing for galaxies started.')

        if isinstance(self.params['gal_dist'],string_types):
            # Provided an ra,dec catalog of object positions.
            radec_file     = fio.FITS(self.params['gal_dist'])[-1]
            self.ra = radec_file.read(columns='ra')*np.pi/180.
            self.dec = radec_file.read(columns='dec')*np.pi/180.            
        else:
            raise ParamError('Bad gal_dist filename.')

        if self.params['gal_type'] == 0:
            # Analytic profile - sersic disk

            tasks = []
            for i in range(self.params['nproc']):
                tasks.append({
                    'n_gal':self.n_gal,
                    'nproc':self.params['nproc'],
                    'proc':i,
                    'phot_file':self.params['gal_sample'],
                    'filter_':self.filter,
                    'timing':self.params['timing'],
                    'seed':self.params['random_seed'],
                    'disk_n':self.params['disk_n'],
                    'shear_list':self.params['shear_list']})

            tasks = [ [(job, k)] for k, job in enumerate(tasks) ]

            results = process.MultiProcess(self.params['nproc'], {}, init_galaxy_loop, tasks, 'init_galaxy', logger=self.logger, done_func=None, except_func=None, except_abort=True)

            if len(results) != self.params['nproc']:
                print 'something went wrong with init_galaxy parallelisation'
                raise
            for i in range(len(results)):
                if i==0:
                    pind_list, rot_list, e_list, obj_list = results[i]
                else:
                    pind_list_, rot_list_, e_list_, obj_list_ = results[i]
                    pind_list.update(pind_list_)
                    rot_list.update(rot_list_)
                    e_list.update(e_list_)
                    obj_list.update(obj_list_)

            sim.dump_truth_gal(e_list,rot_list,pind_list)

        else:
            pass # cosmos gal not guaranteed to work. uncomment at own risk 
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

        self.logger.debug('Pre-processing for galaxies completed.')

        return 

    def galaxy(self):
        """
        Return a list of galaxy objects that fall on the SCAs over a given flux distribution, drawn 
        from the unique image list generated in init_galaxy(). 
        Convert ra,dec to xy for SCAs.

        needs:
        self.use_ind
        self.radec
        self.pind_list
        self.obj_list

        output:
        self.sca
        self.xy
        self.gal_list
        self.psf_list
        self.rot_list
        self.orig_pind_list
        self.e_list
        """

        # self.logger.info('Compiling x,y,ra,dec positions of catalog.')

        # Reset random number generators to make each call of galaxy() deterministic within a run.
        if self.params['timing']:
            print 'begin galaxy',time.time()-t0

        self.reset_rng()

        if hasattr(self,'obj_list'):
            if not hasattr(self,'use_ind'):
                print 'Assuming use_ind = All objects.'
                self.use_ind = np.arange(self.n_gal)

            # Get SCAs for each object. Remove indices that don't fall on an SCA.
            self.SCA  = []
            for i,ind in enumerate(self.use_ind):
                sca = galsim.wfirst.findSCA(self.pointing.WCS, self.radec[ind])
                self.SCA.append(sca)
            if self.params['timing']:
                print 'sca list',time.time()-t0

            self.use_ind = [self.use_ind[i] for i in range(len(self.SCA)) if self.SCA[i] is not None]
            self.SCA     = [self.SCA[i] for i in range(len(self.SCA)) if self.SCA[i] is not None]

            if self.params['timing']:
                print 'sca done',time.time()-t0
            if len(self.SCA)==0:
                return True

            # Already calculated ra,dec distribution, so only need to calculate xy for this pointing.
            self.xy             = []
            self.gal_list       = []
            self.psf_list       = []  # Added by AC
            if self.params['timing']:
                print 'before ind loop',len(self.use_ind),time.time()-t0
            for i,ind in enumerate(self.use_ind):
                if self.params['timing']:
                    if i%100==0:
                        print 'inside ind loop',i,ind,time.time()-t0
                # Save xy positions for this SCA corresponding to the ra,dec.
                self.xy.append(self.pointing.WCS[self.SCA[i]].toImage(self.radec[ind]))

                obj = self.obj_list[ind]
                self.gal_list.append(galsim.Convolve(obj, self.pointing.PSF[self.SCA[i]])) # Convolve with PSF and append to final image list
                psf = galsim.DeltaFunction(flux=1.) * obj.SED
                self.psf_list.append(galsim.Convolve(psf, self.pointing.PSF[self.SCA[i]]))  # Added by AC

        else:
            raise ParamError('Need to run init_galaxy() first.')

        return False

    def star(self):
        """
        Return a list of star objects for psf measurement... Not done yet, but requires minimal 
        cleaning up to work in the same way as galaxy(). Currently just the example code from demo13.
        """

        # Drawing PSF.  Note that the PSF object intrinsically has a flat SED, so if we convolve it
        # with a galaxy, it will properly take on the SED of the galaxy.  For the sake of this demo,
        # we will simply convolve with a 'star' that has a flat SED and unit flux in this band, so
        # that the PSF image will be normalized to unit flux. This does mean that the PSF image
        # being drawn here is not quite the right PSF for the galaxy.  Indeed, the PSF for the
        # galaxy effectively varies within it, since it differs for the bulge and the disk.  To make
        # a real image, one would have to choose SEDs for stars and convolve with a star that has a
        # reasonable SED, but we just draw with a flat SED for this demo.
        out_filename = os.path.join(self.out_path, 'demo13_PSF_{0}.fits'.format(filter_name))
        # Approximate a point source.
        point = galsim.Gaussian(sigma=1.e-8, flux=1.)
        # Use a flat SED here, but could use something else.  A stellar SED for instance.
        # Or a typical galaxy SED.  Depending on your purpose for drawing the PSF.
        star_sed = galsim.SED(lambda x:1, 'nm', 'flambda').withFlux(1.,filter_)  # Give it unit flux in this filter.
        star = galsim.Convolve(point*star_sed, PSF)
        img_psf = galsim.ImageF(self.params['stamp_size'], self.params['stamp_size'])
        star.drawImage(bandpass=filter_, image=img_psf, scale=wfirst.pixel_scale)
        img_psf.write(out_filename)
        self.logger.debug('Created PSF with flat SED for {0}-band'.format(filter_name))

        return

    def init_noise_model(self):
        """
        Generate a poisson noise model.
        """

        self.noise = galsim.PoissonNoise(self.rng)
        self.logger.info('Poisson noise model created.')
        
        return 

    def add_effects(self,im,i,wpos,xy):
        """
        Add detector effects for WFIRST.

        Input:

        im      : Postage stamp or image.
        wpos    : World position (ra,dec).
        xy      : Pixel position (x,y).
        date    : Date of pointing (future proofing - need to be implemented in pointing class still).

        Preserve order:
        1) add_background
        2) add_poisson_noise
        3) recip_failure 
        4) quantize
        5) dark_current
        6) nonlinearity
        7) interpix_cap
        8) e_to_ADU
        9) quantize


        Where does persistence get added? Immediately before/after background?
        """

        # save_image = final_image.copy()

        # # If we had wanted to, we could have specified a different exposure time than the default
        # # one for WFIRST, but otherwise the following routine does not take any arguments.
        # wfirst.addReciprocityFailure(final_image)
        # logger.debug('Included reciprocity failure in {0}-band image'.format(filter_name))

        # if diff_mode:
        #     # Isolate the changes due to reciprocity failure.
        #     diff = final_image-save_image

        #     out_filename = os.path.join(outpath,'demo13_RecipFail_{0}.fits'.format(filter_name))
        #     final_image.write(out_filename)
        #     out_filename = os.path.join(outpath,
        #                                 'demo13_diff_RecipFail_{0}.fits'.format(filter_name))
        #     diff.write(out_filename)

        # im.write('tmpa.fits')
        if self.params['use_background']:
            im, sky_image = self.add_background(im,i,wpos,xy) # Add background to image and save background
            # im.write('tmpb.fits')

        if self.params['use_poisson_noise']:
            im = self.add_poisson_noise(im) # Add poisson noise to image
            # im.write('tmpc.fits')

        if self.params['use_recip_failure']:
            im = self.recip_failure(im) # Introduce reciprocity failure to image
            # im.write('tmpd.fits')

        im.quantize() # At this point in the image generation process, an integer number of photons gets detected
        # im.write('tmpe.fits')

        if self.params['use_dark_current']:
            im = self.dark_current(im) # Add dark current to image
            # im.write('tmpf.fits')

        if self.params['use_nonlinearity']:
            im = self.nonlinearity(im) # Apply nonlinearity
            # im.write('tmpg.fits')

        if self.params['use_interpix_cap']:
            im = self.interpix_cap(im) # Introduce interpixel capacitance to image.
            # im.write('tmph.fits')

        im = self.e_to_ADU(im) # Convert electrons to ADU

        im.quantize() # Finally, the analog-to-digital converter reads in an integer value.
        # im.write('tmpi.fits')

        # Note that the image type after this step is still a float. If we want to actually
        # get integer values, we can do new_img = galsim.Image(im, dtype=int)

        # Since many people are used to viewing background-subtracted images, we return a
        # version with the background subtracted (also rounding that to an int).
        if self.params['use_background']:
            im = self.finalize_background_subtract(im,sky_image)
            # im.write('tmpj.fits')

        # get weight map
        sky_image.invertSelf()

        return im, sky_image

    def add_background(self,im,i,wpos,xy):
        """
        Add backgrounds to image (sky, thermal).

        First we get the amount of zodaical light for a position corresponding to the position of 
        the object. The results are provided in units of e-/arcsec^2, using the default WFIRST
        exposure time since we did not explicitly specify one. Then we multiply this by a factor
        >1 to account for the amount of stray light that is expected. If we do not provide a date
        for the observation, then it will assume that it's the vernal equinox (sun at (0,0) in
        ecliptic coordinates) in 2025.
        """

        sky_level = wfirst.getSkyLevel(self.filters[self.filter], world_pos=wpos, date=self.pointing.date)
        sky_level *= (1.0 + wfirst.stray_light_fraction)
        # Make a image of the sky that takes into account the spatially variable pixel scale. Note
        # that makeSkyImage() takes a bit of time. If you do not care about the variable pixel
        # scale, you could simply compute an approximate sky level in e-/pix by multiplying
        # sky_level by wfirst.pixel_scale**2, and add that to final_image.

        local_wcs = self.pointing.WCS[self.SCA[i]].local(xy)
        sky_stamp = galsim.Image(self.params['stamp_size'], self.params['stamp_size'], wcs=local_wcs)
        local_wcs.makeSkyImage(sky_stamp, sky_level)
        # im_sky.write('tmpa3.fits')
        # This image is in units of e-/pix. Finally we add the expected thermal backgrounds in this
        # band. These are provided in e-/pix/s, so we have to multiply by the exposure time.
        sky_stamp += wfirst.thermal_backgrounds[self.filter]*wfirst.exptime
        # im_sky.write('tmpa4.fits')
        # Adding sky level to the image.
        im += sky_stamp
        # im_sky.write('tmpa5.fits')
        # im.write('tmpa6.fits')
        
        return im,sky_stamp

    def add_poisson_noise(self,im):
        """
        Add pre-initiated poisson noise to image.
        """

        # Check if noise initiated
        if not hasattr(self,'noise'):
            self.init_noise_model()
            self.logger.info('Initialising poisson noise model.')

        im.addNoise(self.noise)

        return im

    def recip_failure(self,im):
        """
        Introduce reciprocity failure to image.

        Reciprocity, in the context of photography, is the inverse relationship between the
        incident flux (I) of a source object and the exposure time (t) required to produce a given
        response(p) in the detector, i.e., p = I*t. However, in NIR detectors, this relation does
        not hold always. The pixel response to a high flux is larger than its response to a low
        flux. This flux-dependent non-linearity is known as 'reciprocity failure', and the
        approximate amount of reciprocity failure for the WFIRST detectors is known, so we can
        include this detector effect in our images.
        """

        # If we had wanted to, we could have specified a different exposure time than the default
        # one for WFIRST, but otherwise the following routine does not take any arguments.
        wfirst.addReciprocityFailure(im)
        # self.logger.debug('Included reciprocity failure in image')

        return im

    def dark_current(self,im):
        """
        Adding dark current to the image.

        Even when the detector is unexposed to any radiation, the electron-hole pairs that
        are generated within the depletion region due to finite temperature are swept by the
        high electric field at the junction of the photodiode. This small reverse bias
        leakage current is referred to as 'dark current'. It is specified by the average
        number of electrons reaching the detectors per unit time and has an associated
        Poisson noise since it is a random event.
        """

        dark_current = wfirst.dark_current*wfirst.exptime
        dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(self.rng, dark_current))
        im.addNoise(dark_noise)

        # NOTE: Sky level and dark current might appear like a constant background that can be
        # simply subtracted. However, these contribute to the shot noise and matter for the
        # non-linear effects that follow. Hence, these must be included at this stage of the
        # image generation process. We subtract these backgrounds in the end.

        # self.logger.debug('Applied nonlinearity to image')
        return im

    def nonlinearity(self,im):
        """
        Applying a quadratic non-linearity.

        Note that users who wish to apply some other nonlinearity function (perhaps for other NIR
        detectors, or for CCDs) can use the more general nonlinearity routine, which uses the
        following syntax:
        final_image.applyNonlinearity(NLfunc=NLfunc)
        with NLfunc being a callable function that specifies how the output image pixel values
        should relate to the input ones.
        """

        # Apply the WFIRST nonlinearity routine, which knows all about the nonlinearity expected in
        # the WFIRST detectors.
        wfirst.applyNonlinearity(im)

        return im

    def interpix_cap(self,im):
        """
        Including Interpixel capacitance

        The voltage read at a given pixel location is influenced by the charges present in the
        neighboring pixel locations due to capacitive coupling of sense nodes. This interpixel
        capacitance effect is modeled as a linear effect that is described as a convolution of a
        3x3 kernel with the image. The WFIRST IPC routine knows about the kernel already, so the
        user does not have to supply it.
        """
        wfirst.applyIPC(im)
        # self.logger.debug('Applied interpixel capacitance to image')

        return im


    def add_read_noise(self,im):
        """
        Adding read noise

        Read noise is the noise due to the on-chip amplifier that converts the charge into an
        analog voltage.  We already applied the Poisson noise due to the sky level, so read noise
        should just be added as Gaussian noise:
        """

        read_noise = galsim.GaussianNoise(self.rng, sigma=wfirst.read_noise)
        im.addNoise(read_noise)
        # self.logger.debug('Added readnoise to image')

        return im

    def e_to_ADU(self,im):
        """
        We divide by the gain to convert from e- to ADU. Currently, the gain value in the WFIRST
        module is just set to 1, since we don't know what the exact gain will be, although it is
        expected to be approximately 1. Eventually, this may change when the camera is assembled,
        and there may be a different value for each SCA. For now, there is just a single number,
        which is equal to 1.
        """

        return im/wfirst.gain


    def finalize_sky_im(self,im):
        """
        Finalize sky background for subtraction from final image. Add dark current, 
        convert to analog voltage, and quantize.
        """

        if (self.params['sub_true_background'])&(self.params['use_dark_current']):
            final_im = (im + round(wfirst.dark_current*wfirst.exptime))
        final_im = self.e_to_ADU(final_im)
        final_im.quantize()

        return final_im

    def finalize_background_subtract(self,im,sky):
        """
        Finalize background subtraction of image.
        """

        sky.quantize() # Quantize sky
        sky = self.finalize_sky_im(sky) # Finalize sky with dark current, convert to ADU, and quantize.
        im -= sky

        return im

    def draw_galaxy(self,igal,ind):
        """
        Draw a postage stamp for one of the galaxy objects using the local wcs for its position in the SCA plane. Apply add_effects.
        """

        # if self.params['timing']:
        #     print 'before wcs',time.time()-t0
        # Get local wcs solution at galaxy position in SCA.
        local_wcs = self.pointing.WCS[self.SCA[igal]].local(self.xy[igal])
        # if self.params['timing']:
        #     print 'after wcs',time.time()-t0
        # Create stamp at this position.
        gal_stamp = galsim.Image(self.params['stamp_size'], self.params['stamp_size'], wcs=local_wcs)
        # if self.params['timing']:
        #     print 'after gal stamp',time.time()-t0

        # ignoring chromatic stuff for now
        gal  = self.gal_list[igal]
        flux = gal.calculateFlux(self.pointing.bpass[self.filter])
        gal  = gal.evaluateAtWavelength(self.filters[self.filter].effective_wavelength)
        gal  = gal.withFlux(flux)
        # if self.params['timing']:
        #     print 'after gal eff lambda',time.time()-t0
        gal.drawImage(image=gal_stamp)
        # gal_stamp.write('tmp'+str(igal)+'.fits')
        # if self.params['timing']:
        #     print 'after gal draw',time.time()-t0
        # replaced by above lines
        # # Draw galaxy igal into stamp.
        # self.gal_list[igal].drawImage(self.filters[self.filter], image=gal_stamp)
        # # Add detector effects to stamp.

        gal_stamp, weight_stamp = self.add_effects(gal_stamp,igal,self.radec[ind],self.xy[igal])
        # if self.params['timing']:
        #     print 'after add effects',time.time()-t0

        if self.params['draw_true_psf']:
            # Also draw the true PSF
            # if self.params['timing']:
            #     print 'before psf',time.time()-t0
            psf_stamp = galsim.ImageF(gal_stamp.bounds) # Use same bounds as galaxy stamp
            # if self.params['timing']:
            #     print 'after psf stamp',time.time()-t0
            # Draw the PSF
            # new effective version for speed
            psf = self.psf_list[igal]
            psf = psf.evaluateAtWavelength(self.filters[self.filter].effective_wavelength)
            # if self.params['timing']:
            #     print 'after psf eff lambda',time.time()-t0
            psf.drawImage(image=psf_stamp,wcs=local_wcs)
            # if self.params['timing']:
            #     print 'after psf draw',time.time()-t0
            # old chromatic version
            # self.psf_list[igal].drawImage(self.pointing.bpass[self.filter],image=psf_stamp, wcs=local_wcs)

            #galaxy_sed = galsim.SED(
            #    os.path.join(sedpath, 'CWW_Sbc_ext.sed'), wave_type='Ang', flux_type='flambda').withFlux(
            #    1.,self.pointing.bpass[self.filter])
            #self.pointing.PSF[self.SCA[igal]] *= galaxy_sed
            #pointing_psf = galsim.Convolve(galaxy_sed, self.pointing.PSF[self.SCA[igal]])
            #self.pointing.PSF[self.SCA[igal]].drawImage(self.pointing.bpass[self.filter],image=psf_stamp, wcs=local_wcs)
            #pointing_psf = galaxy_sed * self.pointing.PSF[self.SCA[igal]]
            #pointing_psf.drawImage(self.pointing.bpass[self.filter],image=psf_stamp, wcs=local_wcs)
            #self.pointing.PSF[self.SCA[igal]].drawImage(self.pointing.bpass[self.filter],image=psf_stamp, wcs=local_wcs)

            return gal_stamp, local_wcs, weight_stamp, psf_stamp
        else:
            return gal_stamp, local_wcs, weight_stamp

    def near_pointing(self, obsRA, obsDec, obsPA, ptRA, ptDec):
        """
        Returns mask of objects too far from pointing.
        """

        if not hasattr(self,'_x'):
            self._x = np.cos(ptDec) * np.cos(ptRA)
            self._y = np.cos(ptDec) * np.sin(ptRA)
            self._z = np.sin(ptDec)

        d2 = (self._x - np.cos(obsDec)*np.cos(obsRA))**2 + (self._y - np.cos(obsDec)*np.sin(obsRA))**2 + (self._z - np.sin(obsDec))**2
        dist = 2.*np.arcsin(np.sqrt(d2)/2.)
        # print MAX_RAD_FROM_BORESIGHT,dist[np.where(dist<=MAX_RAD_FROM_BORESIGHT)[0]]

        return np.where(dist<=MAX_RAD_FROM_BORESIGHT)[0]

    def dither_sim(self):

        # Loops over dithering file
        t0=time.time()

        # Read dither file
        dither = fio.FITS(self.params['dither_file'])[-1].read()
        objs        = []

        mask = (dither['ra']>24)&(dither['ra']<28.5)&(dither['dec']>-28.5)&(dither['dec']<-24)&(dither['filter'] == filter_dither_dict[self.filter]) # isolate relevant pointings

        import pickle
        def save_obj(obj, name ):
            with open(name, 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        save_obj(self.obj_list,'tmp.pickle')

        tasks = []
        for i in range(self.params['nproc']):
            d = np.where(mask)[0][i::int(self.params['nproc'])]
            tasks.append({
                'd_':d,
                'ra':self.ra,
                'dec':self.dec,
                'param_file':self.param_file,
                'filter_':self.filter,
                'obj_list':self.obj_list})

        tasks = [ [(job, k)] for k, job in enumerate(tasks) ]

        results = process.MultiProcess(self.params['nproc'], {}, dither_loop, tasks, 'dithering', logger=self.logger, done_func=None, except_func=None, except_abort=True)

        for i in range(len(results)):
            if i == 0:
                gal_exps, psf_exps, wcs_exps, wgt_exps, dither_list, sca_list = results[i]
            else:
                gal_exps_, psf_exps_, wcs_exps_, wgt_exps_, dither_list_, sca_list_ = results[i]
                for ind in gal_exps_.keys():
                    if ind not in gal_exps.keys():
                        gal_exps[ind]      = gal_exps_[ind]
                        psf_exps[ind]      = psf_exps_[ind]
                        wcs_exps[ind]      = wcs_exps_[ind]
                        wgt_exps[ind]      = wgt_exps_[ind]
                        dither_list[ind]   = dither_list_[ind]
                        sca_list[ind]      = sca_list_[ind]
                    else:
                        for j in range(len(gal_exps_[ind])):
                            gal_exps[ind].append(gal_exps_[ind][j])
                            psf_exps[ind].append(psf_exps_[ind][j])
                            wcs_exps[ind].append(wcs_exps_[ind][j])
                            wgt_exps[ind].append(wgt_exps_[ind][j])
                            dither_list[ind].append(dither_list_[ind][j])
                            sca_list[ind].append(sca_list_[ind][j])

        results[i] = []

        for i in range(self.n_gal):
            if i in gal_exps.keys():
                print 'len 4',len(gal_exps[i]),gal_exps[i]
                obj = des.MultiExposureObject(gal_exps[i], psf=psf_exps[i], wcs=wcs_exps[i], weight=wgt_exps[i], id=i)
                objs.append(obj)
                gal_exps[i]=[]
                psf_exps[i]=[]
                wcs_exps[i]=[]
                wgt_exps[i]=[]

        sim.dump_meds(objs)
        sim.dump_truth_ind(dither_list,sca_list)

        return 

    def dump_meds(self,objs):
        """
        Accepts a list of meds MultiExposureObject's and writes to meds file.
        """

        filename = self.params['output_meds']+'_'+self.filter+'.fits.gz'
        des.WriteMEDS(objs, filename, clobber=True)

        return

    def dump_truth_gal(self,e_list,rot_list,pind_list):
        """
        Accepts a list of meds MultiExposureObject's and writes to meds file.
        """
        # import pickle
        # def save_obj(obj, name ):
        #     with open(name, 'wb') as f:
        #         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        filename = self.params['output_meds']+'_'+self.filter+'_truth_gal.fits.gz'
        out = np.ones(self.n_gal, dtype=[('gal_index',int)]+[('g1',float)]+[('g2',float)]+[('rot_angle',float)]+[('phot_index',int)])
        for name in out.dtype.names:
            out[name] *= -999
        for i,ind in enumerate(e_list.keys()):
            out['gal_index'][i]    = ind
            out['g1'][i]           = self.params['shear_list'][e_list[ind]][0]
            out['g2'][i]           = self.params['shear_list'][e_list[ind]][1]
            out['rot_angle'][i]    = rot_list[ind]
            out['phot_index'][i]   = pind_list[ind]

        fio.write(filename,out,clobber=True)

        return

    def dump_truth_ind(self,dither_list,sca_list):
        """
        Accepts a list of meds MultiExposureObject's and writes to meds file.
        """
        # import pickle
        # def save_obj(obj, name ):
        #     with open(name, 'wb') as f:
        #         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        depth = 0
        for ind in dither_list.keys():
            if len(dither_list[ind])>depth:
                depth = len(dither_list[ind])

        filename = self.params['output_meds']+'_'+self.filter+'_truth_ind.fits.gz'
        out = np.ones(self.n_gal, dtype=[('gal_index',int)]+[('dither_index',int,(depth))]+[('sca',int,(depth))])
        for name in out.dtype.names:
            out[name] *= -999
        for ind in dither_list.keys():
            stop = len(dither_list[ind])
            out['dither_index'][i][:stop] = dither_list[ind]
            out['sca'][i][:stop]          = sca_list[ind]

        fio.write(filename,out,clobber=True)

        return

def init_galaxy_loop(n_gal=None,nproc=None,proc=None,phot_file=None,filter_=None,timing=None,seed=None,shear_list=None,disk_n=None,**kwargs):

    gal_rng   = galsim.UniformDeviate(seed+proc)

    fits      = fio.FITS(phot_file)[-1]
    mag_dist  = fits.read(columns=filter_flux_dict[filter_]) # magnitudes
    size_dist = fwhm_to_hlr(fits.read(columns='fwhm'))
    z_dist    = fits.read(columns='redshift')

    pind_list_ = np.ones(fits.read_header()['NAXIS2']).astype(bool) # storage list for original index of photometry catalog
    pind_list_ = pind_list_&(mag_dist<99)&(mag_dist>0) # remove bad mags
    pind_list_ = pind_list_&(size_dist*2.*0.06/wfirst.pixel_scale<16) # remove large objects to maintain 32x32 stamps
    pind_list_ = np.where(pind_list_)[0]

    pind_list = {}
    rot_list  = {}
    e_list    = {}
    obj_list  = {}

    for i in range(n_gal):
        if i%nproc!=proc:
            continue
        if timing:
            if i%100==0:
                print 'inside init_gal loop',i,time.time()-t0

        pind_list[i] = pind_list_[int(gal_rng()*len(pind_list_))]
        rot_list[i] = gal_rng()*360.
        e_list[i] = int(gal_rng()*len(shear_list))
        obj = galsim.Sersic(disk_n, half_light_radius=1.*size_dist[pind_list[i]])
        obj = obj.rotate(rot_list[i]*galsim.degrees)
        obj = obj.shear(g1=shear_list[e_list[i]][0],g2=shear_list[e_list[i]][1])
        galaxy_sed = galsim.SED(sedpath, wave_type='nm', flux_type='fphotons').withMagnitude(mag_dist[pind_list[i]],wfirst.getBandpasses()[filter_]) * galsim.wfirst.collecting_area * galsim.wfirst.exptime
        galaxy_sed = galaxy_sed.atRedshift(z_dist[pind_list[i]])
        obj = obj * galaxy_sed
        obj_list[i] = obj

    return pind_list, rot_list, e_list, obj_list

def recover_sim_object(param_file,filter_,ra,dec,obj_list):

    sim = wfirst_sim(param_file)
    sim.init_noise_model()
    sim.filter    = filter_
    sim.obj_list  = obj_list
    sim.radec     = []
    for i in range(len(ra)):
        sim.radec.append(galsim.CelestialCoord(ra[i]*galsim.radians,dec[i]*galsim.radians))

    return sim

def dither_loop(d_ = None,
                ra = None,
                dec = None,
                param_file = None,
                filter_ = None,
                obj_list = None,
                **kwargs):
    """

    """

    gal_exps    = {}
    wcs_exps    = {}
    wgt_exps    = {}
    psf_exps    = {}
    dither_list = {}
    sca_list    = {}

    sim = recover_sim_object(param_file,filter_,ra,dec,obj_list)

    dither = fio.FITS(sim.params['dither_file'])[-1].read()
    date   = Time(dither['date'],format='mjd').datetime
    # cnt=0

    for d in d_:

        # Find objects near pointing.
        sim.use_ind = sim.near_pointing(dither['ra'][d]*np.pi/180., dither['dec'][d]*np.pi/180., dither['pa'][d]*np.pi/180., ra, dec)
        if len(sim.use_ind)==0: # If no galaxies in focal plane, skip dither
            continue
        # if cnt>10:
        #     break
        # cnt+=1
        # sim.use_ind=sim.use_ind[:100]
        if sim.params['timing']:
            print 'after use_ind',time.time()-t0

        # This instantiates a pointing object to be iterated over in some way
        # Return pointing object with wcs, psf, etc information.
        sim.pointing = pointing(sim.params,
                                ra=dither['ra'][d], 
                                dec=dither['dec'][d], 
                                PA=dither['pa'][d], 
                                filter_=sim.filters[sim.filter],
                                date=date[d],
                                SCA=None,
                                PA_is_FPA=True, 
                                logger=sim.logger)
        if sim.params['timing']:
            print 'pointing',time.time()-t0
        skip = sim.galaxy()
        if sim.params['timing']:
            print 'galaxy',time.time()-t0
        if skip:
            continue
        #sim..star()

        # u,c = np.unique(sim.SCA,return_counts=True)
        # print 'number of objects in SCAs',u,c

        # print 'before draw galaxy',time.time()-t0
        for i,ind in enumerate(sim.use_ind):
            if sim.params['timing']:
                if i%100==0:
                    print 'drawing galaxy ',i,time.time()-t0
            out = sim.draw_galaxy(i,ind)
            if ind in gal_exps.keys():
                gal_exps[ind].append(out[0])
                wcs_exps[ind].append(out[1])
                wgt_exps[ind].append(out[2])
                if sim.params['draw_true_psf']:
                    psf_exps[ind].append(out[3]) 
                dither_list[ind].append(d)
                sca_list[ind].append(sim.SCA[i])
            else:
                gal_exps[ind]     = [out[0]]
                wcs_exps[ind]     = [out[1]]
                wgt_exps[ind]     = [out[2]]
                if sim.params['draw_true_psf']:
                    psf_exps[ind] = [out[3]] 
                dither_list[ind]  = [d]
                sca_list[ind]     = [sim.SCA[i]]

    return gal_exps, psf_exps, wcs_exps, wgt_exps, dither_list, sca_list


if __name__ == "__main__":
    """
    """

    # This instantiates the simulation based on settings in input param file (argv[1])
    sim = wfirst_sim(sys.argv[1])

    if sim.params['timing']:
        print 'before init galaxy',time.time()-t0
    # Initiate unique galaxy image list and noise models
    sim.init_galaxy()
    if sim.params['timing']:
        print 'after init galaxy',time.time()-t0

    # noise now made in dither loop to allow parallelisation
    # sim.init_noise_model()
    # if sim.params['timing']:
    #     print 'after noise',time.time()-t0

    # Dither function that loops over pointings, SCAs, objects for each filter loop.
    # Returns a meds MultiExposureObject of galaxy stamps, psf stamps, and wcs.
    if sim.params['timing']:
        print 'before dither sim',time.time()-t0
    sim.dither_sim()

