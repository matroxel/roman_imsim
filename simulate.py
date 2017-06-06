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
import galsim.des as des
import fitsio as fio

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

    def __init__(self, params, ra=90., dec=-10., PA=None, date=None, PA_is_FPA=True, SCA=None, logger=None):
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
        self.init_psf(approximate_struts=params['approximate_struts'], n_waves=params['n_waves'])

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

    def init_psf(self, approximate_struts=False, n_waves=None):
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

        self.logger.info('Doing expensive pre-computation of PSF.')
        t0 = time.time()
        self.logger.setLevel(logging.DEBUG)

        self.PSF = wfirst.getPSF(SCAs=self.SCA, approximate_struts=approximate_struts, n_waves=n_waves, logger=self.logger)

        self.logger.setLevel(logging.INFO)
        self.logger.info('Done PSF precomputation in %.1f seconds!'%(time.time()-t0))

        return

class wfirst_sim(object):
    """
    WFIRST image simulation.

    Input:
    param_file : File path for input yaml config file. Example located at: ./example.yaml.
    """

    def __init__(self,param_file):

        # Load parameter file
        self.params = yaml.load(open(param_file))
        # Do some parsing
        for key in self.params.keys():
            if self.params[key]=='None':
                self.params[key]=None
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
                radec_file = fio.FITS(self.params['gal_dist'])[-1].read()
                self.n_gal = len(radec_file)

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

    def fwhm_to_hlr(self,fwhm):

        radius = fwhm*0.06/2. # 1 pix = 0.06 arcsec, factor 2 to convert to hlr

        return radius

    def init_galaxy(self):
        """
        Does the heavy work to return a unique object list with gal_n_use objects. 
        gal_n_use should be <= self.n_gal, and allows you to lower the 
        overhead of creating unique objects. Really only impactful when using real 
        cosmos objects. Reads in and stores ra,dec coordinates from file.
        """

        self.logger.info('Pre-processing for galaxies started.')
        if self.params['gal_type'] == 0:
            # Analytic profile - sersic disk
            # Read distribution of sizes (fwhm, converted to scale radius)

            fits = fio.FITS(self.params['gal_sample'])[-1]
            pind_list = np.ones(fits.read_header()['NAXIS2']).astype(bool) # storage list for original index of photometry catalog
            for filter in filter_flux_dict.keys(): # Loop over filters
                mag_dist = fits.read(columns=filter_flux_dict[filter]) # magnitudes
                pind_list = pind_list&(mag_dist<99)&(mag_dist>0) # remove bad mags

            size_dist = fits.read(columns='fwhm')
            size_dist = self.fwhm_to_hlr(size_dist)
            pind_list = pind_list&(size_dist*2.*0.06/wfirst.pixel_scale<16) # remove large objects to maintain 32x32 stamps
            pind_list = np.where(pind_list)[0]
            self.obj_list = []
            self.pind_list  = []
            for i in range(self.params['gal_n_use']):
                # Create unique object list of length gal_n_use, each with unique size.
                ind = pind_list[int(self.gal_rng()*len(pind_list))]
                self.pind_list.append(ind)
                self.obj_list.append(galsim.Sersic(self.params['disk_n'], half_light_radius=1.*size_dist[ind]))
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

        if isinstance(self.params['gal_dist'],string_types):
            # Provided an ra,dec catalog of object positions.
            radec_file     = fio.FITS(self.params['gal_dist'])[-1].read()
            self.radec     = []
            self.gind_list = []
            for i in range(self.n_gal):
                # Select a random ra,dec position n_gal times.
                self.gind_list.append(i) # Save link to unique object index
                # Allows removal of duplicates - doesn't matter for postage stamp sims?
                self.radec.append(galsim.CelestialCoord(radec_file['ra'][i]*galsim.degrees,radec_file['dec'][i]*galsim.degrees))
        else:
            raise ParamError('Bad gal_dist filename.')

        self.logger.debug('Pre-processing for galaxies completed.')

        return radec_file['ra'][self.gind_list],radec_file['dec'][self.gind_list]

    def galaxy(self):
        """
        Return a list of galaxy objects that fall on the SCAs over a given flux distribution, drawn 
        from the unique image list generated in init_galaxy(). 
        Convert ra,dec to xy for SCAs.
        """

        # self.logger.info('Compiling x,y,ra,dec positions of catalog.')

        # Reset random number generators to make each call of galaxy() deterministic within a run.
        self.reset_rng()

        if hasattr(self,'radec'):
            if not hasattr(self,'use_ind'):
                print 'Assuming use_ind = All objects.'
                self.use_ind = np.arange(self.n_gal)

            # Include random fluxes, rotations for all objects drawn from the unique image list generated by init_galaxy().
            # Magnitudes are drawn from a file containing distributions.
            mag_dist = fio.FITS(self.params['gal_sample'])[-1].read(columns=filter_flux_dict[self.filter]) # magnitudes

            # Get SCAs for each object. Remove indices that don't fall on an SCA.
            self.SCA  = []
            for i,ind in enumerate(self.use_ind):
                sca = galsim.wfirst.findSCA(self.pointing.WCS, self.radec[ind])
                self.SCA.append(sca)

            self.use_ind = [self.use_ind[i] for i in range(len(self.SCA)) if self.SCA[i] is not None]
            self.SCA     = [self.SCA[i] for i in range(len(self.SCA)) if self.SCA[i] is not None]

            if len(self.SCA)==0:
                return True

            # Already calculated ra,dec distribution, so only need to calculate xy for this pointing.
            self.xy    = []
            self.gal_list  = []
            self.rot_list = {}
            self.orig_pind_list = {}
            self.e_list = {}
            for i,ind in enumerate(self.use_ind):
                # Save xy positions for this SCA corresponding to the ra,dec.
                self.xy.append(self.pointing.WCS[self.SCA[i]].toImage(self.radec[ind]))
                # gind_list is meant (more useful in the future) to preserve the link to the original unique galaxy list 
                # for writing exposure lists in meds files
                if not ind in self.orig_pind_list.keys():
                    self.orig_pind_list[ind] = int(self.gal_rng()*len(self.pind_list)) # unique object index
                find = self.pind_list[self.orig_pind_list[ind]] # Random flux index
                obj  = self.obj_list[self.orig_pind_list[ind]].copy() # Copy object image
                if not ind in self.rot_list.keys():
                    self.rot_list[ind] = self.gal_rng()*360.
                obj  = obj.rotate(self.rot_list[ind]*galsim.degrees) # Rotate randomly
                sed  = galsim.SED(lambda x:1, 'nm', 'flambda').withMagnitude(mag_dist[find],self.pointing.bpass[self.filter]) # Create tmp achromatic sed object with right magnitude
                flux = sed.calculateFlux(self.pointing.bpass[self.filter]) # calculate correct flux
                obj  = obj.withFlux(flux) # Set random flux
                if not ind in self.e_list.keys():
                    self.e_list[ind] = int(self.gal_rng()*len(self.params['shear_list']))
                obj = obj.shear(g1=self.params['shear_list'][self.e_list[ind]][0],g2=self.params['shear_list'][self.e_list[ind]][1])
                self.gal_list.append(galsim.Convolve(obj, self.pointing.PSF[self.SCA[i]])) # Convolve with PSF and append to final image list

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

    def add_effects(self,im,i,wpos,xy,date=None):
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

        if self.params['use_background']:
            im, sky_image = self.add_background(im,i,wpos,xy,date=date) # Add background to image and save background

        if self.params['use_poisson_noise']:
            im = self.add_poisson_noise(im) # Add poisson noise to image

        if self.params['use_recip_failure']:
            im = self.recip_failure(im) # Introduce reciprocity failure to image

        im.quantize() # At this point in the image generation process, an integer number of photons gets detected

        if self.params['use_dark_current']:
            im = self.dark_current(im) # Add dark current to image

        if self.params['use_nonlinearity']:
            im = self.nonlinearity(im) # Apply nonlinearity

        if self.params['use_interpix_cap']:
            im = self.interpix_cap(im) # Introduce interpixel capacitance to image.

        im = self.e_to_ADU(im) # Convert electrons to ADU

        im.quantize() # Finally, the analog-to-digital converter reads in an integer value.

        # Note that the image type after this step is still a float. If we want to actually
        # get integer values, we can do new_img = galsim.Image(im, dtype=int)

        # Since many people are used to viewing background-subtracted images, we return a
        # version with the background subtracted (also rounding that to an int).
        if self.params['use_background']:
            im = self.finalize_background_subtract(im,sky_image)

        return im

    def add_background(self,im,i,wpos,xy,date=None):
        """
        Add backgrounds to image (sky, thermal).

        First we get the amount of zodaical light for a position corresponding to the position of 
        the object. The results are provided in units of e-/arcsec^2, using the default WFIRST
        exposure time since we did not explicitly specify one. Then we multiply this by a factor
        >1 to account for the amount of stray light that is expected. If we do not provide a date
        for the observation, then it will assume that it's the vernal equinox (sun at (0,0) in
        ecliptic coordinates) in 2025.
        """

        sky_level = wfirst.getSkyLevel(self.filters[self.filter], world_pos=wpos)
        sky_level *= (1.0 + wfirst.stray_light_fraction)
        # Make a image of the sky that takes into account the spatially variable pixel scale. Note
        # that makeSkyImage() takes a bit of time. If you do not care about the variable pixel
        # scale, you could simply compute an approximate sky level in e-/pix by multiplying
        # sky_level by wfirst.pixel_scale**2, and add that to final_image.
        im_sky = im.copy()
        local_wcs = self.pointing.WCS[self.SCA[i]].local(xy)
        local_wcs.makeSkyImage(im_sky, sky_level)
        # This image is in units of e-/pix. Finally we add the expected thermal backgrounds in this
        # band. These are provided in e-/pix/s, so we have to multiply by the exposure time.
        im_sky += wfirst.thermal_backgrounds[self.filter]*wfirst.exptime
        # Adding sky level to the image.
        im += im_sky

        return im,im_sky

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

        # Get local wcs solution at galaxy position in SCA.
        local_wcs = self.pointing.WCS[self.SCA[igal]].local(self.xy[igal])
        # Create stamp at this position.
        gal_stamp = galsim.Image(self.params['stamp_size'], self.params['stamp_size'], wcs=local_wcs)
        # Draw galaxy igal into stamp.
        self.gal_list[igal].drawImage(self.filters[self.filter], image=gal_stamp)
        # Add detector effects to stamp.
        gal_stamp = self.add_effects(gal_stamp,igal,self.radec[ind],self.xy[igal])

        if self.params['draw_true_psf']:
            # Also draw the true PSF
            psf_stamp = galsim.ImageF(gal_stamp.bounds) # Use same bounds as galaxy stamp
            # Draw the PSF
            self.pointing.PSF[self.SCA[igal]].drawImage(self.pointing.bpass[self.filter],image=psf_stamp, wcs=local_wcs)

            return gal_stamp, local_wcs, psf_stamp
        else:
            return gal_stamp, local_wcs

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

    def dither_sim(self,ra,dec):

        # Loops over dithering file
        from astropy.time import Time

        # Read dither file
        dither = fio.FITS(self.params['dither_file'])[-1].read()
        date   = Time(dither['date'],format='mjd').datetime
        ra*=np.pi/180. # convert to radians
        dec*=np.pi/180.

        for filter in filter_flux_dict.keys(): # Loop over filters
            self.filter = filter
            objs        = []
            gal_exps    = {}
            wcs_exps    = {}
            psf_exps    = {}
            self.sca_list    = {}
            self.dither_list = {}
            for i in self.gind_list:
                gal_exps[i]    = []
                wcs_exps[i]    = []
                psf_exps[i]    = []
                self.sca_list[i]    = []
                self.dither_list[i] = []

            cnt = 0
            for d in (np.where(dither['filter'] == filter_dither_dict[filter])[0]): # Loop over dithers in each filer
                if d%1000==0:
                    print 'dither',d
                if cnt>1:
                    break
                # Temporary skipping of exposure along edge of file to not waste time making pointing for things with no objects in SCAs
                # if (dither['ra'][d]>31)&(dither['ra'][d]<39)&(dither['dec'][d]>-29)&(dither['dec'][d]<-21):
                #     pass
                # else:
                #     continue
                # Calculate which SCAs the galaxies fall on in this dither
                # SCAs = radec_to_chip(dither['ra'][d]*np.pi/180., dither['dec'][d]*np.pi/180., (dither['pa'][d]+90.)*np.pi/180., ra, dec)
                # if np.all(SCAs==0): # If no galaxies in focal plane, skip dither
                #     continue
                # print 'my sca list',dither['ra'][d],dither['dec'][d],SCAs[SCAs!=0],np.where(SCAs!=0)[0]

                # Find objects near pointing.
                self.use_ind = self.near_pointing(dither['ra'][d]*np.pi/180., dither['dec'][d]*np.pi/180., dither['pa'][d]*np.pi/180., ra, dec)
                if len(self.use_ind)==0: # If no galaxies in focal plane, skip dither
                    continue
                # else:
                #     print 'number of potential objects',len(self.use_ind)
                #     print 'ra',dither['ra'][d],ra[self.use_ind]/np.pi*180.
                #     print 'dec',dither['dec'][d],dec[self.use_ind]/np.pi*180.

                # This instantiates a pointing object to be iterated over in some way
                # Return pointing object with wcs, psf, etc information.
                self.pointing = pointing(self.params,
                                        ra=dither['ra'][d], 
                                        dec=dither['dec'][d], 
                                        PA=dither['pa'][d], 
                                        date=date[d],
                                        SCA=None,
                                        PA_is_FPA=True, 
                                        logger=self.logger)

                skip = self.galaxy()
                if skip:
                    continue
                #self..star()

                u,c = np.unique(self.SCA,return_counts=True)
                print 'number of objects in SCAs',u,c

                for i,ind in enumerate(self.use_ind):
                    if i%100==0:
                        print 'drawing galaxy ',i
                    out = self.draw_galaxy(i,ind)
                    gal_exps[ind].append(out[0])
                    wcs_exps[ind].append(out[1])
                    if self.params['draw_true_psf']:
                        psf_exps[ind].append(out[2]) 
                    self.dither_list[ind].append(d)
                    self.sca_list[ind].append(self.SCA[i])
                cnt+=1

            for i in self.gind_list:
                if gal_exps[i] != []:
                    obj = des.MultiExposureObject(images=gal_exps[i], psf=psf_exps[i], wcs=wcs_exps[i], id=i)
                    objs.append(obj)
                    gal_exps[i]=[]
                    psf_exps[i]=[]
                    wcs_exps[i]=[]

            self.dump_meds(filter,objs)
            self.dump_truth(filter,ra,dec)

        return 

    def dump_meds(self,filter,objs):
        """
        Accepts a list of meds MultiExposureObject's and writes to meds file.
        """

        filename = self.params['output_meds']+'_'+filter+'.fits.gz'
        des.WriteMEDS(objs, filename, clobber=True)

        return

    def dump_truth(self,filter,ra,dec):
        """
        Accepts a list of meds MultiExposureObject's and writes to meds file.
        """
        import pickle
        def save_obj(obj, name ):
            with open(name, 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        filename = self.params['output_meds']+'_'+filter+'_truth.fits.gz'
        out = np.empty(len(self.gind_list), dtype=[('gal_index',int)]+[('g1',float)]+[('g2',float)]+[('rot_angle',float)]+[('phot_index',int)])
        out['gal_index'] = -1
        for i,ind in enumerate(self.gind_list):
            out['gal_index'][i]    = ind
            if (ind in self.e_list.keys())&(ind in self.rot_list.keys())&(ind in self.orig_pind_list.keys()):
                out['g1'][i]           = self.params['shear_list'][self.e_list[ind]][0]
                out['g2'][i]           = self.params['shear_list'][self.e_list[ind]][1]
                out['rot_angle'][i]    = self.rot_list[ind]
                out['phot_index'][i]   = self.pind_list[self.orig_pind_list[ind]]
            else:
                out['g1'][i]           = -999
                out['g2'][i]           = -999
                out['rot_angle'][i]    = -999
                out['phot_index'][i]   = -999
        fio.write(filename,out,clobber=True)

        filename = self.params['output_meds']+'_'+filter+'_truth_dither.pickle'
        save_obj(self.dither_list,filename)

        filename = self.params['output_meds']+'_'+filter+'_truth_sca.pickle'
        save_obj(self.sca_list,filename)

        return

if __name__ == "__main__":
    """
    """

    # This instantiates the simulation based on settings in input param file (argv[1])
    sim = wfirst_sim(sys.argv[1])

    # Initiate unique galaxy image list and noise models
    ra,dec = sim.init_galaxy()
    sim.init_noise_model()

    # Dither function that loops over pointings, SCAs, objects for each filter loop.
    # Returns a meds MultiExposureObject of galaxy stamps, psf stamps, and wcs.
    sim.dither_sim(ra,dec)

    # for i in range(len(m['ncutout'])):
    #     plt.imshow(m.get_mosaic(i))
    #     plt.savefig('stamp_'+str(i)+'.png')
    #     plt.close()


    # ai for few time 10^-4
