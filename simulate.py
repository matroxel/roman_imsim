"""
An implementation of galaxy and star image simulations for WFIRST. 
Built from the WFIRST GalSim module. An example 
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
import pickle
from astropy.time import Time
import mpi4py.MPI

path, filename = os.path.split(__file__)
sedpath = os.path.join(galsim.meta_data.share_dir, 'SEDs', 'CWW_Sbc_ext.sed')
sedpath_Star = os.path.join(galsim.meta_data.share_dir, 'SEDs', 'vega.txt')
g_band = os.path.join(galsim.meta_data.share_dir, 'bandpasses', 'LSST_g.dat')
if sys.version_info[0] == 3:
    string_types = str,
else:
    string_types = basestring,

t0=time.time()

MAX_RAD_FROM_BORESIGHT = 0.009

big_fft_params = galsim.GSParams(maximum_fft_size=10240)

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
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

def convert_dither_to_fits(ditherfile='observing_sequence_hlsonly'):

    dither = np.genfromtxt(ditherfile+'.dat',dtype=None,names = ['date','f1','f2','ra','dec','pa','program','filter','f8','f9','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21'])
    dither = dither[['date','ra','dec','pa','filter']][dither['program']==5]
    fio.write(ditherfile+'.fits',dither,clobber=True)

    return

def convert_gaia_to_fits(gaiacsv='../2017-09-14-19-58-07-4430',ralims=[0,360],declims=[-90,90]):

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

class wfirst_sim(object):
    """
    WFIRST image simulation.

    Input:
    param_file : File path for input yaml config file or yaml dict. Example located at: ./example.yaml.
    """

    def __init__(self, param_file, use_mpi = None):

        # Load parameter file
        if isinstance(param_file, string_types):
            self.params     = yaml.load(open(param_file))
            self.param_file = param_file
            # Do some parsing
            for key in self.params.keys():
                if self.params[key]=='None':
                    self.params[key]=None
                if self.params[key]=='True':
                    self.params[key]=True
                if self.params[key]=='False':
                    self.params[key]=False
        else:
            self.params     = param_file

        if use_mpi is not None:
            self.params['use_mpi'] = use_mpi

        # GalSim logger
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
        if isinstance(self.params['gal_dist'],string_types):
            self.n_gal = fio.FITS(self.params['gal_dist'])[-1].read_header()['NAXIS2']
        else:
            raise ParamError('Currently need gal_dist file.')

        # Read in the WFIRST filters, setting an AB zeropoint appropriate for this telescope given its
        # diameter and (since we didn't use any keyword arguments to modify this) using the typical
        # exposure time for WFIRST images.  By default, this routine truncates the parts of the
        # bandpasses that are near 0 at the edges, and thins them by the default amount.
        self.bpass      = wfirst.getBandpasses(AB_zeropoint=True)[self.params['filter']]
        # Setup galaxy SED
        # Need to generalize to vary sed based on input catalog
        self.galaxy_sed = galsim.SED(sedpath, wave_type='Ang', flux_type='flambda')
        # Setup star SED
        self.star_sed   = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')

        return

    def reset_rng(self):
        """
        Reset the (pseudo-)random number generators.
        """

        self.rng     = galsim.BaseDeviate(self.params['random_seed'])
        self.gal_rng = galsim.UniformDeviate(self.params['random_seed'])

        return


    def fwhm_to_hlr(self,fwhm):

        radius = fwhm*0.06/2. # 1 pix = 0.06 arcsec, factor 2 to convert to hlr

        return radius


    def init_galaxy(self):
        """
        Does the work to return a random, unique object property list. 
        """

        if isinstance(self.params['gal_dist'],string_types):
            # Provided an ra,dec catalog of object positions.
            radec_file     = fio.FITS(self.params['gal_dist'])[-1]
        else:
            raise ParamError('Bad gal_dist filename.')

        if self.params['gal_type'] == 0:
            # Analytic profile - sersic disk

            # Check if output truth file path exists or if explicitly remaking galaxy properties 
            filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_truth_gal.fits'
            if self.params['rerun_models']:

                # Read in file with photometry/size/redshift distribution similar to WFIRST galaxies
                phot       = fio.FITS(self.params['gal_sample'])[-1].read(columns=['fwhm','redshift',filter_flux_dict[self.params['filter']]])
                pind_list_ = np.ones(len(phot)).astype(bool) # storage list for original index of photometry catalog
                pind_list_ = pind_list_&(phot[filter_flux_dict[self.params['filter']]]<99)&(phot[filter_flux_dict[self.params['filter']]]>0) # remove bad mags
                pind_list_ = pind_list_&(phot['redshift']>0)&(phot['redshift']<5) # remove bad redshifts
                pind_list_ = pind_list_&(phot['fwhm']*0.06/wfirst.pixel_scale<16) # remove large objects to maintain 32x32 stamps
                pind_list_ = np.where(pind_list_)[0]

                # Create minimal storage array for galaxy properties to pass to parallel tasks
                store = np.ones(self.n_gal, dtype=[('rot','i2')]+[('e','i2')]+[('size','f4')]+[('z','f4')]+[('mag','f4')]+[('ra',float)]+[('dec',float)])
                store['ra'] = radec_file.read(columns='ra')*np.pi/180.
                store['dec'] = radec_file.read(columns='dec')*np.pi/180.
                pind = np.zeros(len(store)).astype(int)
                g1   = np.zeros(len(store))
                g2   = np.zeros(len(store))
                for i in range(self.n_gal):
                    pind[i] = pind_list_[int(self.gal_rng()*len(pind_list_))]
                    store['rot'][i]  = int(self.gal_rng()*360.)
                    store['e'][i]    = int(self.gal_rng()*len(self.params['shear_list']))
                    g1[i] = self.params['shear_list'][store['e'][i]][0]
                    g2[i] = self.params['shear_list'][store['e'][i]][1]
                    store['size'][i] = self.fwhm_to_hlr(phot['fwhm'][pind[i]])
                    store['z'][i]    = phot['redshift'][pind[i]]
                    store['mag'][i]  = phot[filter_flux_dict[self.params['filter']]][pind[i]]

                # Save truth file with galaxy properties
                sim.dump_truth_gal(store,pind,g1,g2)

            else:

                # Load truth file with galaxy properties
                store = sim.load_truth_gal(store)

        else:
            raise ParamError('COSMOS profiles not currently implemented.')            
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

        return store


    def init_star(self):
        """
        Compiles a list of stars properties to draw. 
        """

        if isinstance(self.params['star_sample'],string_types):
            # Provided a catalog of star positions and properties.
            fits = fio.FITS(self.params['star_sample'])[-1]
        else:
            raise ParamError('Bad star_sample filename.')

        stars_ = fits.read(columns=['ra','dec',self.params['filter']])

        # Create minimal storage array for galaxy properties to pass to parallel tasks
        stars         = np.ones(len(stars_), dtype=[('flux','f4')]+[('ra',float)]+[('dec',float)])
        stars['ra']   = stars_['ra']*np.pi/180.
        stars['dec']  = stars_['dec']*np.pi/180.
        stars['flux'] = stars_[self.params['filter']]

        return stars

    def init_noise_model(self):
        """
        Generate a poisson noise model.
        """

        self.noise = galsim.PoissonNoise(self.rng)

        return 

    def add_effects(self,im):
        """
        Add detector effects for WFIRST.

        Input:

        im      : Postage stamp or image.

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

        if self.params['use_background']:
            im, sky_image = self.add_background(im) # Add background to image and save background

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
            im,sky_image = self.finalize_background_subtract(im,sky_image)

        # im = galsim.Image(im, dtype=int)

        # get weight map
        sky_image.invertSelf()

        return im, sky_image

    def add_background(self,im):
        """
        Add backgrounds to image (sky, thermal).

        First we get the amount of zodaical light for a position corresponding to the position of 
        the object. The results are provided in units of e-/arcsec^2, using the default WFIRST
        exposure time since we did not explicitly specify one. Then we multiply this by a factor
        >1 to account for the amount of stray light that is expected. If we do not provide a date
        for the observation, then it will assume that it's the vernal equinox (sun at (0,0) in
        ecliptic coordinates) in 2025.
        """

        sky_level = wfirst.getSkyLevel(self.bpass, world_pos=self.radec, date=self.date)
        sky_level *= (1.0 + wfirst.stray_light_fraction)
        # Make a image of the sky that takes into account the spatially variable pixel scale. Note
        # that makeSkyImage() takes a bit of time. If you do not care about the variable pixel
        # scale, you could simply compute an approximate sky level in e-/pix by multiplying
        # sky_level by wfirst.pixel_scale**2, and add that to final_image.

        sky_stamp = galsim.Image(self.params['stamp_size'], self.params['stamp_size'], wcs=self.local_wcs)
        self.local_wcs.makeSkyImage(sky_stamp, sky_level)

        # This image is in units of e-/pix. Finally we add the expected thermal backgrounds in this
        # band. These are provided in e-/pix/s, so we have to multiply by the exposure time.
        sky_stamp += wfirst.thermal_backgrounds[self.params['filter']]*wfirst.exptime

        # Adding sky level to the image.
        im += sky_stamp
        
        return im,sky_stamp

    def add_poisson_noise(self,im):
        """
        Add pre-initiated poisson noise to image.
        """

        # Check if noise initiated
        if not hasattr(self,'noise'):
            self.init_noise_model()

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

        im.addReciprocityFailure(exp_time=wfirst.exptime, alpha=wfirst.reciprocity_alpha,
                              base_flux=1.0)
        # wfirst.addReciprocityFailure(im)

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
        im.applyNonlinearity(NLfunc=wfirst.NLfunc)
        # wfirst.applyNonlinearity(im)

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
        im.applyIPC(wfirst.ipc_kernel, edge_treatment='extend', fill_value=None)
        # wfirst.applyIPC(im)

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

        return im,sky

    def galaxy(self, ind, radec, bound = [wfirst.n_pix,wfirst.n_pix], return_xy = False):
        """
        Draw a postage stamp for one of the galaxy objects using the local wcs for its position in the SCA plane. Apply add_effects. 
        """

        # Check if galaxy falls on SCA and continue if not
        xy = self.WCS.toImage(radec)
        if (xy.x<bound[0])|(xy.y<bound[0])|(xy.x>bound[1])|(xy.y>bound[1]):
            if return_xy:
                return None, xy
            else:
                return None

        # Generate galaxy model
        gal          = galsim.Sersic(self.params['disk_n'], half_light_radius=1.*self.store['size'][ind]) # sersic disk galaxy
        gal          = gal.rotate(self.store['rot'][ind]*galsim.degrees) # random rotation
        gal          = gal.shear(g1=self.params['shear_list'][self.store['e'][ind]][0],g2=self.params['shear_list'][self.store['e'][ind]][1]) # apply a shear
        galaxy_sed   = self.galaxy_sed.atRedshift(self.store['z'][ind]) # redshift SED
        galaxy_sed   = galaxy_sed.withMagnitude(self.store['mag'][ind],self.bpass) # apply correct flux from magnitude
        gal          = gal * galaxy_sed * galsim.wfirst.collecting_area * galsim.wfirst.exptime

        # ignoring chromatic stuff for now
        flux = gal.calculateFlux(self.bpass) # store flux
        gal  = gal.evaluateAtWavelength(self.bpass.effective_wavelength) # make achromatic
        gal  = gal.withFlux(flux) # reapply correct flux
        
        gal  = galsim.Convolve(gal, self.PSF, gsparams=big_fft_params) # Convolve with PSF and append to final image list

        # replaced by above lines
        # # Draw galaxy igal into stamp.
        # self.gal_list[igal].drawImage(self.pointing.bpass[self.params['filter']], image=gal_stamp)
        # # Add detector effects to stamp.

        if return_xy:
            return gal, xy
        else:
            return gal

    def star(self, sed, flux = 1., radec = None, bound = [wfirst.n_pix,wfirst.n_pix], return_xy = False):

        # Check if star falls on SCA and continue if not
        if radec is not None:
            xy = self.WCS.toImage(radec)
            if (xy.x<bound[0])|(xy.y<bound[0])|(xy.x>bound[1])|(xy.y>bound[1]):
                if return_xy:
                    return None, xy
                else:
                    return None

        # Generate star model
        psf = galsim.DeltaFunction() * sed
        # Draw the star
        # new effective version for speed
        psf = psf.evaluateAtWavelength(self.bpass.effective_wavelength)
        psf = psf.withFlux(flux)
        psf = galsim.Convolve(psf, self.PSF, gsparams=big_fft_params)

        # old chromatic version
        # self.psf_list[igal].drawImage(self.pointing.bpass[self.params['filter']],image=psf_stamp, wcs=local_wcs)

        #galaxy_sed = galsim.SED(
        #    os.path.join(sedpath, 'CWW_Sbc_ext.sed'), wave_type='Ang', flux_type='flambda').withFlux(
        #    1.,self.pointing.bpass[self.params['filter']])
        #self.pointing.PSF[self.SCA[igal]] *= galaxy_sed
        #pointing_psf = galsim.Convolve(galaxy_sed, self.pointing.PSF[self.SCA[igal]])
        #self.pointing.PSF[self.SCA[igal]].drawImage(self.pointing.bpass[self.params['filter']],image=psf_stamp, wcs=local_wcs)
        #pointing_psf = galaxy_sed * self.pointing.PSF[self.SCA[igal]]
        #pointing_psf.drawImage(self.pointing.bpass[self.params['filter']],image=psf_stamp, wcs=local_wcs)
        #self.pointing.PSF[self.SCA[igal]].drawImage(self.pointing.bpass[self.params['filter']],image=psf_stamp, wcs=local_wcs)

        if return_xy:
            return star, xy
        else:
            return star

    def draw_galaxy(self, ind):


        self.radec = galsim.CelestialCoord(self.store['ra'][ind]*galsim.radians,self.store['dec'][ind]*galsim.radians)
        gal = self.galaxy(ind,self.radec)
        if gal is None:
            return None

        # Get local wcs solution at galaxy position in SCA.
        self.local_wcs = self.WCS.local(xy)
        # Create stamp at this position.
        gal_stamp = galsim.Image(self.params['stamp_size'], self.params['stamp_size'], wcs=self.local_wcs)
        gal.drawImage(image=gal_stamp) # draw galaxy stamp
        # Apply background, noise, and WFIRST detector effects
        gal_stamp, weight_stamp = self.add_effects(gal_stamp) # Get final galaxy stamp and weight map

        if self.params['draw_true_psf']:
            psf       = self.draw_star(galaxy_sed)
            psf_stamp = galsim.Image(self.params['stamp_size'], self.params['stamp_size'], wcs=self.local_wcs)
            psf.drawImage(image=psf_stamp,wcs=self.local_wcs)

            return gal_stamp, weight_stamp, psf_stamp
        else:
            return gal_stamp, weight_stamp

    def near_pointing(self, obsRA, obsDec, obsPA, ptRA, ptDec):
        """
        Returns mask of objects too far from pointing to consider more expensive checking.
        """

        x = np.cos(ptDec) * np.cos(ptRA)
        y = np.cos(ptDec) * np.sin(ptRA)
        z = np.sin(ptDec)

        d2 = (x - np.cos(obsDec)*np.cos(obsRA))**2 + (y - np.cos(obsDec)*np.sin(obsRA))**2 + (z - np.sin(obsDec))**2
        dist = 2.*np.arcsin(np.sqrt(d2)/2.)

        return np.where(dist<=MAX_RAD_FROM_BORESIGHT)[0].astype('i4')

    def dither_sim(self,sca):
        """
        Loop over each dither - prepares task list for each process and loops over dither_loop().
        """

        # Loops over dithering file
        tasks = []
        for i in range(self.params['nproc']):
            tasks.append({
                'proc'       : i,
                'sca'        : sca,
                'params'     : self.params,
                'store'      : self.store,
                'stars'      : self.stars})

        tasks = [ [(job, k)] for k, job in enumerate(tasks) ]

        results = process.MultiProcess(self.params['nproc'], {}, dither_loop, tasks, 'dithering', logger=self.logger, done_func=None, except_func=except_func, except_abort=True)

        return 

    def draw_pure_stamps(self,sca,proc,dither,d_,d,cnt,dumps):

        # Find objects near pointing.
        gal_use_ind = self.near_pointing(dither['ra'][d], dither['dec'][d], dither['pa'][d], self.store['ra'], self.store['dec'])
        if len(gal_use_ind)==0: # If no galaxies in focal plane, skip dither
            return cnt,dumps
        if self.params['timing']:
            print 'after gal_use_ind',time.time()-t0

        print '------------- dither ',d_[d]
        for i,ind in enumerate(gal_use_ind):
            out = self.draw_galaxy(ind)
            if out is None:
                continue
            if self.params['timing']:
                if i%100==0:
                    print 'drawing galaxy ',i,time.time()-t0

            cnt+= 1
            if ind in gal_exps.keys():
                self.gal_exps[ind].append(out[0])
                self.wcs_exps[ind].append(self.local_wcs)
                self.wgt_exps[ind].append(out[1])
                if self.params['draw_true_psf']:
                    self.psf_exps[ind].append(out[2]) 
                self.dither_list[ind].append(d_[d])
                self.sca_list[ind].append(sca)
            else:
                self.gal_exps[ind]     = [out[0]]
                self.wcs_exps[ind]     = [self.local_wcs]
                self.wgt_exps[ind]     = [out[1]]
                if self.params['draw_true_psf']:
                    self.psf_exps[ind] = [out[2]] 
                self.dither_list[ind]  = [d_[d]]
                self.sca_list[ind]     = [sca]
        print '------------- dither done ',d_[d]

        if cnt>self.params['pickle_size']:
            dumps,cnt = self.dump_stamps_pickle(sca,proc,dumps,cnt)

        return cnt,dumps

    def dump_stamps_pickle(self,sca,proc,dumps,cnt):

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_stamps_'+str(sca)+'_'+str(proc)+'_'+str(dumps)+'.pickle'
        save_obj([self.gal_exps,self.wcs_exps,self.wgt_exps,self.psf_exps,self.dither_list,self.sca_list], filename )

        cnt   = 0
        dumps+= 1
        self.gal_exps    = {}
        self.wcs_exps    = {}
        self.wgt_exps    = {}
        self.psf_exps    = {}
        self.dither_list = {}
        self.sca_list    = {}

        return dumps,cnt

    def draw_sca(self,sca,proc,dither,d_,d):

        # Find objects near pointing.
        gal_use_ind = self.near_pointing(dither['ra'][d], dither['dec'][d], dither['pa'][d], self.store['ra'], self.store['dec'])

        if self.params['draw_stars']:
            # Find stars near pointing.
            star_use_ind = self.near_pointing(dither['ra'][d], dither['dec'][d], dither['pa'][d], self.stars['ra'], self.stars['dec'])

        if len(gal_use_ind)+len(star_use_ind)==0: # If nothing in focal plane, skip dither
            return None
        if self.params['timing']:
            print 'after _use_ind',time.time()-t0        

        # Setup image for SCA
        im = galsim.ImageF(wfirst.n_pix+2*self.params['stamp_size'],wfirst.n_pix+2*self.params['stamp_size'], wcs=self.WCS)

        print '------------- dither ',d_[d]
        for i,ind in enumerate(gal_use_ind):
            radec  = galsim.CelestialCoord(self.store['ra'][ind]*galsim.radians,self.store['dec'][ind]*galsim.radians)
            gal,xy = self.galaxy(ind,
                                radec,
                                bound=[-self.params['stamp_size'],wfirst.n_pix+self.params['stamp_size']], 
                                return_xy = True)
            if self.params['timing']:
                if i%1==0:
                    print 'drawing galaxy ',i,time.time()-t0
            if gal is None:
                continue
            b = galsim.BoundsI(xmin=xy.x-self.params['stamp_size']/2,
                                ymin=xy.y-self.params['stamp_size']/2,
                                xmax=xy.x+self.params['stamp_size']/2,
                                ymax=xy.y+self.params['stamp_size']/2)
            b = b & im.bounds
            obj.drawImage(image=im[b], add_to_image=True, offset=xy-im[b].trueCenter())

            if ind in self.dither_list[0].keys():
                self.dither_list[0][ind].append(d_[d])
                self.sca_list[0][ind].append(sca)
                self.xy_list[0][ind].append(xy)
            else:
                self.dither_list[0][ind]  = [d_[d]]
                self.sca_list[0][ind]     = [sca]
                self.xy_list[0][ind]      = [xy]

        if self.params['draw_stars']:
            for i,ind in enumerate(star_use_ind):
                radec    = galsim.CelestialCoord(self.stars['ra'][ind]*galsim.radians,self.stars['dec'][ind]*galsim.radians)
                star_sed = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')
                star,xy  = self.star(star_sed, 
                                    flux = self.stars['flux'][ind], 
                                    radec = radec, 
                                    bound = [-self.params['stamp_size'],wfirst.n_pix+self.params['stamp_size']],
                                    return_xy = True)
                if star is None:
                    continue
                star.drawImage(image=im, add_to_image=True, offset=xy-im.trueCenter())
                if self.params['timing']:
                    if i%100==0:
                        print 'drawing star ',i,time.time()-t0

                if ind in self.dither_list[1].keys():
                    self.dither_list[1][ind].append(d_[d])
                    self.sca_list[1][ind].append(sca)
                    self.xy_list[1][ind].append(xy)
                else:
                    self.dither_list[1][ind]  = [d_[d]]
                    self.sca_list[1][ind]     = [sca]
                    self.xy_list[1][ind]      = [xy]

        return im[galsim.BoundsI(xmin=im.bounds.xmin+self.params['stamp_size'],
                                ymin=im.bounds.ymin+self.params['stamp_size'],
                                xmax=im.bounds.xmax-self.params['stamp_size'],
                                ymax=im.bounds.ymax-self.params['stamp_size'])]

    def dump_sca_pickle(self,sca,proc):

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_sca_'+str(sca)+'_'+str(proc)+'.pickle'
        save_obj([self.dither_list,self.sca_list,self.xy_list], filename )

        return

    def dump_sca_fits_pickle(self,im,sca,d):

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_image_'+str(sca)+'_'+str(d)+'.pickle'
        save_obj(im, filename)

        return

    def dump_sca_fits(self,im,d):

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_image_'+str(sca)+'_'+str(d)+'.pickle'
        galsim.fits.writeMulti(im, file_name=filename)

        return

    def accumulate_sca(self):
        """
        Accumulate the written pickle files that contain the images for each SCA, with SCA and dither ids.
        Write images to fits file.
        """

        d_ = self.setup_dither(only_index = True)
        for d in d_:
            im_list = []
            for sca in range(18):
                try:
                    filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_image_'+str(sca)+'_'+str(d)+'.pickle'
                    im = load_obj(filename)
                except:
                    im = galsim.ImageF(wfirst.n_pix,wfirst.n_pix)
                im_list.append(im)

            self.dump_sca_fits(im_list,d)

        return

    def accumulate_stamps(self):
        """
        Accumulate the written pickle files that contain the postage stamps for all objects, with SCA and dither ids.
        Write stamps to MEDS file, and SCA and dither ids to truth files. 
        """

        # Loop over chunks of galaxies because full output too large to hold in memory to create MEDS file
        for chunk in range(self.n_gal//self.params['meds_size']):
            low = chunk*self.params['meds_size']
            high = (chunk+1)*self.params['meds_size']
            if high > self.n_gal:
                high = self.n_gal
            # Create storage dictionaries.
            gal_exps    = {}
            wcs_exps    = {}
            wgt_exps    = {}
            psf_exps    = {}
            sca_list    = {}
            dither_list = {}
            for ind in range(low,high):
                gal_exps[ind]    = []
                wcs_exps[ind]    = []
                wgt_exps[ind]    = []
                psf_exps[ind]    = []
                sca_list[ind]    = []
                dither_list[ind] = []

            # Loop over each sca and dither pickles to accumulate into meds and truth files
            for sca in range(18):
                for proc in range(20):
                    print sca, proc
                    for dumps in range(10):
                        try:
                            filename = sim.out_path+'/'+sim.params['output_meds']+'_'+sim.params['filter']+'_stamps_'+str(sca)+'_'+str(proc)+'_'+str(dumps)+'.pickle'
                            gal_exps_,wcs_exps_,wgt_exps_,psf_exps_,dither_list_,sca_list_ = load_obj(filename)

                            keys = np.array(gal_exps_.keys())
                            keys = keys[(keys>=low)&(keys<high)]
                            for ind in keys:
                                gal_exps[ind]    = gal_exps[ind] + gal_exps_[ind]
                                psf_exps[ind]    = psf_exps[ind] + psf_exps_[ind]
                                wcs_exps[ind]    = wcs_exps[ind] + wcs_exps_[ind]
                                wgt_exps[ind]    = wgt_exps[ind] + wgt_exps_[ind]
                                sca_list[ind]    = sca_list[ind] + sca_list_[ind]
                                dither_list[ind] = dither_list[ind] + dither_list_[ind]

                        except:
                            if dumps == 0:
                                # pass
                                raise ParamError('Problem reading pickle file.')

            # Create dummy coadd stamp in first position
            for ind in keys:
                gal_exps[ind].insert(0,gal_exps[ind][0])
                psf_exp[ind].insert(0,psf_exp[ind][0])
                wcs_exp[ind].insert(0,wcs_exp[ind][0])
                wgt_exp[ind].insert(0,wgt_exp[ind][0])

            # write truth file for sca and dither indices
            sim.dump_truth_ind(dither_list,sca_list,chunk)
            sca_list    = None
            dither_list = None

            # Create MEDS objects list
            objs   = []
            for ind in gal_exps.keys():
                if len(gal_exps[ind])>0:
                    obj = des.MultiExposureObject(gal_exps[ind], psf=psf_exps[ind], wcs=wcs_exps[ind], weight=wgt_exps[ind], id=ind)
                    objs.append(obj)
                    gal_exps[ind]=[]
                    psf_exps[ind]=[]
                    wcs_exps[ind]=[]
                    wgt_exps[ind]=[]

            # Save MEDS file
            sim.dump_meds(objs,chunk)
            objs = None

        return

    def dump_meds(self,objs,chunk):
        """
        Accepts a list of meds MultiExposureObject's and writes to meds file.
        """

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_'+str(chunk)+'.fits'
        des.WriteMEDS(objs, filename, clobber=True)

        return

    def dump_truth_gal(self,store,pind,g1,g2):
        """
        Accepts a list of meds MultiExposureObject's and writes to meds file.
        """

        if len(store)!=self.n_gal:
            raise ParamError('Lengths of truth array and expected number of galaxies do not match.')

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_truth_gal.fits'
        out = np.ones(self.n_gal, dtype=[('gal_index','i4')]+[('ra',float)]+[('dec',float)]+[('g1','f4')]+[('g2','f4')]+[('e_index','i2')]+[('rot_angle','i2')]+[('gal_size','f4')]+[('redshift','f4')]+[('magnitude',float)]+[('phot_index','i4')])

        out['gal_index']    = np.arange(len(store))
        out['ra']           = store['ra']
        out['dec']          = store['dec']
        out['rot_angle']    = store['rot']
        out['gal_size']     = store['size']
        out['redshift']     = store['z']
        out['magnitude']    = store['mag']
        out['e_index']      = store['e']
        out['g1']           = g1
        out['g2']           = g2
        out['phot_index']   = pind

        fio.write(filename,out,clobber=True)

        return

    def load_truth_gal(self):
        """
        Accepts a list of meds MultiExposureObject's and writes to meds file.
        """

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_truth_gal.fits'
        store = np.ones(self.n_gal, dtype=[('rot','i2')]+[('e','i2')]+[('size','f4')]+[('z','f4')]+[('mag','f4')]+[('ra',float)]+[('dec',float)])
        out = fio.FITS(filename)[-1].read()

        if len(out)!=self.n_gal:
            raise ParamError('Lengths of truth array and expected number of galaxies do not match.')

        store['rot']  = out['rot_angle']
        store['e']    = out['e_index']
        store['size'] = out['gal_size']
        store['z']    = out['redshift']
        store['mag']  = out['magnitude']
        store['ra']   = out['ra']
        store['dec']  = out['dec']

        return store

    def dump_truth_ind(self,dither_list,sca_list,chunk):
        """
        Accepts a list of meds MultiExposureObject's and writes to meds file.
        """

        depth = 0
        for ind in dither_list.keys():
            if len(dither_list[ind])>depth:
                depth = len(dither_list[ind])

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_truth_ind_'+str(chunk)+'.fits'
        out = np.ones(self.n_gal, dtype=[('gal_index',int)]+[('dither_index',int,(depth))]+[('sca',int,(depth))])
        for name in out.dtype.names:
            out[name] *= -999
        for ind in dither_list.keys():
            stop = len(dither_list[ind])
            out['dither_index'][ind][:stop] = dither_list[ind]
            out['sca'][ind][:stop]          = sca_list[ind]

        fio.write(filename,out,clobber=True)

        return

    def setup_dither(self, proc, only_index = False):

        fits    = fio.FITS(self.params['dither_file'])[-1]
        date    = fits.read(columns='date')
        dfilter = fits.read(columns='filter')
        dither  = fits.read(columns=['ra','dec','pa'])

        chunk   = len(dither)//self.params['nproc']
        d_      = np.where((dither['ra']>24)&(dither['ra']<28.5)&(dither['dec']>-28.5)&(dither['dec']<-24)&(dfilter == filter_dither_dict[self.params['filter']]))[0]
        if only_index:
            return d_
        dfilter = None
        dither  = dither[d_[proc::self.params['nproc']]]
        date    = Time(date[d_[proc::self.params['nproc']]],format='mjd').datetime

        for name in dither.dtype.names:
            dither[name] *= np.pi/180.

        return dither,date,d_

def dither_loop(proc = None, sca = None, params = None, store = None, stars = None, **kwargs):
    """

    """

    sim = wfirst_sim(params)
    sim.store = store
    sim.stars = stars

    if sim.params['draw_sca']:
        sim.dither_list = [{},{}]
        sim.sca_list    = [{},{}]
        sim.xy_list     = [{},{}]
    else:
        sim.gal_exps    = {}
        sim.wcs_exps    = {}
        sim.wgt_exps    = {}
        sim.psf_exps    = {}
        sim.dither_list = {}
        sim.sca_list    = {}

    dither,date,d_ = sim.setup_dither(proc)

    cnt   = 0
    dumps = 0
    print '------------- sca ',sca
    # Here we carry out the initial steps that are necessary to get a fully chromatic PSF.  We use
    # the getPSF() routine in the WFIRST module, which knows all about the telescope parameters
    # (diameter, bandpasses, obscuration, etc.).
    # only doing this once to save time when its chromatic - need to check if duplicating other steps outweights this, though, once chromatic again
    sim.PSF = wfirst.getPSF(SCAs=sca+1, approximate_struts=sim.params['approximate_struts'], n_waves=sim.params['n_waves'], logger=sim.logger, wavelength=sim.bpass)[sca+1]
    # sim.logger.info('Done PSF precomputation in %.1f seconds!'%(time.time()-t0))

    for d in range(len(dither)):
        sim.date = date[d]

        # Get the WCS for an observation at this position. We are not supplying a date, so the routine
        # will assume it's the vernal equinox. The output of this routine is a dict of WCS objects, one 
        # for each SCA. We then take the WCS for the SCA that we are using.
        sim.WCS = wfirst.getWCS(world_pos=galsim.CelestialCoord(ra=dither['ra'][d]*galsim.radians, dec=dither['dec'][d]*galsim.radians), PA=dither['pa'][d]*galsim.radians, date=date[d], SCAs=sca+1, PA_is_FPA=True)[sca+1]

        if sim.params['draw_sca']:
            im = sim.draw_sca(sca,proc,dither,d_,d)
            if im is not None:
                sim.dump_sca_fits_pickle(im,sca,d_[d])
        else:
            cnt,dumps = sim.draw_pure_stamps(sca,proc,dither,d_,d,cnt,dumps)

    if sim.params['draw_sca']:
        sim.dump_sca_pickle(sca,proc)
    else:
        dumps,cnt = sim.dump_stamps_pickle(sca,proc,dumps,cnt)

    print 'dither loop done for proc ',proc

    return

def sca_loop(params,sca,store,stars):

    sim       = wfirst_sim(params)
    sim.store = store
    sim.stars = stars

    # Dither function that loops over pointings, SCAs, objects for each filter loop.
    # Returns a meds MultiExposureObject of galaxy stamps, psf stamps, and wcs.
    if sim.params['timing']:
        print 'before dither sim',time.time()-t0
    sim.dither_sim(sca)

    return

# global_class = None

def task(calcs):
    params,sca,store,stars=calcs
    sca_loop(params,sca,store,stars)
    # global_class.sca_loop(params,sca,store)

if __name__ == "__main__":
    """
    """

    # This instantiates the simulation based on settings in input param file (argv[1])
    sim = wfirst_sim(sys.argv[1])
    # sim.accumulate_stamps()
    # sys.exit()

    if sim.params['mpi']:
        from mpi_pool import MPIPool
        comm = mpi4py.MPI.COMM_WORLD
        pool = MPIPool(comm)
        if not pool.is_master():
            sim = None
            pool.wait()
            sys.exit(0)

    if sim.params['timing']:
        print 'before init galaxy',time.time()-t0
    # Initiate unique galaxy image list and noise models
    store = sim.init_galaxy()
    if sim.params['draw_stars']:
        stars = sim.init_star()
    else:
        stars = None
    if sim.params['timing']:
        print 'after init galaxy',time.time()-t0

    # define loop over SCAs
    calcs = []
    for i in range(18):
        calcs.append((sim.params,i,store,stars))

    if sim.params['mpi']:
        pool.map(task, calcs)
        pool.close()
    else:
        map(task, calcs)


