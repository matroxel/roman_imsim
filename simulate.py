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
Largely stolen from demo13...
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

filter_flux_dict = {
    'J129' : 'j_WFIRST',
    'F184' : 'F184W_WFIRST',
    'Y106' : 'y_WFIRST',
    'H158' : 'h_WFIRST'
}

class pointing(object):

    # Stores information about a pointing

    def __init__(self, params, ra=90., dec=-10., PA=None, SCA=None, PA_is_FPA=False, logger=None):

        self.ra         = ra  * galsim.degrees
        self.dec        = dec * galsim.degrees
        self.PA         = PA  * galsim.degrees
        self.PA_is_FPA  = PA_is_FPA
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
            self.logger =logger

        self.get_wcs()
        self.init_psf(approximate_struts=params['approximate_struts'], n_waves=params['n_waves'])

        return


    def get_wcs(self):

        # We choose a particular (RA, dec) location on the sky for our observation.
        pointing_pos = galsim.CelestialCoord(ra=self.ra, dec=self.dec)

        # Get the WCS for an observation at this position.  We are not supplying a date, so the routine
        # will assume it's the vernal equinox.  We are also not supplying a position angle for the
        # observatory, which means that it will just find the optimal one (the one that has the solar
        # panels pointed most directly towards the Sun given this targ_pos and date).  The output of
        # this routine is a dict of WCS objects, one for each SCA.  We then take the WCS for the SCA
        # that we are using.
        self.WCS = wfirst.getWCS(world_pos=pointing_pos, PA=self.PA, SCAs=self.SCA, PA_is_FPA=self.PA_is_FPA)

        # We need to find the center position for this SCA.  We'll tell it to give us a CelestialCoord
        # corresponding to (X, Y) = (wfirst.n_pix/2, wfirst.n_pix/2).
        self.SCA_centpos = {}
        for SCA in self.SCA:
            self.SCA_centpos[SCA] = self.WCS[SCA].toWorld(galsim.PositionD(wfirst.n_pix/2, wfirst.n_pix/2))

        return

    def init_psf(self, approximate_struts=False, n_waves=None):

        # Here we carry out the initial steps that are necessary to get a fully chromatic PSF.  We use
        # the getPSF() routine in the WFIRST module, which knows all about the telescope parameters
        # (diameter, bandpasses, obscuration, etc.).  Note that we arbitrarily choose a single SCA
        # (Sensor Chip Assembly) rather than all of them, for faster calculations, and use a simple
        # representation of the struts for faster calculations.  To do a more exact calculation of the
        # chromaticity and pupil plane configuration, remove the `self.params['approximate_struts']` and the `self.params['n_waves']`
        # keyword from the call to getPSF():
        self.logger.info('Doing expensive pre-computation of PSF.')
        t1 = time.time()
        self.logger.setLevel(logging.DEBUG)
        self.PSF = wfirst.getPSF(SCAs=self.SCA, approximate_struts=approximate_struts, n_waves=n_waves, logger=self.logger)
        self.logger.setLevel(logging.INFO)
        t2 = time.time()
        self.logger.info('Done PSF precomputation in %.1f seconds!'%(t2-t1))

        return

class wfirst_sim(object):

    def __init__(self,param_file):

        # load parameter file
        self.params = yaml.load(open(param_file))
        for key in self.params.keys():
            if self.params[key]=='None':
                self.params[key]=None
        logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
        # In non-script code, use getself.logger(__name__) at module scope instead.
        self.logger = logging.getLogger('wfirst_sim')
        # Initialize (pseudo-)random number generator.
        self.reset_rng()
        # Where to find and output data.
        path, filename = os.path.split(__file__)
        self.out_path = os.path.abspath(os.path.join(path, "output/"))
        # Make output directory if not already present.
        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)

        # set total number of objects
        if self.params['n_gal'] is not None:
            self.n_gal = self.params['n_gal']
        else:
            if isinstance(self.params['gal_dist'],string_types):
                radec_file = fio.FITS(self.params['gal_dist'])[-1].read()
                self.n_gal = len(radec_file)
            else:
                self.n_gal = int(self.params['gal_dist']*wfirst.n_pix*wfirst.n_pix)

        # Check that various params make sense
        if (self.params['size_dist'] is None)&(self.params['gal_type']==0):
            raise ParamError('Need size_dist filename if using Sersic galaxies.')

        # Read in the WFIRST filters, setting an AB zeropoint appropriate for this telescope given its
        # diameter and (since we didn't use any keyword arguments to modify this) using the typical
        # exposure time for WFIRST images.  By default, this routine truncates the parts of the
        # bandpasses that are near 0 at the edges, and thins them by the default amount.
        self.filters = wfirst.getBandpasses(AB_zeropoint=True)
        self.logger.debug('Read in WFIRST imaging filters.')

        return

    def reset_rng(self):

        self.rng = galsim.BaseDeviate(self.params['random_seed'])
        self.gal_rng = galsim.UniformDeviate(self.params['random_seed'])

        return

    def init_galaxy(self):
        """
        Return information to produce a 
        """

        self.logger.info('Pre-processing for galaxies started.')
        if self.params['gal_type'] == 0:
            size_dist = fio.FITS(self.params['size_dist'])[-1].read()['size']
            self.obj_list=[]
            for i in range(self.params['gal_n_use']):
                self.obj_list.append(galsim.Sersic(self.params['disk_n'], scale_radius=size_dist[int(self.gal_rng()*len(size_dist))]))
                # Does galsim have a pdf random sampler?
        else:
            if self.params['gal_type'] == 1:
                use_real = False
                gtype = 'parametric'
            else:
                use_real = True
                gtype = 'real'

            cat = galsim.COSMOSCatalog(self.params['cat_name'], dir=self.params['cat_dir'], use_real=use_real)
            self.logger.info('Read in %d galaxies from catalog'%cat.nobjects)

            rand_ind = []
            for i in range(self.params['gal_n_use']):
                rand_ind.append(int(self.gal_rng()*cat.nobjects))
            self.obj_list = cat.makeGalaxy(rand_ind, chromatic=True, gal_type=gtype)

        self.logger.debug('Pre-processing for galaxies completed.')

        return

    def galaxy(self):
        """
        Return a list of galaxy objects over a given flux and size distribution. If real galaxy used, 
        flux distribution is drawn from cosmos sample.
        Either read in ra,dec catalog and convert to xy for SCA or create random distribution of xy 
        for SCA and convert to radec to later enable dithering stuff (realistic appearance of galaxy 
        across SCA positions and angles.
        """

        self.reset_rng()
        self.logger.info('Compiling x,y,ra,dec positions of catalog.')

        if isinstance(self.params['gal_dist'],string_types):
            radec_file = fio.FITS(self.params['gal_dist'])[-1].read()
            self.radec = []
            self.xy    = []
            for i in xrange(self.n_gal):
                self.radec.append(galsim.CelestialCoord(radec_file['ra'],radec_file['dec']))
                self.xy.append(self.pointing.WCS[self.SCA].toImage(self.radec[i]))
                # Need to do check for if objects don't fall on SCA - don't know how galsim wcs handles that... 
                # n_gal currently wouldn't work, becaue some objects fall outside the SCA
        else:
            self.xy      = []
            self.radec   = []
            for i in xrange(self.n_gal):
                x_ = self.gal_rng()*wfirst.n_pix
                y_ = self.gal_rng()*wfirst.n_pix
                self.xy.append(galsim.PositionD(x_,y_))
                self.radec.append(self.pointing.WCS[self.SCA].toWorld(self.xy[i]))

        flux_dist = fio.FITS(self.params['flux_dist'])[-1].read(columns=filter_flux_dict[self.filter]) # magnitudes
        flux_dist = flux_dist[(flux_dist<99)&(flux_dist>0)] # remove bad mags
        flux_dist = 10**(flux_dist/2.5) # converting to fluxes
        self.gal_list  = []
        # gind_list is meant (more useful in the future) to preserve the link to the original galaxy catalog 
        # for writing exposure lists in meds files
        self.gind_list = []
        for i in range(self.n_gal):
            gind = int(self.gal_rng()*self.params['gal_n_use'])
            find = int(self.gal_rng()*len(flux_dist))
            obj  = self.obj_list[gind].copy()
            obj  = obj.rotate(int(self.gal_rng()*360.)*galsim.degrees)
            obj  = obj.withFlux(flux_dist[find])
            self.gal_list.append(galsim.Convolve(obj, self.pointing.PSF[self.SCA]))
            self.gind_list.append(i)

        return

    def star(self):
        """
        Return a list of star objects for psf measurement... Not done at all yet
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
        img_psf = galsim.ImageF(64,64)
        star.drawImage(bandpass=filter_, image=img_psf, scale=wfirst.pixel_scale)
        img_psf.write(out_filename)
        self.logger.debug('Created PSF with flat SED for {0}-band'.format(filter_name))

        return

    def init_noise_model(self):

        # Generate a Poisson noise model.
        self.noise = galsim.PoissonNoise(self.rng)
        self.logger.info('Poisson noise model created.')
        
        return 

    def add_effects(self,im,wpos,xy,date=None):

        # Preserve order:
        # 1) add_background
        # 2) add_poisson_noise
        # 3) recip_failure 
        # 4) quantize
        # 5) dark_current
        # 6) interpix_cap
        # 7) e_to_ADU
        # 8) quantize

        im, sky_image = self.add_background(im,wpos,xy,date=date)
        im = self.add_poisson_noise(im)
        im = self.recip_failure(im)
        # At this point in the image generation process, an integer number of photons gets
        # detected, hence we have to round the pixel values to integers:
        im.quantize()
        im = self.dark_current(im)
        im = self.interpix_cap(im)
        im = self.e_to_ADU(im)

        # Finally, the analog-to-digital converter reads in an integer value.
        im.quantize()
        # Note that the image type after this step is still a float.  If we want to actually
        # get integer values, we can do new_img = galsim.Image(im, dtype=int)

        # Since many people are used to viewing background-subtracted images, we provide a
        # version with the background subtracted (also rounding that to an int).
        sky_image = self.finalize_sky_im(sky_image)
        im -= sky_image

        return im

    def add_background(self,im,wpos,xy,date=None):
        """
        Add backgrounds to image (sky, thermal).
        """

        # First we get the amount of zodaical light for a position corresponding to the center of
        # this SCA.  The results are provided in units of e-/arcsec^2, using the default WFIRST
        # exposure time since we did not explicitly specify one.  Then we multiply this by a factor
        # >1 to account for the amount of stray light that is expected.  If we do not provide a date
        # for the observation, then it will assume that it's the vernal equinox (sun at (0,0) in
        # ecliptic coordinates) in 2025.
        sky_level = wfirst.getSkyLevel(self.filters[self.filter], world_pos=wpos)
        sky_level *= (1.0 + wfirst.stray_light_fraction)
        # Make a image of the sky that takes into account the spatially variable pixel scale.  Note
        # that makeSkyImage() takes a bit of time.  If you do not care about the variable pixel
        # scale, you could simply compute an approximate sky level in e-/pix by multiplying
        # sky_level by wfirst.pixel_scale**2, and add that to final_image.
        im_sky = im.copy()
        local_wcs = self.pointing.WCS[self.SCA].local(xy)
        local_wcs.makeSkyImage(im_sky, sky_level)
        # This image is in units of e-/pix.  Finally we add the expected thermal backgrounds in this
        # band.  These are provided in e-/pix/s, so we have to multiply by the exposure time.
        im_sky += wfirst.thermal_backgrounds[self.filter]*wfirst.exptime
        # Adding sky level to the image.
        im += im_sky

        return im,im_sky

    def add_poisson_noise(self,im):
        """
        Add pre-initiated poisson noise to image.
        """

        im.addNoise(self.noise)

        return im

    def recip_failure(self,im):

        # Reciprocity failure:
        # Reciprocity, in the context of photography, is the inverse relationship between the
        # incident flux (I) of a source object and the exposure time (t) required to produce a given
        # response(p) in the detector, i.e., p = I*t. However, in NIR detectors, this relation does
        # not hold always. The pixel response to a high flux is larger than its response to a low
        # flux. This flux-dependent non-linearity is known as 'reciprocity failure', and the
        # approximate amount of reciprocity failure for the WFIRST detectors is known, so we can
        # include this detector effect in our images.

        if self.params['diff_mode']:
            # Save the image before applying the transformation to see the difference
            im_save = im.copy()

        # If we had wanted to, we could have specified a different exposure time than the default
        # one for WFIRST, but otherwise the following routine does not take any arguments.
        wfirst.addReciprocityFailure(im)
        self.logger.debug('Included reciprocity failure in image')

        if self.params['diff_mode']:
            # Isolate the changes due to reciprocity failure.
            diff = im-im_save

        if self.params['diff_mode']:
            return im, diff
        else:
            return im

    def dark_current(self,im):

        # Adding dark current to the image:
        # Even when the detector is unexposed to any radiation, the electron-hole pairs that
        # are generated within the depletion region due to finite temperature are swept by the
        # high electric field at the junction of the photodiode. This small reverse bias
        # leakage current is referred to as 'dark current'. It is specified by the average
        # number of electrons reaching the detectors per unit time and has an associated
        # Poisson noise since it is a random event.
        dark_current = wfirst.dark_current*wfirst.exptime
        dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(self.rng, dark_current))
        im.addNoise(dark_noise)

        # NOTE: Sky level and dark current might appear like a constant background that can be
        # simply subtracted. However, these contribute to the shot noise and matter for the
        # non-linear effects that follow. Hence, these must be included at this stage of the
        # image generation process. We subtract these backgrounds in the end.

        # 3) Applying a quadratic non-linearity:
        # In order to convert the units from electrons to ADU, we must use the gain factor. The gain
        # has a weak dependency on the charge present in each pixel. This dependency is accounted
        # for by changing the pixel values (in electrons) and applying a constant nominal gain
        # later, which is unity in our demo.

        # Save the image before applying the transformation to see the difference:
        if self.params['diff_mode']:
            im_save = im.copy()

        # Apply the WFIRST nonlinearity routine, which knows all about the nonlinearity expected in
        # the WFIRST detectors.
        wfirst.applyNonlinearity(im)
        # Note that users who wish to apply some other nonlinearity function (perhaps for other NIR
        # detectors, or for CCDs) can use the more general nonlinearity routine, which uses the
        # following syntax:
        # final_image.applyNonlinearity(NLfunc=NLfunc)
        # with NLfunc being a callable function that specifies how the output image pixel values
        # should relate to the input ones.
        self.logger.debug('Applied nonlinearity to image')
        if self.params['diff_mode']:
            diff = im-im_save
            return im,diff
        else:
            return im

    def interpix_cap(self,im):

        # Including Interpixel capacitance:
        # The voltage read at a given pixel location is influenced by the charges present in the
        # neighboring pixel locations due to capacitive coupling of sense nodes. This interpixel
        # capacitance effect is modeled as a linear effect that is described as a convolution of a
        # 3x3 kernel with the image.  The WFIRST IPC routine knows about the kernel already, so the
        # user does not have to supply it.

        # Save this image to do the diff after applying IPC.
        im_save = im.copy()

        wfirst.applyIPC(im)
        self.logger.debug('Applied interpixel capacitance to image')

        if self.params['diff_mode']:
            # Isolate the changes due to the interpixel capacitance effect.
            diff = im-im_save
            return im,diff
        else:
            return im


    def add_read_noise(self,im):
        # Adding read noise:
        # Read noise is the noise due to the on-chip amplifier that converts the charge into an
        # analog voltage.  We already applied the Poisson noise due to the sky level, so read noise
        # should just be added as Gaussian noise:
        read_noise = galsim.GaussianNoise(self.rng, sigma=wfirst.read_noise)
        im.addNoise(read_noise)
        self.logger.debug('Added readnoise to image')

        return im

    def e_to_ADU(self,im):
        # We divide by the gain to convert from e- to ADU. Currently, the gain value in the WFIRST
        # module is just set to 1, since we don't know what the exact gain will be, although it is
        # expected to be approximately 1. Eventually, this may change when the camera is assembled,
        # and there may be a different value for each SCA. For now, there is just a single number,
        # which is equal to 1.

        return im/wfirst.gain


    def finalize_sky_im(self,im):
        """
        Finalize sky background for subtraction from final image. Add dark current, 
        convert to analog voltage, and quantize.
        """
        im.quantize()
        final_im = (im + round(wfirst.dark_current*wfirst.exptime))
        final_im = self.e_to_ADU(final_im)
        final_im.quantize()

        return final_im

    def draw_galaxy(self,igal):
        """
        Draw a postage stamp for one of the galaxy objects using the local wcs for its position in the SCA plane. Apply add_effects.
        """

        local_wcs = self.pointing.WCS[self.SCA].local(self.xy[igal])
        gal_stamp = galsim.Image(self.params['stamp_size'], self.params['stamp_size'], wcs=local_wcs)
        self.gal_list[igal].drawImage(self.filters[self.filter], image=gal_stamp)
        gal_stamp = self.add_effects(gal_stamp,self.radec[igal],self.xy[igal])

        if self.params['draw_true_psf']:
            # Also draw the PSF
            psf_stamp = galsim.ImageF(gal_stamp.bounds) # Use same bounds as galaxy stamp
            self.pointing.PSF[self.SCA].drawImage(self.pointing.bpass[self.filter],image=psf_stamp, wcs=local_wcs)

            # Add effects to psf_stamp - i think this is needed?
            psf_stamp = self.add_effects(psf_stamp,self.radec[igal],self.xy[igal])

            return gal_stamp, local_wcs, psf_stamp
        else:
            return gal_stamp, local_wcs

    def dither_sim(self):

        # currently just loops over SCAs and filters to collate exposure lists of an object 
        # that appears once at the same xy pos in every SCA (using the same pointing) to test 
        # the meds output.

        objs = []
        gal_exps = {}
        wcs_exps = {}
        psf_exps = {}
        for i in range(self.n_gal):
            gal_exps[i] = []
            wcs_exps[i] = []
            psf_exps[i] = []

        for SCA in self.pointing.SCA:
            self.SCA = SCA
            print 'SCA',SCA
            sim.galaxy()
            #sim.star()
            for i in range(self.n_gal):
                if i%1==0:
                    print 'drawing galaxy ',i
                if i in self.gind_list:
                    out = sim.draw_galaxy(i)
                    gal_exps[i].append(out[0])
                    wcs_exps[i].append(out[1])
                    if self.params['draw_true_psf']:
                        psf_exps[i].append(out[2])

        for i in range(self.n_gal):
            obj = des.MultiExposureObject(images=gal_exps[i], psf=psf_exps[i], wcs=wcs_exps[i], id=i)
            objs.append(obj)
            gal_exps[i]=None
            psf_exps[i]=None
            wcs_exps[i]=None

        return objs

    def dump_meds(self,filter,objs):

        filename = self.params['output_meds']+'_'+filter+'.fits.gz'
        des.WriteMEDS(objs, filename, clobber=True)

        return

if __name__ == "__main__":
    # this instantiates a pointing object that contains information for wcs, psf across SCAs 
    # requested in input param file (argv[1])
    sim = wfirst_sim(sys.argv[1])

    # interate over pointings in some way
    # return pointing object with wcs and psf information
    sim.pointing = pointing(sim.params, 
                            ra=sim.params['pointing_ra'], 
                            dec=sim.params['pointing_dec'], 
                            PA=sim.params['PA'], 
                            SCA=sim.params['SCAs'], 
                            PA_is_FPA=sim.params['PA_is_FPA'], 
                            logger=sim.logger)

    # init galaxy and noise models
    sim.init_galaxy()
    sim.init_noise_model()

    # loop this over galaxy objects, filters, SCAs, etc... Draws and returns a galaxy and 
    # corresponding psf postage stamp (psf=False for no psf stamp)
    for filter in filter_flux_dict.keys():
        sim.filter = filter
        print 'filter',filter
        objs = sim.dither_sim()
        sim.dump_meds(filter, objs)

    # there's still a lot of work to do dithering correctly and looping over filters, SCAs, 
    # pointings, etc. I think the easiest way to do this consistently is to produce an input 
    # catalog of ra,dec and pointings,PA that dither appropriately the extent of the ra,dec 
    # catalog using a dither module (not written). The reading of hte ra,dec catalog like 
    # this is supported, but need a way to track exposures of objects per band across pointings 
    # for later compiling exposure lists for meds output. (this works for simple test case now)

    # I also haven't tested that this is bug free (i.e., actually produces things that look like galaxies) 
    # - its essentially rearranging demo13 with some wrapper and object structures, but will require some 
    # testing. It does import and run at least.



