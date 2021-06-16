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
import fitsio as fio
import cPickle as pickle
from astropy.time import Time
from mpi4py import MPI
import cProfile, pstats
import glob
import shutil
from ngmix.jacobian import Jacobian
from ngmix.observation import Observation, ObsList, MultiBandObsList
from ngmix.galsimfit import GalsimRunner,GalsimSimple,GalsimTemplateFluxFitter
from ngmix.guessers import R50FluxGuesser
from ngmix.bootstrap import PSFRunner
from ngmix import priors, joint_prior
import mof
import meds
import psc

import matplotlib
matplotlib.use ('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pylab

path, filename = os.path.split(__file__)
sedpath_Star   = os.path.join(galsim.meta_data.share_dir, 'SEDs', 'vega.txt')
bpass  = wfirst.getBandpasses(AB_zeropoint=True)['H158']
sedpath_E           = '/users/PCON0003/cond0083/GalSim/share/SEDs/NGC_4926_spec.dat'
sedpath_Scd         = '/users/PCON0003/cond0083/GalSim/share/SEDs/NGC_4670_spec.dat'
sedpath_Im          = '/users/PCON0003/cond0083/GalSim/share/SEDs/Mrk_33_spec.dat'
galaxy_sed_b = galsim.SED(sedpath_E, wave_type='Ang', flux_type='flambda')
galaxy_sed_d = galsim.SED(sedpath_Scd, wave_type='Ang', flux_type='flambda')
galaxy_sed_n = galsim.SED(sedpath_Im,  wave_type='Ang', flux_type='flambda')

PSF = wfirst.getPSF(1,
                        'H158',
                        SCA_pos             = None, # - in branch 919
                        approximate_struts  = True, 
                        n_waves             = 10, 
                        logger              = logging.getLogger('wfirst_sim'), 
                        wavelength          = bpass.effective_wavelength,
                        )
WCS = wfirst.getWCS(world_pos  = galsim.CelestialCoord(ra=np.radians(27.1656)*galsim.radians,
                        dec=np.radians(-16.5013)*galsim.radians), 
                        PA          = np.radians(270)*galsim.radians, 
                        date        = Time(61021.800069,format='mjd').datetime,
                        SCAs        = 1,
                        PA_is_FPA   = True
                        )[1]
ind=0

rng   = galsim.BaseDeviate(314)


class modify_image():
    """
    Class to simulate non-idealities and noise of wfirst detector images.
    """

    def __init__(self,params):
        """
        Set up noise properties of image

        Input
        params  : parameter dict
        rng     : Random generator
        """

        self.params    = params

    def add_effects(self,im,pointing,radec,local_wcs,rng,phot=False):
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

        self.rng       = rng
        self.noise     = self.init_noise_model()

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

        sky_level = wfirst.getSkyLevel(bpass, world_pos=radec, date=Time(61021.800069,format='mjd').datetime)
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
            sky_level = wfirst.getSkyLevel(bpass, world_pos=radec, date=Time(61021.800069,format='mjd').datetime)
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
            sky_stamp += wfirst.thermal_backgrounds['H158']*wfirst.exptime
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

def make_sed_model(model, sed):
    """
    Modifies input SED to be at appropriate redshift and magnitude, then applies it to the object model.

    Input
    model : Galsim object model
    sed   : Template SED for object
    flux  : flux fraction in this sed
    """

    # Apply correct flux from magnitude for filter bandpass
    sed_ = sed.atRedshift(gal['z'])
    sed_ = sed_.withMagnitude(gal['H158'], bpass)

    # Return model with SED applied
    return model * sed_

def galaxy_model():
    """
    Generate the intrinsic galaxy model based on truth catalog parameters
    """

    # Generate galaxy model
    # Calculate flux fraction of disk portion 
    flux = (1.-gal['bflux']) * gal['dflux']
    if flux > 0:
        # If any flux, build Sersic disk galaxy (exponential) and apply appropriate SED
        gal_model = galsim.Sersic(1, half_light_radius=1.*gal['size'], flux=flux, trunc=10.*gal['size'])
        gal_model = make_sed_model(gal_model, galaxy_sed_d)
        # self.gal_model = self.gal_model.withScaledFlux(flux)

    # Calculate flux fraction of knots portion 
    flux = (1.-gal['bflux']) * (1.-gal['dflux'])
    if flux > 0:
        # If any flux, build star forming knots model and apply appropriate SED
        rng   = galsim.BaseDeviate(314)
        knots = galsim.RandomWalk(npoints=10, half_light_radius=1.*gal['size'], flux=flux, rng=rng) 
        knots = make_sed_model(knots, galaxy_sed_n)
        # knots = knots.withScaledFlux(flux)
        # Sum the disk and knots, then apply intrinsic ellipticity to the disk+knot component. Fixed intrinsic shape, but can be made variable later.
        gal_model = galsim.Add([gal_model, knots])
        gal_model = gal_model.shear(e1=0.25, e2=0.25)

    # Calculate flux fraction of bulge portion 
    flux = gal['bflux']
    if flux > 0:
        # If any flux, build Sersic bulge galaxy (de vacaleurs) and apply appropriate SED
        bulge = galsim.Sersic(4, half_light_radius=1.*gal['size'], flux=flux, trunc=10.*gal['size']) 
        # Apply intrinsic ellipticity to the bulge component. Fixed intrinsic shape, but can be made variable later.
        bulge = bulge.shear(e1=0.25, e2=0.25)
        # Apply the SED
        bulge = make_sed_model(bulge, galaxy_sed_b)
        # bulge = bulge.withScaledFlux(flux)

        gal_model = galsim.Add([gal_model, bulge])

    return gal_model



def galaxy():
    """
    Call galaxy_model() to get the intrinsic galaxy model, then apply properties relevant to its observation
    """

    # Build intrinsic galaxy model
    gal_model = galaxy_model()

    # Random rotation (pairs of objects are offset by pi/2 to cancel shape noise)
    gal_model = gal_model.rotate(gal['rot']*galsim.radians) 
    # Apply a shear
    gal_model = gal_model.shear(g1=gal['g1'],g2=gal['g1'])
    # Rescale flux appropriately for wfirst
    gal_model = gal_model * galsim.wfirst.collecting_area * galsim.wfirst.exptime

    # Ignoring chromatic stuff for now for speed, so save correct flux of object
    flux = gal_model.calculateFlux(bpass)
    mag = gal_model.calculateMagnitude(bpass)
    # print 'galaxy flux',flux
    # Evaluate the model at the effective wavelength of this filter bandpass (should change to effective SED*bandpass?)
    # This makes the object achromatic, which speeds up drawing and convolution
    gal_model  = gal_model.evaluateAtWavelength(bpass.effective_wavelength)
    # Reassign correct flux
    gal_model  = gal_model.withFlux(flux) # reapply correct flux
    sky_level = wfirst.getSkyLevel(bpass, world_pos=WCS.toWorld(galsim.PositionI(wfirst.n_pix/2,wfirst.n_pix/2)), date=Time(61021.800069,format='mjd').datetime)
    sky_level *= (1.0 + wfirst.stray_light_fraction)*wfirst.pixel_scale**2 # adds stray light and converts to photons/cm^2
    sky_level *= 256*256 # Converts to photons, but uses smallest stamp size to do so - not optimal

    if sky_level/flux < galsim.GSParams().folding_threshold:
        gsparams = galsim.GSParams( folding_threshold=sky_level/flux,
                                    maximum_fft_size=16384 )
    else:
        gsparams = galsim.GSParams( maximum_fft_size=16384 )

    # Convolve with PSF
    gal_model = galsim.Convolve(gal_model.withGSParams(gsparams), PSF, propagate_gsparams=False)

    return gal_model


def draw_galaxy(xyI):
    """
    Draw the galaxy model into the SCA (neighbors and blending) and/or the postage stamp (isolated).
    """

    # Build galaxy model that will be drawn into images
    gal_model = galaxy()

    # Create postage stamp bounds at position of object
	b = galsim.BoundsI( xmin=xyI.x-int(192)/2+1,
	                    ymin=xyI.y-int(192)/2+1,
	                    xmax=xyI.x+int(192)/2,
	                    ymax=xyI.y+int(192)/2)

    # Create postage stamp for galaxy
	gal_stamp = galsim.Image(b, wcs=WCS)

    gal_model.drawImage(image=gal_stamp,offset=offset,method='phot',rng=rng)
    # gal_stamp.write(str(self.ind)+'.fits')

    # gal_stamp, weight = modify_image.add_effects(gal_stamp[b&self.b],self.pointing,self.radec,self.pointing.WCS,self.rng,phot=True)

    # gal_stamp = galsim.Image(b, wcs=WCS)
    # gal_stamp[b&self.b] = self.gal_stamp[b&self.b] + gal_stamp[b&self.b]
    # weight_stamp = galsim.Image(b, wcs=self.pointing.WCS)
    # weight_stamp[b&self.b] = self.weight_stamp[b&self.b] + weight[b&self.b]

    gal_stamp, weight = modify_image_.add_effects(gal_stamp,None,galsim.CelestialCoord(ra=np.radians(27.1656)*galsim.radians,
                        dec=np.radians(-16.5013)*galsim.radians),WCS,rng,phot=True)

    # Generate star model (just a delta function) and apply SED
    st_model = galsim.DeltaFunction(flux=1.)
    gsparams = galsim.GSParams( maximum_fft_size=16384 )

    # Evaluate the model at the effective wavelength of this filter bandpass (should change to effective SED*bandpass?)
    # This makes the object achromatic, which speeds up drawing and convolution
    st_model = st_model.evaluateAtWavelength(bpass.effective_wavelength)

    st_model = galsim.Convolve(st_model, PSF)

     # Create postage stamp bounds at position of object
    b_psf = galsim.BoundsI( xmin=xyI.x-int(8)/2+1,
                        ymin=xyI.y-int(8)/2+1,
                        xmax=xyI.x+int(8)/2,
                        ymax=xyI.y+int(8)/2)
    # Create postage stamp bounds at position of object
    # b_psf2 = galsim.BoundsI( xmin=self.xyI.x-int(self.params['psf_stampsize']*self.params['oversample'])/2+1,
    #                     ymin=self.xyI.y-int(self.params['psf_stampsize']*self.params['oversample'])/2+1,
    #                     xmax=self.xyI.x+int(self.params['psf_stampsize']*self.params['oversample'])/2,
    #                     ymax=self.xyI.y+int(self.params['psf_stampsize']*self.params['oversample'])/2)
    # Create psf stamp with oversampled pixelisation
    psf_stamp = galsim.Image(b_psf, wcs=WCS)
    # self.psf_stamp2 = galsim.Image(b_psf2, wcs=wcs)
    # Draw PSF into postage stamp
    st_model.drawImage(image=psf_stamp,wcs=WCS)
    # self.st_model.drawImage(image=self.psf_stamp2,wcs=wcs)

    sky_level = wfirst.getSkyLevel(bpass, world_pos=galsim.CelestialCoord(ra=np.radians(27.1656)*galsim.radians,
                        dec=np.radians(-16.5013)*galsim.radians), date=Time(61021.800069,format='mjd').datetime)
    sky_level *= (1.0 + wfirst.stray_light_fraction)
    sky_stamp = galsim.Image(bounds=gal_stamp.bounds, wcs=WCS)
    WCS.makeSkyImage(sky_stamp, sky_level)
    sky_stamp += wfirst.thermal_backgrounds['H158']*wfirst.exptime
    sky_stamp.invertSelf()

    return gal_stamp, psf_stamp, weight

def shape(gal_stamp,psf_stamp,weight):

    obs_list=ObsList()
    psf_list=ObsList()

	origin_x = gal_stamp.origin.x
	origin_y = gal_stamp.origin.y
	gal_stamp.setOrigin(0,0)
	wcs = gal_stamp.wcs.affine(image_pos=gal_stamp.true_center)

    gal_jacob=Jacobian(
        row=origin_y,
        col=origin_x,
        dvdrow=wcs.dvdy,
        dvdcol=wcs.dvdx,
        dudrow=wcs.dudy,
        dudcol=wcs.dudx)
    psf_jacob=Jacobian(
        row=origin_y,
        col=origin_x,
        dvdrow=wcs.dvdy,
        dvdcol=wcs.dvdx,
        dudrow=wcs.dudy,
        dudcol=wcs.dudx)

    psf_obs = Observation(psf_stamp.array, jacobian=psf_jacob, meta={'offset_pixels':None,'file_id':None})
    obs = Observation(gal_stamp.array, weight=weight.array, jacobian=gal_jacob, psf=psf_obs, meta={'offset_pixels':None,'file_id':None})
    obs.set_noise(1./weight.array)
    obs_list.append(obs)

    pix_range = 0.005
    e_range = 0.05
    fdev = 0.1
    def pixe_guess(n):
        return 2.*n*np.random.random() - n

    multi_obs_list=MultiBandObsList()
    multi_obs_list.append(obs_list)

    # possible models are 'exp','dev','bdf'
    flux=1000.0
    cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.wfirst.pixel_scale, galsim.wfirst.pixel_scale)
    gp = ngmix.priors.GPriorBA(0.2)
    hlrp = ngmix.priors.FlatPrior(1.0e-4, 1.0e9)
    fracdevp = ngmix.priors.TruncatedGaussian(0.5, 0.1, -2, 3)
    fluxp = ngmix.priors.FlatPrior(-1, 1.0e9) # not sure what lower bound should be in general

    prior = joint_prior.PriorBDFSep(cp, gp, hlrp, fracdevp, fluxp)
    fitter = mof.KGSMOF([multi_obs_list], 'bdf', prior)
    # center1 + center2 + shape + hlr + fracdev + fluxes for each object
    guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),gal['size'],0.5+pixe_guess(fdev),1000.])
    fitter.go(guess)

    res = fitter.get_object_result(0)
    res0 = fitter.get_result()

    return res['pars'][2], res['pars'][3], res0['flags']

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

x = np.random.rand(100)*3500+500
y = np.random.rand(100)*3500+500
e1 = []
e2 = []
flags = []

params     = yaml.load(open('wfirst_imsim_paper1/code/fid.yaml'))
# Do some parsing
for key in params.keys():
    if params[key]=='None':
       params[key]=None
    if params[key]=='none':
        params[key]=None
    if params[key]=='True':
        params[key]=True
    if params[key]=='False':
        params[key]=False
modify_image_ = modify_image(params)

for i in range(len(x)):
    if rank==0:
        print i

    gal = fio.FITS('/fs/scratch/cond0083/wfirst_sim_out/truth/fiducial_lensing_galaxia_truth_gal.fits')[-1].read()[int(i+1000*(rank))]

	# Galsim image coordinate object 
	xy = galsim.PositionD(x[i],y[i])

	# Galsim integer image coordinate object 
	xyI = galsim.PositionI(int(xy.x),int(xy.y))

	# Galsim image coordinate object holding offset from integer pixel grid 
	offset = xy-xyI

	# Define the local_wcs at this world position
	local_wcs = WCS.local(xy)

    gal_stamp, psf_stamp, weight = draw_galaxy(xyI)

    e1_,e2_,flags_ = shape(gal_stamp,psf_stamp,weight)
    e1.append(e1_)
    e2.append(e2_)
    flags.append(flags_)

e1 = np.array(e1)
e2 = np.array(e2)
flags = np.array(flags)
if rank==0:
    for i in range(1,size):
        e1_ = comm.recv(source=i)
        e2_ = comm.recv(source=i)
        flags_ = comm.recv(source=i)
        e1 = np.append(e1,e1_)
        e2 = np.append(e2,e2_)
        flags = np.append(flags,flags_)
    print np.mean(e1),np.mean(e2),np.mean(e1[flags==0]),np.mean(e2[flags==0])
else:
    comm.send(e1, dest=0)
    comm.send(e2, dest=0)
    comm.send(flags, dest=0)
