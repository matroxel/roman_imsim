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

class modify_image(object):
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

    def add_effects(self,im,pointing,radec,local_wcs,rng,phot=False, ps_save=False):
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
        6) add_persistence
        7) nonlinearity
        8) interpix_cap
        9) Read noise
        10) e_to_ADU
        11) quantize

        Where does persistence get added? Immediately before/after background?
        Chien-Hao: I added persistence between dark current and nonlinearity.
        """

        self.rng       = rng
        self.noise     = self.init_noise_model()

        im, sky_image = self.add_background(im,pointing,radec,local_wcs,phot=phot) # Add background to image and save background
        im = self.add_poisson_noise(im,sky_image,phot=phot) # Add poisson noise to image
        im = self.recip_failure(im) # Introduce reciprocity failure to image
        im.quantize() # At this point in the image generation process, an integer number of photons gets detected
        im = self.dark_current(im) # Add dark current to image
        if ps_save: #don't apply persistence for stamps
            im = self.add_persistence(im, pointing)
        im,dq = self.nonlinearity(im) # Apply nonlinearity
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
        if not self.params['use_background']:
            return im,None
        sky_image.invertSelf()

        return im, sky_image,dq


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
            self.noise = self.init_noise_model()

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

    def add_persistence(self, img, pointing):
        """
        Applying the persistence effect.

        Even after reset, some charges from prior illuminations are trapped in defects of semiconductors.
        Trapped charges are gradually released and generate the flux-dependent persistence signal.
        Here we adopt the same fermi-linear model to describe the illumination dependence and time dependence
        of the persistence effect for all SCAs.
        """
        if not self.params['use_persistence']:
            return img

        prev_exposures_filename = get_filename(self.params['out_path'],
                                'prev_exp',
                                'prev_exp',
                                var=str(pointing.sca),
                                ftype='pkl',
                                overwrite=False)
        try:
            with open(prev_exposures_filename, 'rb') as fp:
                prev_exposures = pickle.load(fp)
        except FileNotFoundError:
            prev_exposures = []

        if not hasattr(prev_exposures,'__iter__'):
            raise TypeError("prev_exposures must be a list of Image instances")
        n_exp = len(prev_exposures)
        for i in range(n_exp):
            img._array += galsim.wfirst.wfirst_detectors.fermi_linear(
            prev_exposures[i].array,
             (0.5+i)*galsim.wfirst.exptime)*galsim.wfirst.exptime

        prev_exposures = [img.copy()] + prev_exposures[:]
        with open(prev_exposures_filename, 'wb') as fw:
            pickle.dump(prev_exposures, fw)

        return img

    def nonlinearity(self,im,NLfunc=wfirst.NLfunc,saturation=100000):
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

        # Saturation
        dq = np.zeros_like(im.array,dtype='int16')
        dq[np.where(im.array>saturation)] = 1
        im.array[:,:] = np.clip(im.array,None,saturation)

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

        return im,dq

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