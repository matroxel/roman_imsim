from functools import lru_cache
from dataclasses import dataclass, fields, MISSING
import numpy as np
import galsim
import galsim.roman as roman
from galsim.config import StampBuilder, RegisterStampType, GetAllParams, GetInputObj




import warnings
import sqlite3
import numpy as np
import astropy
import astropy.coordinates
import pandas as pd
from galsim.config import InputLoader, RegisterInputType, RegisterValueType
import galsim
from lsst.obs.lsst.translators.lsst import SIMONYI_LOCATION as RUBIN_LOC




import galsim.roman as roman
import galsim.config
from galsim.config import RegisterObjectType,RegisterInputType

class RomanPSF(object):
    """Class building needed Roman PSFs.
    """
    def __init__(self):

        logger = galsim.config.LoggerWrapper(logger)

        req = {}
        opt = {
            'n_waves' : int,
            'use_SCA_pos': bool,
        }
        ignore += ['extra_aberrations']

        # If SCA is in base, then don't require it in the config file.
        # (Presumably because using Roman image type, which sets it there for convenience.)
        if 'SCA' in base:
            opt['SCA'] = int
        else:
            req['SCA'] = int

        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore)

        # If not given in kwargs, then it must have been in base, so this is ok.
        if 'SCA' not in kwargs:
            kwargs['SCA'] = base['SCA']

        # It's slow to make a new PSF for each galaxy at every location.
        # So the default is to use the same PSF object for the whole image.
        if kwargs.pop('use_SCA_pos', False):
            SCA_pos = base['image_pos']
        else: 
            SCA_pos = None

        kwargs['extra_aberrations'] = galsim.config.ParseAberrations('extra_aberrations', config, base, 'RomanPSF')

        WCS    = galsim.config.BuildWCS(base['image'], 'wcs', base, logger=logger)
        bpass  = galsim.config.BuildBandpass(base['image'], 'bandpass', base, logger)[0]

        self.PSF = {}
        self.PSF[8] = roman.getPSF(kwargs['SCA'],
                                bpass.name,
                                SCA_pos             = SCA_pos,
                                wcs                 = WCS,
                                pupil_bin           = 8,
                                n_waves             = kwargs['n_waves'],
                                logger              = logger,
                                # wavelength          = self.bpass.effective_wavelength,
                                extra_aberrations   = kwargs['extra_aberrations']
                                ).withGSParams(galsim.GSParams(maximum_fft_size=16384))
        self.PSF[4] = roman.getPSF(kwargs['SCA'],
                                bpass.name,
                                SCA_pos             = SCA_pos,
                                wcs                 = WCS,
                                pupil_bin           = 4,
                                n_waves             = kwargs['n_waves'],
                                logger              = logger,
                                wavelength          = bpass.effective_wavelength,
                                extra_aberrations   = kwargs['extra_aberrations']
                                ).withGSParams(galsim.GSParams(maximum_fft_size=16384, folding_threshold=1e-3))
        self.PSF[2] = roman.getPSF(kwargs['SCA'],
                                bpass.name,
                                SCA_pos             = SCA_pos,
                                wcs                 = WCS,
                                pupil_bin           = 2,
                                n_waves             = kwargs['n_waves'],
                                logger              = logger,
                                wavelength          = bpass.effective_wavelength,
                                extra_aberrations   = kwargs['extra_aberrations']
                                ).withGSParams(galsim.GSParams(maximum_fft_size=16384, folding_threshold=1e-4))
        self.PSF['achromatic'] = roman.getPSF(kwargs['SCA'],
                                bpass.name,
                                SCA_pos             = SCA_pos,
                                wcs                 = WCS,
                                pupil_bin           = 8,
                                n_waves             = kwargs['n_waves'],
                                logger              = logger,
                                wavelength          = bpass.effective_wavelength,
                                extra_aberrations   = kwargs['extra_aberrations']
                                )

    def getPSF(self):
        """
        Return a PSF to be convolved with sources.

        @param [in] what pupil binning to request.
        """
        return self.PSF

def BuildRomanPSF(config, base, ignore, gsparams, logger):
    """Build the Roman PSF from the information in the config file.
    """
    roman_psf = galsim.config.GetInputObj('roman_psf', config, base, 'RomanPSF')
    psf = roman_psf.getPSF()
    return psf, False

# Register this as a valid type
RegisterInputType('roman_psf', RomanPSF())
RegisterObjectType('RomanPSF', BuildRomanPSF, input_type='roman_psf')