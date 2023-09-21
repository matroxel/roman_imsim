import galsim
import galsim.roman as roman
import galsim.config
from galsim.config import RegisterObjectType,RegisterInputType,OpticalPSF,InputLoader

class RomanPSF(object):
    """Class building needed Roman PSFs.
    """
    def __init__(self, SCA=None, SCA_pos=None, WCS=None, n_waves=None, bpass=None, extra_aberrations=None, logger=None):

        logger = galsim.config.LoggerWrapper(logger)

        self.PSF = {}
        self.PSF[8] = roman.getPSF(SCA,
                                bpass.name,
                                SCA_pos             = SCA_pos,
                                wcs                 = WCS,
                                pupil_bin           = 8,
                                n_waves             = n_waves,
                                logger              = logger,
                                # wavelength          = self.bpass.effective_wavelength,
                                extra_aberrations   = extra_aberrations
                                ).withGSParams(galsim.GSParams(maximum_fft_size=16384))
        self.PSF[4] = roman.getPSF(SCA,
                                bpass.name,
                                SCA_pos             = SCA_pos,
                                wcs                 = WCS,
                                pupil_bin           = 4,
                                n_waves             = n_waves,
                                logger              = logger,
                                wavelength          = bpass.effective_wavelength,
                                extra_aberrations   = extra_aberrations
                                ).withGSParams(galsim.GSParams(maximum_fft_size=16384, folding_threshold=1e-3))
        self.PSF[2] = roman.getPSF(SCA,
                                bpass.name,
                                SCA_pos             = SCA_pos,
                                wcs                 = WCS,
                                pupil_bin           = 2,
                                n_waves             = n_waves,
                                logger              = logger,
                                wavelength          = bpass.effective_wavelength,
                                extra_aberrations   = extra_aberrations
                                ).withGSParams(galsim.GSParams(maximum_fft_size=16384, folding_threshold=1e-4))
        self.PSF['achromatic'] = roman.getPSF(SCA,
                                bpass.name,
                                SCA_pos             = SCA_pos,
                                wcs                 = WCS,
                                pupil_bin           = 8,
                                n_waves             = n_waves,
                                logger              = logger,
                                wavelength          = bpass.effective_wavelength,
                                extra_aberrations   = extra_aberrations
                                )

    def getPSF(self):
        """
        Return a PSF to be convolved with sources.

        @param [in] what pupil binning to request.
        """
        return self.PSF


class PSFLoader(InputLoader):
    """Custom AtmosphericPSF loader that only loads the atmosphere once per exposure.

    Note: For now, this just loads the atmosphere once for an entire imsim run.
          If we ever decide we want to have a single config processing run handle multiple
          exposures (rather than just multiple CCDs for a single exposure), we'll need to
          reconsider this implementation.
    """
    def __init__(self):
        # Override some defaults in the base init.
        super().__init__(init_func=RomanPSF,
                         takes_logger=True, use_proxy=False)

    def getKwargs(self, config, base, logger):
        logger.debug("Get kwargs for PSF")

        req = {}
        opt = {
            'n_waves' : int,
            'use_SCA_pos': bool,
        }
        ignore = ['extra_aberrations']

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

        kwargs['WCS']    = galsim.config.BuildWCS(base['image'], 'wcs', base, logger=logger)
        kwargs['bpass']  = galsim.config.BuildBandpass(base['image'], 'bandpass', base, logger)[0]

        logger.debug("kwargs = %s",kwargs)

        return kwargs, False

def BuildRomanPSF(config, base, ignore, gsparams, logger):
    """Build the Roman PSF from the information in the config file.
    """
    roman_psf = galsim.config.GetInputObj('roman_psf', config, base, 'PSFLoader')
    psf = roman_psf.getPSF()
    return psf, False

# Register this as a valid type
RegisterInputType('romanpsf_loader', PSFLoader())
RegisterObjectType('roman_psf', BuildRomanPSF, input_type='romanpsf_loader')