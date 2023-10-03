import galsim
import galsim.roman as roman
import galsim.config
from galsim.config import RegisterObjectType,RegisterInputType,OpticalPSF,InputLoader

class RomanPSF(object):
    """Class building needed Roman PSFs.
    """
    def __init__(self, SCA=None, WCS=None, n_waves=None, bpass=None, extra_aberrations=None, logger=None):

        logger = galsim.config.LoggerWrapper(logger)

        corners = [galsim.PositionD(1,1),galsim.PositionD(1,roman.n_pix),galsim.PositionD(roman.n_pix,1),galsim.PositionD(roman.n_pix,roman.n_pix)]
        tags = ['ll','lu','ul','uu']
        self.PSF = {}
        for pupil_bin in [8,4,2,'achromatic']:
            self.PSF[pupil_bin] = {}
            for tag,SCA_pos in tuple(zip(tags,corners)):
                self.PSF[pupil_bin][tag] = self._psf_call(SCA,bpass,SCA_pos,WCS,pupil_bin,n_waves,logger,extra_aberrations)

    def _parse_pupil_bin(self,pupil_bin):
        if pupil_bin=='achromatic':
            return 8
        else:
            return pupil_bin

    def _psf_call(self,SCA,bpass,SCA_pos,WCS,pupil_bin,n_waves,logger,extra_aberrations):

        if pupil_bin==8:
            psf = roman.getPSF(SCA,
                    bpass.name,
                    SCA_pos             = SCA_pos,
                    wcs                 = WCS,
                    pupil_bin           = pupil_bin,
                    n_waves             = n_waves,
                    logger              = logger,
                    # Don't set wavelength for this one.
                    # We want this to be chromatic for photon shooting.
                    # wavelength          = bpass.effective_wavelength,
                    extra_aberrations   = extra_aberrations
                    )
        else:
            psf = roman.getPSF(SCA,
                    bpass.name,
                    SCA_pos             = SCA_pos,
                    wcs                 = WCS,
                    pupil_bin           = self._parse_pupil_bin(pupil_bin),
                    n_waves             = n_waves,
                    logger              = logger,
                    # Note: setting wavelength makes it achromatic.
                    # We only use pupil_bin = 2,4 for FFT objects.
                    wavelength          = bpass.effective_wavelength,
                    extra_aberrations   = extra_aberrations
                    )
        if pupil_bin==4:
            return psf.withGSParams(maximum_fft_size=16384, folding_threshold=1e-3)
        elif pupil_bin==2:
            return psf.withGSParams(maximum_fft_size=16384, folding_threshold=1e-4)
        else:
            return psf.withGSParams(maximum_fft_size=16384)

    def getPSF(self,pupil_bin,pos):
        """
        Return a PSF to be convolved with sources.

        @param [in] what pupil binning to request.
        """

        wll = (roman.n_pix-pos.x)*(roman.n_pix-pos.y)
        wlu = (roman.n_pix-pos.x)*(pos.y-1)
        wul = (pos.x-1)*(roman.n_pix-pos.y)
        wuu = (pos.x-1)*(pos.y-1)
        psf = self.PSF[pupil_bin]
        return (wll*psf['ll']+wlu*psf['lu']+wul*psf['ul']+wuu*psf['uu'])/((roman.n_pix-1)*(roman.n_pix-1))

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

        kwargs['extra_aberrations'] = galsim.config.ParseAberrations('extra_aberrations', config, base, 'RomanPSF')
        kwargs['WCS']    = galsim.config.BuildWCS(base['image'], 'wcs', base, logger=logger)
        kwargs['bpass']  = galsim.config.BuildBandpass(base['image'], 'bandpass', base, logger)[0]

        logger.debug("kwargs = %s",kwargs)

        return kwargs, False

# def BuildRomanPSF(config, base, ignore, gsparams, logger):
#     """Build the Roman PSF from the information in the config file.
#     """
#     roman_psf = galsim.config.GetInputObj('romanpsf_loader', config, base, 'BuildRomanPSF')
#     SCA_pos = base['image_pos']
#     psf = roman_psf.getPSF()
#     return psf, False

# Register this as a valid type
RegisterInputType('roman_psf', PSFLoader())
# RegisterObjectType('roman_psf', BuildRomanPSF, input_type='romanpsf_loader')
