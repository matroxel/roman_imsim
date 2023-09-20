import galsim
import galsim.roman as roman
from galsim.config import WCSBuilder,RegisterWCSType

class RomanWCS(WCSBuilder):

    def buildWCS(self, config, base, logger):

        req = { 'SCA' : int,
                'ra'  : float,
                'dec' : float,
                'pa'  : float,
                'date': None
              }

        kwargs, safe = galsim.config.GetAllParams(config, base, opt=GalSimWCS._opt_params,
                                                      single=GalSimWCS._single_params)
        pointing = CelestialCoord(ra=kwargs['ra']*galsim.degrees, dec=kwargs['dec']*galsim.degrees)
        wcs = roman.getWCS(world_pos        = pointing,
                                PA          = kwargs['pa']*galsim.degrees,
                                date        = kwargs['date'],
                                SCAs        = kwargs['SCA'],
                                PA_is_FPA   = True
                                )[kwargs['SCA']]
        return wcs

RegisterWCSType('RomanWCS', RomanWCS())