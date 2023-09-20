import galsim
import galsim.roman as roman
from galsim.config import WCSBuilder,RegisterWCSType

class RomanWCS(WCSBuilder):

    def buildWCS(self, config, base, logger):

        req = { 'SCA' : int,
                'ra'  : float,
                'dec' : float,
                'pa'  : float,
                'mjd' : float
              }

        kwargs, safe = galsim.config.GetAllParams(config, base, req=req)
        pointing = CelestialCoord(ra=kwargs['ra']*galsim.degrees, dec=kwargs['dec']*galsim.degrees)
        wcs = roman.getWCS(world_pos        = pointing,
                                PA          = kwargs['pa']*galsim.degrees,
                                date        = Time(kwargs['mjd'],format='mjd').datetime,
                                SCAs        = kwargs['SCA'],
                                PA_is_FPA   = True
                                )[kwargs['SCA']]
        return wcs

RegisterWCSType('RomanWCS', RomanWCS())