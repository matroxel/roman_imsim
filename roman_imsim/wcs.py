import galsim
import romanisim.models as models

from astropy.time import Time
from astropy.wcs import WCS
from galsim.angle import Angle
from galsim.celestial import CelestialCoord
from galsim.config import RegisterWCSType, WCSBuilder


class RomanWCS(WCSBuilder):
    def buildWCS(self, config, base, logger):

        req = {
            "SCA": int,
            "ra": Angle,
            "dec": Angle,
            "pa": Angle,
            "mjd": float,
        }
        opt = {"max_sun_angle": float, "force_cvz": bool}

        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        if "max_sun_angle" in kwargs:
            models.parameters.max_sun_angle = kwargs["max_sun_angle"]
            models.wcs_utils.max_sun_angle = kwargs["max_sun_angle"]
        pointing = CelestialCoord(ra=kwargs["ra"], dec=kwargs["dec"])
        wcs = models.wcs_utils.getWCS(
            world_pos=pointing,
            PA=kwargs["pa"],
            date=Time(kwargs["mjd"], format="mjd").datetime,
            SCAs=kwargs["SCA"],
            PA_is_FPA=True,
        )[kwargs["SCA"]]
        return wcs


RegisterWCSType("RomanWCS", RomanWCS())


class ImcomWCS(WCSBuilder):
    def _get_IMCOM_WCS(
        self,
        world_pos,
        xsize=2688,
        ysize=2688,
        pixel_scale=0.0390625,
        as_astropy=False,
        crpix=None,
    ):
        """
        Create a WCS for IMCOM coadds block
        """
        w_astropy = WCS(naxis=2)
        w_astropy.wcs.ctype = ["RA---STG", "DEC--STG"]
        w_astropy.wcs.crval = [
            world_pos.ra / galsim.degrees,
            world_pos.dec / galsim.degrees,
        ]
        if crpix is not None:
            w_astropy.wcs.crpix = crpix
        else:
            w_astropy.wcs.crpix = [xsize / 2.0, ysize / 2.0]
        w_astropy.wcs.cdelt = [-pixel_scale / 3600.0, pixel_scale / 3600.0]
        w_astropy.wcs.cunit = ["deg", "deg"]

        if as_astropy:
            return w_astropy

        header = w_astropy.to_header()
        wcs = galsim.GSFitsWCS(header=header)
        wcs.header = header

        return wcs

    def buildWCS(self, config, base, logger):
        req = {}
        opt = {
            "coadd_file": str,
            "hdu": int,
            "ra": float,
            "dec": float,
            "crpix1": float,
            "crpix2": float,
        }
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        if "coadd_file" in kwargs:
            wcs = galsim.GSFitsWCS(file_name=kwargs["coadd_file"], hdu=kwargs.get("hdu", 0))
        elif "ra" in kwargs and "dec" in kwargs:
            if "crpix1" in kwargs and "crpix2" in kwargs:
                crpix = (kwargs["crpix1"], kwargs["crpix2"])
            else:
                crpix = None
            world_pos = CelestialCoord(
                ra=kwargs["ra"] * galsim.degrees,
                dec=kwargs["dec"] * galsim.degrees,
            )
            wcs = self._get_IMCOM_WCS(
                world_pos=world_pos,
                xsize=base["image"]["xsize"],
                ysize=base["image"]["ysize"],
                pixel_scale=base["image"]["pixel_scale"],
                crpix=crpix,
            )
        else:
            raise galsim.GalSimConfigError("ImcomWCS requires either coadd_file or both ra and dec")
        return wcs


RegisterWCSType("ImcomWCS", ImcomWCS())
