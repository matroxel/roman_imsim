import galsim
import galsim.config
import numpy as np
import romanisim.models as models

from astropy.io import fits
from astropy.time import Time
from galsim.config import RegisterImageType
from galsim.config.image_scattered import ScatteredImageBuilder
from galsim.errors import GalSimConfigError, GalSimConfigValueError
from galsim.image import Image

valid_coadd_geometry_types = {}


class CoaddGeometry:
    """Interface for configuring the geometry of a Roman coadd image."""

    def configure(self, config, image_config, base, logger):
        raise NotImplementedError("configure must be implemented by CoaddGeometry subclasses")


class ImcomFileGeometry(CoaddGeometry):
    """Derive image dimensions and WCS from an existing IMCOM product."""

    def configure(self, config, image_config, base, logger):
        req = {"file_name": str}
        opt = {"hdu": int, "pixel_scale": float}
        params = galsim.config.GetAllParams(config, base, req=req, opt=opt)[0]

        file_name = params["file_name"]
        hdu = params.get("hdu", 0)
        header = fits.getheader(file_name, hdu)
        try:
            xsize = int(header["NAXIS1"])
            ysize = int(header["NAXIS2"])
        except KeyError as error:
            raise GalSimConfigError(
                f"IMCOM geometry HDU {hdu} in {file_name!r} does not define NAXIS1/NAXIS2"
            ) from error

        if "pixel_scale" in params:
            pixel_scale = params["pixel_scale"]
        elif "pixel_scale" in image_config:
            pixel_scale = galsim.config.ParseValue(image_config, "pixel_scale", base, float)[0]
        else:
            file_wcs = galsim.GSFitsWCS(file_name=file_name, hdu=hdu)
            center = galsim.PositionD((xsize + 1) / 2.0, (ysize + 1) / 2.0)
            pixel_scale = np.sqrt(file_wcs.local(image_pos=center).pixelArea())

        image_config["xsize"] = xsize
        image_config["ysize"] = ysize
        image_config["pixel_scale"] = pixel_scale
        if "wcs" not in image_config:
            image_config["wcs"] = {
                "type": "ImcomWCS",
                "coadd_file": file_name,
                "hdu": hdu,
            }

        return xsize, ysize, pixel_scale


class SkyPositionGeometry(CoaddGeometry):
    """Construct coadd geometry from a sky position and pixel grid."""

    def configure(self, config, image_config, base, logger):
        req = {
            "ra": float,
            "dec": float,
            "xsize": int,
            "ysize": int,
            "pixel_scale": float,
        }
        opt = {"crpix": list}
        params = galsim.config.GetAllParams(config, base, req=req, opt=opt)[0]

        wcs_config = {
            "type": "ImcomWCS",
            "ra": params["ra"],
            "dec": params["dec"],
        }
        if "crpix" in params:
            if len(params["crpix"]) != 2:
                raise GalSimConfigError("SkyPosition crpix must contain exactly two values")
            wcs_config["crpix1"] = float(params["crpix"][0])
            wcs_config["crpix2"] = float(params["crpix"][1])

        image_config["xsize"] = params["xsize"]
        image_config["ysize"] = params["ysize"]
        image_config["pixel_scale"] = params["pixel_scale"]
        image_config["wcs"] = wcs_config

        return params["xsize"], params["ysize"], params["pixel_scale"]


def RegisterCoaddGeometryType(geometry_type, geometry):
    """Register a geometry provider for ``image.geometry.type``."""

    if not isinstance(geometry, CoaddGeometry):
        raise TypeError("geometry must be an instance of CoaddGeometry")
    valid_coadd_geometry_types[geometry_type] = geometry


RegisterCoaddGeometryType("ImcomFile", ImcomFileGeometry())
RegisterCoaddGeometryType("SkyPosition", SkyPositionGeometry())


class RomanCoaddImageBuilder(ScatteredImageBuilder):

    def _configure_geometry(self, config, base, image_num, logger):
        """Materialize geometry before inputs request the image WCS and size."""

        logger = galsim.config.LoggerWrapper(logger)
        original_index_key = base.get("index_key")
        base["index_key"] = "image_num"
        base["image_num"] = image_num
        try:
            if "geometry" not in config or not isinstance(config["geometry"], dict):
                raise GalSimConfigError("image.geometry must be a dict")
            geometry_config = config["geometry"]
            geometry_type = galsim.config.ParseValue(geometry_config, "type", base, str)[0]
            try:
                geometry = valid_coadd_geometry_types[geometry_type]
            except KeyError as error:
                raise GalSimConfigValueError(
                    "Invalid Roman coadd geometry type",
                    geometry_type,
                    list(valid_coadd_geometry_types),
                ) from error

            xsize, ysize, pixel_scale = geometry.configure(
                geometry_config,
                config,
                base,
                logger,
            )
            base["coadd_geometry_type"] = geometry_type
            base["coadd_pixel_scale"] = pixel_scale
            logger.warning(
                "Roman coadd geometry %s uses pixel scale %.8f arcsec/pixel",
                geometry_type,
                pixel_scale,
            )
            return xsize, ysize, pixel_scale
        finally:
            base["index_key"] = original_index_key

    def getNObj(self, config, base, image_num, logger=None, approx=False):
        """Prepare coadd geometry before loading inputs used to count objects."""

        self._configure_geometry(config, base, image_num, logger)
        return super().getNObj(config, base, image_num, logger=logger, approx=approx)

    def setup(self, config, base, image_num, obj_num, ignore, logger):
        """Do the initialization and setup for building the image.

        This figures out the size that the image will be, but doesn't actually build it yet.

        Parameters:
            config:     The configuration dict for the image field.
            base:       The base configuration dict.
            image_num:  The current image number.
            obj_num:    The first object number in the image.
            ignore:     A list of parameters that are allowed to be in config that we can
                        ignore here. i.e. it won't be an error if these parameters are present.
            logger:     If given, a logger object to log progress.

        Returns:
            xsize, ysize
        """
        logger.debug(
            "image %d: Building Roman coadd: image, obj = %d,%d",
            image_num,
            image_num,
            obj_num,
        )

        self.nobjects = self.getNObj(config, base, image_num, logger=logger)
        logger.debug("image %d: nobj = %d", image_num, self.nobjects)

        # These are allowed for Scattered, or belong to the deprecated flat
        # coadd configuration, but are not image-builder parameters.
        extra_ignore = [
            "image_pos",
            "world_pos",
            "stamp_size",
            "stamp_xsize",
            "stamp_ysize",
            "nobjects",
            "coadd_file",
            "white_noise_weight",
            "pink_noise_weight",
            "ignore_noise",
            "xsize",
            "ysize",
            "geometry",
        ]
        req = {
            "SCA": int,
            "filter": str,
            "mjd": float,
            "exptime": float,
        }
        opt = {
            "draw_method": str,
            "use_fft_bright": bool,
        }
        params = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore + extra_ignore)[0]

        self.sca = params["SCA"]
        base["SCA"] = self.sca
        self.filter = params["filter"]
        self.mjd = params["mjd"]
        self.exptime = params["exptime"]

        # If draw_method isn't in image field, it may be in stamp.  Check.
        self.draw_method = params.get("draw_method", base.get("stamp", {}).get("draw_method", "auto"))

        self.pixel_scale = base["coadd_pixel_scale"]
        return int(config["xsize"]), int(config["ysize"])

    def buildImage(self, config, base, image_num, obj_num, logger):
        """Build an Image containing multiple objects placed at arbitrary locations.

        Parameters:
            config:     The configuration dict for the image field.
            base:       The base configuration dict.
            image_num:  The current image number.
            obj_num:    The first object number in the image.
            logger:     If given, a logger object to log progress.

        Returns:
            the final image and the current noise variance in the image as a tuple
        """
        full_xsize = base["image_xsize"]
        full_ysize = base["image_ysize"]
        wcs = base["wcs"]

        full_image = Image(full_xsize, full_ysize, dtype=float)
        full_image.setOrigin(base["image_origin"])
        full_image.wcs = wcs
        full_image.setZero()

        full_image.header = galsim.FitsHeader()
        full_image.header["EXPTIME"] = self.exptime
        full_image.header["MJD-OBS"] = self.mjd
        full_image.header["DATE-OBS"] = Time(self.mjd, format="mjd").datetime.isoformat()
        full_image.header["FILTER"] = self.filter
        full_image.header["ZPTMAG"] = 2.5 * np.log10(self.exptime * models.parameters.collecting_area)

        base["current_image"] = full_image

        if "image_pos" in config and "world_pos" in config:
            raise galsim.GalSimConfigValueError(
                "Both image_pos and world_pos specified for Scattered image.",
                (config["image_pos"], config["world_pos"]),
            )

        if "image_pos" not in config and "world_pos" not in config:
            xmin = base["image_origin"].x
            xmax = xmin + full_xsize - 1
            ymin = base["image_origin"].y
            ymax = ymin + full_ysize - 1
            config["image_pos"] = {
                "type": "XY",
                "x": {"type": "Random", "min": xmin, "max": xmax},
                "y": {"type": "Random", "min": ymin, "max": ymax},
            }

        nbatch = self.nobjects // 1000 + 1
        for batch in range(nbatch):
            start_obj_num = self.nobjects * batch // nbatch
            end_obj_num = self.nobjects * (batch + 1) // nbatch
            nobj_batch = end_obj_num - start_obj_num
            if nbatch > 1:
                logger.warning(
                    "Start batch %d/%d with %d objects [%d, %d)",
                    batch + 1,
                    nbatch,
                    nobj_batch,
                    start_obj_num,
                    end_obj_num,
                )
            stamps, current_vars = galsim.config.BuildStamps(
                nobj_batch,
                base,
                logger=logger,
                obj_num=start_obj_num,
                do_noise=False,
            )
            base["index_key"] = "image_num"

            for k in range(nobj_batch):
                # This is our signal that the object was skipped.
                if stamps[k] is None:
                    continue
                bounds = stamps[k].bounds & full_image.bounds
                if not bounds.isDefined():  # pragma: no cover
                    # These noramlly show up as stamp==None, but technically it is possible
                    # to get a stamp that is off the main image, so check for that here to
                    # avoid an error.  But this isn't covered in the imsim test suite.
                    continue

                logger.debug(
                    "image %d: full bounds = %s",
                    image_num,
                    str(full_image.bounds),
                )
                logger.debug(
                    "image %d: stamp %d bounds = %s",
                    image_num,
                    k + start_obj_num,
                    str(stamps[k].bounds),
                )
                logger.debug("image %d: Overlap = %s", image_num, str(bounds))
                full_image[bounds] += stamps[k][bounds]
            stamps = None

            # # [TODO]
            # break

        # # Bring the image so far up to a flat noise variance
        # current_var = FlattenNoiseVariance(
        #         base, full_image, stamps, current_vars, logger)

        logger.info("Roman native pixel scale: %.5f", models.parameters.pixel_scale)
        full_image /= (self.pixel_scale / models.parameters.pixel_scale) ** 2

        return full_image, None


# Register this as a valid type
RegisterImageType("roman_coadd", RomanCoaddImageBuilder())
