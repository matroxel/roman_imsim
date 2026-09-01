import galsim
import numpy as np
import romanisim.models as models

from astropy.io import fits
from astropy.time import Time
from galsim.config import NoiseBuilder, RegisterNoiseType
from galsim.errors import GalSimConfigError, GalSimConfigValueError

valid_coadd_noise_models = {}


class RomanNoiseBuilder(NoiseBuilder):
    def addNoise(self, config, base, image, rng, current_var, draw_method, logger):
        """Read the noise parameters from the config dict and add the appropriate noise to the
        given image.

        Parameters
        ----------
            config: dict
                The configuration dict for the noise field.
            base: dict
                The base configuration dict.
            im: galsim.Image
                The image onto which to add the noise
            rng: galsim.BaseDeviate
                The random number generator to use for adding the noise.
            current_var: float
                The current noise variance present in the image already.
            draw_method: str
                The method that was used to draw the objects on the image.
            logger: logging.Logger
                If given, a logger object to log progress.

        Returns
        -------
        var (None)
            the variance of the noise model (units are ADU if gain != 1)
            NOT IMPLEMENTED
        """

        opt = {
            "mjd": float,
            "stray_light": bool,
            "thermal_background": bool,
            "reciprocity_failure": bool,
            "dark_current": dict,
            "nonlinearity": dict,
            "ipc": dict,
            "read_noise": dict,
            "sky_subtract": bool,
            "gain": dict,
        }

        params, safe = galsim.config.GetAllParams(config, base, req={}, opt=opt, ignore=[])

        mjd = params.get("mjd", None)
        stray_light = params.get("stray_light", False)
        thermal_background = params.get("thermal_background", False)
        reciprocity_failure = params.get("reciprocity_failure", False)
        dark_current = params.get("dark_current", {"turn_on": False, "use_crds": False})
        nonlinearity = params.get("nonlinearity", {"turn_on": False, "use_crds": False})
        ipc = params.get("ipc", {"turn_on": False, "use_crds": False})
        read_noise = params.get("read_noise", {"turn_on": False, "use_crds": False})
        sky_subtract = params.get("sky_subtract", True)
        gain = params.get("gain", {"turn_on": False, "use_crds": False})

        base["current_noise_image"] = base["current_image"]
        wcs = base["wcs"]
        bp = base["bandpass"]
        filter_name = bp.name
        exptime, _ = galsim.config.ParseValue(base["image"], "exptime", base, float)
        date = Time(mjd, format="mjd").to_datetime() if mjd is not None else None
        logger.info(
            "image %d: Start RomanSCA detector effects",
            base.get("image_num", 0),
        )

        # Things that will eventually be subtracted (if sky_subtract) will have their expectation
        # value added to sky_image.  So technically, this includes things that aren't just sky.
        # E.g. includes dark_current and thermal backgrounds.
        sky_image = image.copy()
        sky_level = models.backgrounds.getSkyLevel(bp, world_pos=wcs.toWorld(image.true_center), date=date)
        logger.debug("Adding sky_level = %s", sky_level)
        if stray_light:
            logger.debug(
                "Stray light fraction = %s",
                models.parameters.stray_light_fraction,
            )
            sky_level *= 1.0 + models.parameters.stray_light_fraction
        wcs.makeSkyImage(sky_image, sky_level)

        # The other background is the expected thermal backgrounds in this band.
        # These are provided in e-/pix/s, so we have to multiply by the exposure time.
        if thermal_background:
            tb = models.backgrounds.thermal_backgrounds[filter_name] * exptime
            logger.debug("Adding thermal background: %s", tb)
            sky_image += tb

        # The image up to here is an expectation value.
        # Realize it as an integer number of photons.
        poisson_noise = galsim.noise.PoissonNoise(rng)
        if draw_method == "phot":
            logger.debug("Adding poisson noise to sky photons")
            sky_image1 = sky_image.copy()
            sky_image1.addNoise(poisson_noise)
            image.quantize()  # In case any profiles used InterpolatedImage, in which case
            # the image won't necessarily be integers.
            image += sky_image1
        else:
            logger.debug("Adding poisson noise")
            image += sky_image
            image.addNoise(poisson_noise)

        # Apply the detector effects here.  Not all of these are "noise" per se, but they
        # happen interspersed with various noise effects, so apply them all in this step.

        # Note: according to Gregory Mosby & Bernard J. Rauscher, the following effects all
        # happen "simultaneously" in the photo diodes: dark current, persistence,
        # reciprocity failure (aka CRNL), burn in, and nonlinearity (aka CNL).
        # Right now, we just do them in some order, but this could potentially be improved.
        # The order we chose is historical, matching previous recommendations, but Mosby and
        # Rauscher don't seem to think those recommendations are well-motivated.

        # TODO: Add burn-in and persistence here.

        if reciprocity_failure:
            logger.debug("Applying reciprocity failure")
            models.nonlinearity.addReciprocityFailure(img=image)

        if dark_current["turn_on"]:
            logger.debug("Adding dark current: %s")
            dc = models.DarkCurrent(usecrds=dark_current["use_crds"])
            dc.apply(img=image, exptime=exptime)

        if nonlinearity["turn_on"]:
            logger.debug("Applying classical nonlinearity")
            non_linear = models.Nonlinearity(usecrds=nonlinearity["use_crds"])
            non_linear.apply(img=image, electrons=False)

        # Mosby and Rauscher say there are two read noises. One happens before IPC, the other
        # one after.
        # TODO: Add read_noise1
        if ipc["turn_on"]:
            logger.debug("Applying IPC")
            ipc_model = models.IPC(usecrds=ipc["use_crds"])
            ipc_model.apply(img=image)

        if read_noise["turn_on"]:
            logger.debug("Adding read noise")
            rn = models.ReadNoise(usecrds=read_noise["use_crds"])
            rn.apply(img=image)

        if gain["turn_on"]:
            logger.debug("Applying gain")
            gain_model = models.Gain(usecrds=gain["use_crds"])
            gain_model.apply(img=image)

        # Make integer ADU now.
        image.quantize()

        if sky_subtract:
            logger.debug("Subtracting sky image")
            if gain["turn_on"]:
                logger.debug("Applying gain")
                gain_model.apply(img=sky_image)
            sky_image.quantize()
            image -= sky_image

        return None


class CoaddNoiseModel:
    """Interface for coadd-noise implementations."""

    def add_noise(self, config, base, image, rng, logger):
        raise NotImplementedError("add_noise must be implemented by CoaddNoiseModel subclasses")


class ImcomLayerCombinationNoiseModel(CoaddNoiseModel):
    """Add a linear combination of stored layers from an IMCOM product."""

    def _get_source(self, params, base):
        if "file_name" in params:
            return params["file_name"], params.get("hdu", 0)

        source = params.get("source", "geometry")
        if source != "geometry":
            raise GalSimConfigValueError("Invalid coadd noise source", source, ["geometry"])

        geometry_config = base["image"].get("geometry", {})
        if geometry_config.get("type") != "ImcomFile":
            raise GalSimConfigError(
                "ImcomLayerCombination with source=geometry requires image.geometry.type=ImcomFile; "
                "provide noise.file_name or select another noise model"
            )

        file_name = galsim.config.ParseValue(geometry_config, "file_name", base, str)[0]
        if "hdu" in params:
            hdu = params["hdu"]
        elif "hdu" in geometry_config:
            hdu = galsim.config.ParseValue(geometry_config, "hdu", base, int)[0]
        else:
            hdu = 0
        return file_name, hdu

    def add_noise(self, config, base, image, rng, logger):
        req = {"components": list}
        opt = {"file_name": str, "hdu": int, "source": str}
        params = galsim.config.GetAllParams(
            config,
            base,
            req=req,
            opt=opt,
            ignore=["model"],
        )[0]
        if not params["components"]:
            raise GalSimConfigError("ImcomLayerCombination requires at least one component")

        file_name, hdu = self._get_source(params, base)
        with fits.open(file_name, memmap=True) as hdul:
            data = hdul[hdu].data
            if data is None:
                raise GalSimConfigError(f"Noise HDU {hdu} in {file_name!r} contains no data")

            for component in params["components"]:
                component_req = {"data_index": list, "coefficient": float}
                component_opt = {"name": str}
                component_params = galsim.config.GetAllParams(
                    component,
                    base,
                    req=component_req,
                    opt=component_opt,
                )[0]
                data_index = component_params["data_index"]
                if not data_index or not all(isinstance(value, int) for value in data_index):
                    raise GalSimConfigError(
                        "Each noise component data_index must be a non-empty list of integers"
                    )

                try:
                    layer = data[tuple(data_index)]
                except IndexError as error:
                    raise GalSimConfigError(
                        f"Noise component data_index {data_index!r} is invalid for data shape {data.shape}"
                    ) from error
                if layer.shape != image.array.shape:
                    raise GalSimConfigError(
                        f"Noise component shape {layer.shape} does not match image shape {image.array.shape}"
                    )

                coefficient = component_params["coefficient"]
                if not np.isfinite(coefficient):
                    raise GalSimConfigError("Noise component coefficient must be finite")
                name = component_params.get("name", str(data_index))
                logger.debug(
                    "Adding IMCOM noise layer %s from index %s with coefficient %s",
                    name,
                    data_index,
                    coefficient,
                )
                image.array[:] += coefficient * layer

        return None


def RegisterCoaddNoiseModel(model_name, model):
    """Register a model for ``image.noise.model``."""

    if not isinstance(model, CoaddNoiseModel):
        raise TypeError("model must be an instance of CoaddNoiseModel")
    valid_coadd_noise_models[model_name] = model


RegisterCoaddNoiseModel("ImcomLayerCombination", ImcomLayerCombinationNoiseModel())


class RomanCoaddNoiseBuilder(NoiseBuilder):
    """Dispatch GalSim noise processing to a configured coadd-noise model."""

    def addNoise(self, config, base, image, rng, current_var, draw_method, logger):
        model_name = galsim.config.ParseValue(config, "model", base, str)[0]
        try:
            model = valid_coadd_noise_models[model_name]
        except KeyError as error:
            raise GalSimConfigValueError(
                "Invalid Roman coadd noise model",
                model_name,
                list(valid_coadd_noise_models),
            ) from error
        return model.add_noise(config, base, image, rng, logger)


class NoNoiseBuilder(NoiseBuilder):
    def addNoise(self, config, base, image, rng, current_var, draw_method, logger):
        return None


RegisterNoiseType("RomanNoise", RomanNoiseBuilder())
RegisterNoiseType("RomanCoaddNoise", RomanCoaddNoiseBuilder())
RegisterNoiseType("NoNoise", NoNoiseBuilder())
