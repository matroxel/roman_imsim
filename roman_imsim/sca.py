import galsim
import galsim.roman as roman
import galsim.config
from galsim.config import RegisterImageType
from galsim.config.image_scattered import ScatteredImageBuilder

class RomanSCAImageBuilder(ScatteredImageBuilder):

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
        logger.debug('image %d: Building RomanSCA: image, obj = %d,%d',
                     image_num,image_num,obj_num)

        self.nobjects = self.getNObj(config, base, image_num, logger=logger)
        logger.debug('image %d: nobj = %d',image_num,self.nobjects)

        # These are allowed for Scattered, but we don't use them here.
        extra_ignore = [ 'image_pos', 'world_pos', 'stamp_size', 'stamp_xsize', 'stamp_ysize',
                         'nobjects' ]
        req = {
            'SCA' : int,
            'ra' : Angle,
            'dec' : Angle,
            'pa'  : Angle,
            'filter' : str,
            'date' : None,  # Should be a datetime.datetime instance
        }
        opt = {
            'draw_method' : str,
            'stray_light' : bool,
            'thermal_background' : bool,
            'reciprocity_failure' : bool,
            'dark_current' : bool,
            'nonlinearity' : bool,
            'ipc' : bool,
            'read_noise' : bool,
            'sky_subtract' : bool,
        }
        params = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore+extra_ignore)[0]

        self.sca = params['SCA']
        base['SCA'] = self.sca

        self.exptime = params.get('exptime', roman.exptime)  # Default is roman standard exposure time.
        self.stray_light = params.get('stray_light', False)
        self.thermal_background = params.get('thermal_background', False)
        self.reciprocity_failure = params.get('reciprocity_failure', False)
        self.dark_current = params.get('dark_current', False)
        self.nonlinearity = params.get('nonlinearity', False)
        self.ipc = params.get('ipc', False)
        self.read_noise = params.get('read_noise', False)
        self.sky_subtract = params.get('sky_subtract', False)

        # If draw_method isn't in image field, it may be in stamp.  Check.
        self.draw_method = params.get('draw_method',
                                      base.get('stamp',{}).get('draw_method','auto'))

        pointing = CelestialCoord(ra=params['ra'], dec=params['dec'])
        wcs = roman.getWCS(world_pos        = pointing,
                                PA          = params['pa']*galsim.degrees,
                                date        = params['date'],
                                SCAs        = self.sca,
                                PA_is_FPA   = True
                                )[self.sca]

        # GalSim expects a wcs in the image field.
        config['wcs'] = wcs

        # If user hasn't overridden the bandpass to use, get the standard one.
        if 'bandpass' not in config:
            base['bandpass'] = self.getBandpass(self.filter)

        return n_pix, n_pix

    def getBandpass(self, filter_name):
        if not hasattr(self, 'all_roman_bp'):
            self.all_roman_bp = roman.getBandpasses()
        return self.all_roman_bp[filter_name]

    def addNoise(self, image, config, base, image_num, obj_num, current_var, logger):
        """Add the final noise to a Scattered image

        Parameters:
            image:          The image onto which to add the noise.
            config:         The configuration dict for the image field.
            base:           The base configuration dict.
            image_num:      The current image number.
            obj_num:        The first object number in the image.
            current_var:    The current noise variance in each postage stamps.
            logger:         If given, a logger object to log progress.
        """
        base['current_noise_image'] = base['current_image']
        wcs = base['wcs']
        bp = base['bandpass']
        rng = galsim.config.GetRNG(config, base)
        logger.info('image %d: Start RomanSCA detector effects',base.get('image_num',0))

        # Things that will eventually be subtracted (if sky_subtract) will have their expectation
        # value added to sky_image.  So technically, this includes things that aren't just sky.
        # E.g. includes dark_current and thermal backgrounds.
        sky_image = image.copy()
        sky_level = roman.getSkyLevel(bp, world_pos=wcs.toWorld(image.true_center))
        logger.debug('Adding sky_level = %s',sky_level)
        if self.stray_light:
            logger.debug('Stray light fraction = %s',stray_light_fraction)
            sky_level *= (1.0 + stray_light_fraction)
        wcs.makeSkyImage(sky_image, sky_level)

        # The other background is the expected thermal backgrounds in this band.
        # These are provided in e-/pix/s, so we have to multiply by the exposure time.
        if self.thermal_background:
            tb = roman.thermal_backgrounds[self.filter] * self.exptime
            logger.debug('Adding thermal background: %s',tb)
            sky_image += roman.thermal_backgrounds[self.filter] * self.exptime

        # The image up to here is an expectation value.
        # Realize it as an integer number of photons.
        poisson_noise = galsim.noise.PoissonNoise(rng)
        if self.draw_method == 'phot':
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

        if self.reciprocity_failure:
            logger.debug("Applying reciprocity failure")
            roman.addReciprocityFailure(image)

        if self.dark_current:
            dc = dark_current * self.exptime
            logger.debug("Adding dark current: %s",dc)
            sky_image += dc
            dark_noise = galsim.noise.DeviateNoise(galsim.random.PoissonDeviate(rng, dc))
            image.addNoise(dark_noise)

        if self.nonlinearity:
            logger.debug("Applying classical nonlinearity")
            roman.applyNonlinearity(image)

        # Mosby and Rauscher say there are two read noises.  One happens before IPC, the other
        # one after.
        # TODO: Add read_noise1
        if self.ipc:
            logger.debug("Applying IPC")
            roman.applyIPC(image)

        if self.read_noise:
            logger.debug("Adding read noise %s",read_noise)
            image.addNoise(GaussianNoise(rng, sigma=read_noise))

        logger.debug("Applying gain %s",gain)
        image /= gain

        # Make integer ADU now.
        image.quantize()

        if self.sky_subtract:
            logger.debug("Subtracting sky image")
            sky_image /= gain
            sky_image.quantize()
            image -= sky_image

# Register this as a valid type
RegisterImageType('roman_sca', RomanSCAImageBuilder())