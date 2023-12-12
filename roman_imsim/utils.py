import numpy as np
import galsim
import galsim.config
import galsim.roman as roman

class roman_utils(object):
    """
    Class to contain a variety of helper routines to work with the simulation data.
    """
    def __init__(self, config_file,visit=None,sca=None,image_name=None,setup_skycat=False):
        """
        Setup information about a simulated Roman image.
        Parameters:
            config_file: the GalSim config file that produced the simulation
            visit: the visit (observation sequence) number of the pointing 
            sca: the SCA number
            image_name: the filename of the image (can be used instead of visit, sca)
            setup_skycat: setup the skycatalog information to have access to
        """
        config = galsim.config.ReadConfig(config_file)[0]

        self.visit,self.sca = self.check_input(visit,sca,image_name)

        if not setup_skycat:
            del config['input']['sky_catalog']
        config['input']['obseq_data']['visit'] = self.visit
        config['image']['SCA'] = self.sca
        galsim.config.ProcessInput(config)
        if setup_skycat:
            self.skycat = galsim.config.GetInputObj('sky_catalog',config['input']['sky_catalog'],config,'sky_catalog')
        self.PSF        = galsim.config.GetInputObj('roman_psf',config['input']['roman_psf'],config,'roman_psf')
        self.wcs        = galsim.config.BuildWCS(config['image'], 'wcs', config)
        self.bpass      = galsim.config.BuildBandpass(config['image'], 'bandpass', config, None)[0]

    def check_input(self,visit,sca,image_name):
        if image_name is not None:
            print('Inferring visit and sca from image_name.')
            start = 21
            end = -5
            if 'simple_model' in image_name:
                start = 28
            if 'gz' in image_name:
                end = -8
            tmp = np.array(image_name[start:end].split('_')).astype(int)
            return tmp[0],tmp[1]
        if (visit is None) | (sca is None):
            raise ValueError('Insufficient information to construct visit info - all inputs are None.')
        return visit,sca

    def getPSF(self,x=None,y=None,pupil_bin=8):
        """
        Return Roman PSF for image position
        Parameters:
            x: x-position in SCA
            y: y-position in SCA
            pupil_bin: pupil image binning factor
        Returns:
            the chromatic GalSim PSF model object
        """
        if pupil_bin!=8:
            if (x is not None)|(y is not None):
                raise ValueError('x,y position for pupil_bin values other than 8 not supported. Using SCA center.')
            return self.PSF.getPSF(pupil_bin,galsim.PositionD(roman.n_pix/2,roman.n_pix/2))
        if (x is None) | (y is None):
            return self.PSF.getPSF(8,galsim.PositionD(roman.n_pix/2,roman.n_pix/2))
        return self.PSF.getPSF(8,galsim.PositionD(x,y))

    def getWCS(self):
        """
        Return Roman WCS for image
        """
        return self.wcs

    def getBandpass(self):
        """
        Return Roman bandpass for image
        """
        return self.bpass

    def getPSF_Image(self,stamp_size,x=None,y=None,pupil_bin=8,sed=None):
        """
        Return a Roman PSF image for some image position
        Parameters:
            stamp_size: size of output PSF model stamp
            x: x-position in SCA
            y: y-position in SCA
            pupil_bin: pupil image binning factor
            sed: SED to be used to draw the PSF - default is a flat SED.
        Returns:
            the PSF GalSim image object (use image.array to get a numpy array representation)
        """
        if sed is None:
            sed = galsim.SED(galsim.LookupTable([100, 2600], [1,1], interpolant='linear'),
                              wave_type='nm', flux_type='fphotons')
        point = galsim.DeltaFunction()*sed
        point = point.withFlux(1,self.bpass)
        psf = galsim.Convolve(point, self.getPSF(x,y,pupil_bin))
        stamp = galsim.Image(stamp_size,stamp_size,wcs=self.wcs)
        return psf.drawImage(self.bpass,image=stamp,wcs=self.wcs,method='no_pixel')

