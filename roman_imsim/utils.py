import numpy as np
import galsim.config

class roman_utils(object):
    """
    Class to contain a variety of helper routines to work with the simulation data.
    """
    def __init__(self, config_file,visit=None,sca=None,image_name=None,setup_skycat=False):
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
        if pupil_bin!=8:
            if (x is not None)|(y is not None):
                raise ValueError('x,y position for pupil_bin values other than 8 not supported. Using SCA center.')
            return self.PSF_.getPSF(pupil_bin,galsim.PositionD(roman.n_pix/2,roman.n_pix/2))
        if (x is None) | (y is None):
            return self.PSF_.getPSF(8,galsim.PositionD(roman.n_pix/2,roman.n_pix/2))
        return self.PSF_.getPSF(8,galsim.PositionD(x,y))

    def getWCS(self):
        return self.WCS

    def getBandpass(self):
        return self.bpass

    def getPSF_Image(self,stamp_size,x=None,y=None,pupil_bin=8):
        psf = galsim.Convolve(galsim.DeltaFunction(), self.getPSF(x,y,pupil_bin))
        stamp = galsim.Image(stamp_size,stamp_size,wcs=self.WCS)
        return psf.drawImage(self.bpass,image=stamp,wcs=self.WCS,method='no_pixel')

