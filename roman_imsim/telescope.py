# from __future__ import division
# from __future__ import print_function

# from future import standard_library
# standard_library.install_aliases()
# from builtins import str
# from builtins import range
# from past.builtins import basestring
# from builtins import object
# from past.utils import old_div

import numpy as np
import healpy as hp
import sys, os, io
import math
import copy
import logging
import time
import yaml
import copy
import galsim as galsim
import galsim.roman as roman
import galsim.config.process as process
import galsim.des as des
# import ngmix
import fitsio as fio
import pickle as pickle
import pickletools
from astropy.time import Time
from mpi4py import MPI
# from mpi_pool import MPIPool
import cProfile, pstats, psutil
import glob
import shutil
import h5py

import roman_imsim

from .misc import ParamError
from .misc import except_func
from .misc import save_obj
from .misc import load_obj
from .misc import convert_dither_to_fits
from .misc import convert_gaia
from .misc import convert_galaxia
from .misc import create_radec_fits
from .misc import hsm
from .misc import get_filename
from .misc import get_filenames
from .misc import write_fits

# Chip coordinates
cptr = np.array([
0.002689724,  1.000000000,  0.181995021, -0.002070809, -1.000000000,  0.807383134,  1.000000000,  0.004769437,  1.028725015, -1.000000000, -0.000114163, -0.024579913,
0.003307633,  1.000000000,  1.203503349, -0.002719257, -1.000000000, -0.230036847,  1.000000000,  0.006091805,  1.028993582, -1.000000000, -0.000145757, -0.024586416,
0.003888409,  1.000000000,  2.205056241, -0.003335597, -1.000000000, -1.250685466,  1.000000000,  0.007389324,  1.030581048, -1.000000000, -0.000176732, -0.024624426,
0.007871078,  1.000000000, -0.101157485, -0.005906926, -1.000000000,  1.095802866,  1.000000000,  0.009147586,  2.151242511, -1.000000000, -0.004917673, -1.151541644,
0.009838715,  1.000000000,  0.926774753, -0.007965112, -1.000000000,  0.052835488,  1.000000000,  0.011913584,  2.150981875, -1.000000000, -0.006404157, -1.151413352,
0.011694346,  1.000000000,  1.935534773, -0.009927853, -1.000000000, -0.974276664,  1.000000000,  0.014630945,  2.153506744, -1.000000000, -0.007864196, -1.152784334,
0.011758070,  1.000000000, -0.527032681, -0.008410887, -1.000000000,  1.529873670,  1.000000000,  0.012002262,  3.264990040, -1.000000000, -0.008419930, -2.274065453,
0.015128555,  1.000000000,  0.510881058, -0.011918799, -1.000000000,  0.478274989,  1.000000000,  0.016194244,  3.262719942, -1.000000000, -0.011359106, -2.272508364,
0.018323436,  1.000000000,  1.530828790, -0.015281655, -1.000000000, -0.558879607,  1.000000000,  0.020320244,  3.264721809, -1.000000000, -0.014251259, -2.273955111,
-0.002689724,  1.000000000,  0.181995021,  0.002070809, -1.000000000,  0.807383134,  1.000000000, -0.000114163, -0.024579913, -1.000000000,  0.004769437,  1.028725015,
-0.003307633,  1.000000000,  1.203503349,  0.002719257, -1.000000000, -0.230036847,  1.000000000, -0.000145757, -0.024586416, -1.000000000,  0.006091805,  1.028993582,
-0.003888409,  1.000000000,  2.205056241,  0.003335597, -1.000000000, -1.250685466,  1.000000000, -0.000176732, -0.024624426, -1.000000000,  0.007389324,  1.030581048,
-0.007871078,  1.000000000, -0.101157485,  0.005906926, -1.000000000,  1.095802866,  1.000000000, -0.004917673, -1.151541644, -1.000000000,  0.009147586,  2.151242511,
-0.009838715,  1.000000000,  0.926774753,  0.007965112, -1.000000000,  0.052835488,  1.000000000, -0.006404157, -1.151413352, -1.000000000,  0.011913584,  2.150981875,
-0.011694346,  1.000000000,  1.935534773,  0.009927853, -1.000000000, -0.974276664,  1.000000000, -0.007864196, -1.152784334, -1.000000000,  0.014630945,  2.153506744,
-0.011758070,  1.000000000, -0.527032681,  0.008410887, -1.000000000,  1.529873670,  1.000000000, -0.008419930, -2.274065453, -1.000000000,  0.012002262,  3.264990040,
-0.015128555,  1.000000000,  0.510881058,  0.011918799, -1.000000000,  0.478274989,  1.000000000, -0.011359106, -2.272508364, -1.000000000,  0.016194244,  3.262719942,
-0.018323436,  1.000000000,  1.530828790,  0.015281655, -1.000000000, -0.558879607,  1.000000000, -0.014251259, -2.273955111, -1.000000000,  0.020320244,  3.264721809 ])

# SCAs' central coordinates
sca_center = np.array([
    [21.94, 13.12],
    [-22.09, -31,77],
    [-22.24, -81.15],
    [-65.82, 23.76],
    [-66.32, -20.77],
    [-66.82, -70.15],
    [-109.70, 44.12],
    [-110.46, 0.24],
    [-111.56, -49.15],
    [21.94, 13.12],
    [22.09, -31.77],
    [22.24, -81.15],
    [65.82, 23.76],
    [66.32, -20.77],
    [66.82, -70.15],
    [109.70, 44.12],
    [110.46, 0.24],
    [111.56, -49.15]])

class pointing(object):
    """
    Class to manage and hold informaiton about a roman pointing, including WCS and PSF.
    """

    def __init__(self, params, logger, filter_=None, sca=None, dither=None, sca_pos=None, max_rad_from_boresight=0.009,rank=None):
        """
        Initializes some information about a pointing.

        Input
        params                  : Parameter dict.
        logger                  : logger instance
        filter_                 : The filter name for this pointing.
        sca                     : The SCA number (1-18)
        dither                  : The index of this pointing in the survey simulation file.
        sca_pos                 : Used to simulate the PSF at a position other than the center
                                    of the SCA.
        max_rad_from_boresight  : Distance around pointing to attempt to simulate objects.
        chip_enlarge            : Factor to enlarge chip geometry by to account for small
                                    inaccuracies relative to precise WCS.
        """

        self.params             = params
        self.ditherfile         = params['dither_file']
        self.n_waves            = params['n_waves'] # Number of wavelenghts of PSF to simulate
        self.approximate_struts = params['approximate_struts'] # Whether to approsimate struts
        self.extra_aberrations  = params['extra_aberrations']  # Extra aberrations to include in the PSF model. See galsim documentation.

        self.logger = logger
        self.rank   = rank
        self.sca    = None
        self.PSF    = None
        self.WCS    = None
        self.dither = None
        self.filter = None
        self.los_motion = None

        if filter_ is not None:
            self.get_bpass(filter_)

        if sca is not None:
            self.update_sca(sca,sca_pos=sca_pos)

        if dither is not None:
            self.update_dither(dither)

        self.bore           = max_rad_from_boresight
        self.sbore2         = np.sin(max_rad_from_boresight/2.)
        self.chip_enlarge   = params['chip_enlarge']

    def get_bpass(self, filter_):
        """
        Read in the Roman filters, setting an AB zeropoint appropriate for this telescope given its
        diameter and (since we didn't use any keyword arguments to modify this) using the typical
        exposure time for Roman images.  By default, this routine truncates the parts of the
        bandpasses that are near 0 at the edges, and thins them by the default amount.

        Input
        filter_ : Fiter name for this pointing.
        """

        self.filter = filter_
        self.bpass  = roman.getBandpasses(AB_zeropoint=True)[self.filter]

    def update_dither(self,dither,force_filter=False):
        """
        This updates the pointing to a new dither position.

        Input
        dither     : Pointing index in the survey simulation file.
        sca        : SCA number
        """

        self.dither = dither

        d = fio.FITS(self.ditherfile)[-1][self.dither]

        # Check that nothing went wrong with the filter specification.
        # if filter_dither_dict[self.filter] != d['filter']:
        #     raise ParamError('Requested filter and dither pointing do not match.')

        self.ra     = d['ra']  * np.pi / 180. # RA of pointing
        self.dec    = d['dec'] * np.pi / 180. # Dec of pointing
        self.pa     = d['pa']  * np.pi / 180.  # Position angle of pointing
        self.sdec   = np.sin(self.dec) # Here and below - cache some geometry stuff
        self.cdec   = np.cos(self.dec)
        self.sra    = np.sin(self.ra)
        self.cra    = np.cos(self.ra)
        self.spa    = np.sin(self.pa)
        self.cpa    = np.cos(self.pa)
        self.date   = Time(d['date'],format='mjd').datetime # Date of pointing
        self.mjd    = d['date']


        if (self.filter is None) or force_filter:
            self.get_bpass(roman_imsim.filter_dither_dict_[d['filter']])

    def update_sca(self,sca,psf=True):
        """
        This assigns an SCA to the pointing, and evaluates the PSF and WCS.

        Input
        dither     : Pointing index in the survey simulation file.
        sca        : SCA number
        """

        self.sca    = sca
        self.get_wcs() # Get the new WCS
        if psf:
            self.get_psf() # Get the new PSF
        radec           = self.WCS.toWorld(galsim.PositionI(int(roman.n_pix/2),int(roman.n_pix/2)))
        print('SCA is at position ',radec.ra,galsim.degrees,radec.dec,galsim.degrees)
        self.sca_sdec   = np.sin(radec.dec) # Here and below - cache some geometry  stuff
        self.sca_cdec   = np.cos(radec.dec)
        self.sca_sra    = np.sin(radec.ra)
        self.sca_cra    = np.cos(radec.ra)
        self.radec      = radec

    def get_psf(self, sca_pos=None, high_accuracy=False):
        """
        This updates the pointing to a new SCA, replacing the stored PSF to the new SCA.

        Input
        sca_pos : Used to simulate the PSF at a position other than the center of the SCA.
        """

        # Add extra aberrations that vary sca-to-sca across the focal plane
        extra_aberrations = None
        # gradient across focal plane
        if 'gradient_aberration' in self.params:
            if self.params['gradient_aberration']:
                extra_aberrations = sca_center[self.sca-1][1]*np.array(self.extra_aberrations)*np.sqrt(3.)/88.115
        # random assignment chip-to-chip
        if 'random_aberration' in self.params:
            if self.params['random_aberration']:
                np.random.seed(self.sca)
                extra_aberrations = np.array(self.extra_aberrations)*np.random.rand()

        # Time-dependent oscillation of the aberrations
        # Unlike others, must pass unit self.extra_aberrations array
        if 'oscillating_aberration' in self.params:
            if self.params['oscillating_aberration']:
                extra_aberrations = np.array(self.extra_aberrations)*self.time_aberration()

        # Define a high-frequency smearing to convolve the PSF by later
        if 'los_motion' in self.params:
            # symmetric smearing
            if self.params['los_motion'] is not None:
                self.los_motion = galsim.Gaussian(fwhm=2.*np.sqrt(2.*np.log(2.))*self.params['los_motion'])
            # assymetric smearing
            if ('los_motion_e1' in self.params) and ('los_motion_e2' in self.params):
                if (self.params['los_motion_e1'] is not None) and (self.params['los_motion_e2'] is not None):
                    self.los_motion = self.los_motion.shear(g1=self.params['los_motion_e1'],g2=self.params['los_motion_e2']) # assymetric jitter noise
                if (self.params['los_motion_e1'] is None) or (self.params['los_motion_e2'] is None):
                    raise ParamError('Must provide both los motion e1 and e2.')

        # assymetric smearing on random subset of pointings
        if 'random_los_motion' in self.params:
            if self.params['random_los_motion']:
                np.random.seed(self.dither)
                if np.random.rand()>0.15:
                    self.los_motion = None

        # aberration gradient across chip
        if 'random_aberration_gradient' in self.params:
            if self.params['random_aberration_gradient']:
                np.random.seed(self.sca)
                extra_aberrations = np.array(self.extra_aberrations)*np.random.rand()*np.sqrt(3.)/(roman.n_pix/2.)
        else:
            self.params['random_aberration_gradient'] = False

        # No special changes, populate from yaml file
        if extra_aberrations is None:
            extra_aberrations = self.extra_aberrations

        if self.params['random_aberration_gradient']:

            self.PSF = None
            self.extra_aberrations = extra_aberrations

        elif 'gauss_psf' in self.params:
            if self.params['gauss_psf'] is not None:
                self.PSF = galsim.Gaussian(half_light_radius=self.params['gauss_psf'])
        else:

            # print(self.sca,self.filter,sca_pos,self.bpass.effective_wavelength)
            self.PSF = {}
            self.PSF[8] = roman.getPSF(self.sca,
                                    self.filter,
                                    SCA_pos             = sca_pos,
                                    wcs=self.WCS,
                                    pupil_bin = 8,
                                    n_waves             = self.n_waves,
                                    logger              = self.logger,
                                    wavelength          = self.bpass.effective_wavelength,
                                    extra_aberrations   = extra_aberrations
                                    )
            self.PSF[4] = roman.getPSF(self.sca,
                                    self.filter,
                                    SCA_pos             = sca_pos,
                                    wcs=self.WCS,
                                    pupil_bin = 4,
                                    n_waves             = self.n_waves,
                                    logger              = self.logger,
                                    wavelength          = self.bpass.effective_wavelength,
                                    extra_aberrations   = extra_aberrations
                                    )
            self.PSF[2] = roman.getPSF(self.sca,
                                    self.filter,
                                    SCA_pos             = sca_pos,
                                    wcs=self.WCS,
                                    pupil_bin = 2,
                                    n_waves             = self.n_waves,
                                    logger              = self.logger,
                                    wavelength          = self.bpass.effective_wavelength,
                                    extra_aberrations   = extra_aberrations
                                    )

        # sim.logger.info('Done PSF precomputation in %.1f seconds!'%(time.time()-t0))

    def load_psf(self,pos,pupil_bin=8,sca_pos=None, high_accuracy=False):
        """
        Interface to access self.PSF.

        pos : GalSim PositionI
        """
        if self.params['random_aberration_gradient']:
            print('-------------------')

            np.random.seed(self.sca)
            if np.random.rand()>0.5:
                i = pos.x
            else:
                i = pos.y
            return roman.getPSF(self.sca,
                                self.filter,
                                SCA_pos             = sca_pos,
                                wcs=self.WCS,
                                pupil_bin=8,
                                n_waves             = self.n_waves,
                                logger              = self.logger,
                                wavelength          = self.bpass.effective_wavelength,
                                extra_aberrations   = self.extra_aberrations*(i-roman.n_pix/2.+0.5)
                                )

        else:

            return self.PSF[pupil_bin]

        return

    def time_aberration(self):
        """
        A time-varying aberration. Returns a function of the datetime of pointing to modulate the extra_aberrations.
        """

        delta_t = 60. # s
        fid_wavelength=1293. # nm

        total_T = 5*365*24*60*60 # mission time [s]

        with open('time_aberration.pickle', 'rb') as file:
            ft=pickle.load(file,encoding='bytes')

        t=np.linspace(0,total_T,num=len(ft))
        ft_interp=interp1d(t,ft)

        date = fio.FITS(self.ditherfile)[-1].read()['date']
        mission_start_time = Time(min(date),format='mjd')# format='mjd'

        dither_time = Time(date[self.dither],format='mjd')
        dt = (dither_time-mission_start_time).sec/delta_t

        time_aberration = ft_interp(dt)/fid_wavelength

        return time_aberration

    def get_wcs(self):
        """
        Get the WCS for an observation at this position. We are not supplying a date, so the routine will assume it's the vernal equinox. The output of this routine is a dict of WCS objects, one for each SCA. We then take the WCS for the SCA that we are using.
        """
        self.WCS = roman.getWCS(world_pos  = galsim.CelestialCoord(ra=self.ra*galsim.radians, \
                                                                    dec=self.dec*galsim.radians),
                                PA          = self.pa*galsim.radians,
                                date        = self.date,
                                SCAs        = self.sca,
                                PA_is_FPA   = True
                                )[self.sca]

    def in_sca(self, ra, dec):
        """
        Check if ra, dec falls on approximate SCA area.

        Input
        ra  : Right ascension of object
        dec : Declination of object
        """

        # Catch some problems, like the pointing not being defined
        if self.dither is None:
            raise ParamError('No dither defined to check ra, dec against.')

        if self.sca is None:
            raise ParamError('No sca defined to check ra, dec against.')

        # # Discard any object greater than some dec from pointing
        # if np.abs(dec-self.dec)>self.bore:
        #     return False

        # Position of the object in boresight coordinates
        mX  = -self.sdec   * np.cos(dec) * np.cos(self.ra-ra) + self.cdec * np.sin(dec)
        mY  =  np.cos(dec) * np.sin(self.ra-ra)

        xi  = -(self.spa * mX + self.cpa * mY) / 0.0021801102 # Image plane position in chips
        yi  = (self.cpa * mX - self.spa * mY) / 0.0021801102

        # Check if object falls on SCA
        if hasattr(ra,'__len__'):
            return   np.where((cptr[0+12*(self.sca-1)]*xi+cptr[1+12*(self.sca-1)]*yi  \
                                <cptr[2+12*(self.sca-1)]+self.chip_enlarge)       \
                            & (cptr[3+12*(self.sca-1)]*xi+cptr[4+12*(self.sca-1)]*yi  \
                                <cptr[5+12*(self.sca-1)]+self.chip_enlarge)       \
                            & (cptr[6+12*(self.sca-1)]*xi+cptr[7+12*(self.sca-1)]*yi  \
                                <cptr[8+12*(self.sca-1)]+self.chip_enlarge)       \
                            & (cptr[9+12*(self.sca-1)]*xi+cptr[10+12*(self.sca-1)]*yi \
                                <cptr[11+12*(self.sca-1)]+self.chip_enlarge))[0]

        if    (cptr[0+12*(self.sca-1)]*xi+cptr[1+12*(self.sca-1)]*yi  \
                <cptr[2+12*(self.sca-1)]+self.chip_enlarge)       \
            & (cptr[3+12*(self.sca-1)]*xi+cptr[4+12*(self.sca-1)]*yi  \
                <cptr[5+12*(self.sca-1)]+self.chip_enlarge)       \
            & (cptr[6+12*(self.sca-1)]*xi+cptr[7+12*(self.sca-1)]*yi  \
                <cptr[8+12*(self.sca-1)]+self.chip_enlarge)       \
            & (cptr[9+12*(self.sca-1)]*xi+cptr[10+12*(self.sca-1)]*yi \
                <cptr[11+12*(self.sca-1)]+self.chip_enlarge):

            return True

        return False

    def near_pointing(self, ra, dec, min_date=None, max_date=None, sca=False):
        """
        Returns objects close to pointing, using usual orthodromic distance.

        Input
        ra  : Right ascension array of objects
        dec : Declination array of objects
        min_date, max_date : Optional date range for transients
        """

        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)

        if sca:
            d2 = (x - self.sca_cdec*self.sca_cra)**2 + (y - self.sca_cdec*self.sca_sra)**2 + (z - self.sca_sdec)**2
        else:
            d2 = (x - self.cdec*self.cra)**2 + (y - self.cdec*self.sra)**2 + (z - self.sdec)**2

        if min_date is None:
            return np.where(np.sqrt(d2)/2.<=self.sbore2)[0]
        else:
            return np.where((np.sqrt(d2)/2.<=self.sbore2) & (min_date<=self.mjd) & (self.mjd<=max_date))[0]
