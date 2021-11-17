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
#from mpi4py import MPI
# from mpi_pool import MPIPool
import cProfile, pstats, psutil
import glob
import shutil
import h5py

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
from .telescope import pointing as Pointing

class modify_image(object):
    """
    Class to simulate non-idealities and noise of roman detector images.
    """

    def __init__(self,params):
        """
        Set up noise properties of image

        Input
        params  : parameter dict
        rng     : Random generator
        """

        roman.exptime  = 139.8
        self.params    = params
        ## create fitsio instance
        self.df = fio.FITS(self.params['detector_file'])


    def add_effects(self,im,wt,pointing):
        """
        Add detector effects for Roman.

        Input:
        im        : Postage stamp or image.
        pointing  : Pointing object
        radec     : World coordinate position of image
        local_wcs : The local WCS
        phot      : photon shooting mode

        Preserve order:
        1) qe
        2) brighter-fatter
        3) persistence
        4) quantize
        5) dark current
        6) saturation
        7) CNL
        8) IPC
        9) dead pixel mask
        10) vertical trailing pixel effect
        11) read noise (e-)
        12) gain (in unit of e/adu)
        13) bias
        14) quantize

        Take 4088x4088 sky image as input
        Pad the image to 4096x4096
        Output 4088x4088 images in uint16



        """
        # Option to change exposure time (in seconds)
        if 'exposure_time' in self.params:
            if self.params['exposure_time'] == 'deep':
                if pointing.filter[0] == 'Y':
                    roman.exptime = 300.0
                if pointing.filter[0] == 'J':
                    roman.exptime = 300.0
                if pointing.filter[0] == 'H':
                    roman.exptime = 300.0
                if pointing.filter[0] == 'F':
                    roman.exptime = 900.0
        print(pointing.filter)

        ## check input dimension
        if not im.array.shape==(4088,4088):
            raise ValueError("input image for detector effects must be 4088x4088.")


        im = self.add_background(im) # Add background to image and save background



        ## create padded image
        bound_pad = galsim.BoundsI( xmin=1, ymin=1,
                                    xmax=4096, ymax=4096)
        im_pad = galsim.Image(bound_pad)
        im_pad.array[4:-4, 4:-4] = im.array[:,:]

        self.set_diff(im_pad)

        im_pad = self.qe(im_pad)
        self.diff('qe', im_pad)

        im_pad = self.bfe(im_pad)
        self.diff('bfe', im_pad)

        im_pad = self.add_persistence(im_pad, pointing)
        self.diff('pers', im_pad)

        im_pad.quantize()
        self.diff('quantize1', im_pad)

        im_pad = self.dark_current(im_pad)
        self.diff('dark', im_pad)

        im_pad = self.saturate(im_pad)
        self.diff('sat', im_pad)

        im_pad = self.nonlinearity(im_pad)
        self.diff('cnl', im_pad)

        im_pad = self.interpix_cap(im_pad)
        self.diff('ipc', im_pad)

        im_pad = self.deadpix(im_pad)
        self.diff('deadpix', im_pad)

        im_pad = self.vtpe(im_pad)
        self.diff('vtpe', im_pad)

        im_pad = self.add_read_noise(im_pad)
        self.diff('read', im_pad)

        im_pad = self.add_gain(im_pad)
        self.diff('gain', im_pad)

        im_pad = self.add_bias(im_pad)
        self.diff('bias', im_pad)

        im_pad.quantize()
        self.diff('quantize2', im_pad)

        # output 4088x4088 img in uint16
        im.array[:,:] = im_pad.array[4:-4, 4:-4]
        im = galsim.Image(im, dtype=np.uint16)

        # data quality image
        # 0x1 -> non-responsive
        # 0x2 -> hot pixel
        # 0x4 -> very hot pixel
        # 0x8 -> adjacent to pixel with strange response
        # 0x10 -> low CDS, high total noise pixel (may have strange settling behaviors, not recommended for precision applications)
        # 0x20 -> CNL fit went down to the minimum number of points (remaining degrees of freedom = 0)
        # 0x40 -> no solid-waffle solution for this region (set gain value to array median). normally occurs in a few small regions of some SCAs with lots of bad pixels. [recommend not to use these regions for WL analysis]
        # 0x80 -> wt==0
        dq = self.df['BADPIX'][4:4092, 4:4092]
        # get weight map
        if not self.params['use_background']:
            return im,None

        if wt is not None:
           dq[wt==0] += 128

        return im, self.sky[self.sky.bounds&im.bounds]-self.sky_mean, dq, self.sky_mean

    def set_diff(self, im=None):
        self.t0 = time.time()
        self.t1 = time.time()

        if self.params['save_diff']:
            self.pre = im.copy()
            self.pre.write('bg.fits', dir=self.params['diff_dir'])
        return self.t0, self.t1

    def diff(self, msg, im=None, verbose=True):
        self.t1 = time.time()
        dt = self.t1-self.t0
        self.t0 = time.time()

        if self.params['save_diff']:
            diff = im-self.pre
            diff.write('%s_diff.fits'%msg , dir=self.params['diff_dir'])
            self.pre = im.copy()
            im.write('%s_cumul.fits'%msg, dir=self.params['diff_dir'])

        if verbose:
            print('=======  %s   dt = %.2f s    ======'%(msg,dt))
        return dt


    def qe(self, im):
        """
        Apply the wavelength-independent relative QE to the image.

        Input
        im                  : Image
        RELQE1[4096,4096]   : relative QE map
        """

        # If effect is turned off, return image unchanged
        if not self.params['use_qe']:
            return im

        im.array[:,:] *= self.df['RELQE1'][:,:] #4096x4096 array
        return im


    def bfe(self, im):
        """
        Apply brighter-fatter effect.
        Brighter fatter effect is a non-linear effect that deflects photons due to the
        the eletric field built by the accumulated charges. This effect exists in both
        CCD and CMOS detectors and typically percent level change in charge.
        The built-in electric field by the charges in pixels tends to repulse charges
        to nearby pixels. Thus, the profile of more illuminous ojbect becomes broader.
        This effect can also be understood effectly as change in pixel area and pixel
        boundaries.
        BFE is defined in terms of the Antilogus coefficient kernel of total pixel area change
        in the detector effect charaterization file. Kernel of the total pixel area, however,
        is not sufficient. Image simulation of the brighter fatter effect requires the shift
        of the four pixel boundaries. Before we get better data, we solve for the boundary
        shift components from the kernel of total pixel area by assumming several symmetric constraints.

        Input
        im                                      : Image
        BFE[nbfe+Delta y, nbfe+Delta x, y, x]   : bfe coefficient kernel, nbfe=2
        """

        # If effect is turned off, return image unchanged
        if not self.params['use_bfe']:
            return im

        nbfe = 2 ## kernel of bfe in shape (2 x nbfe+1)*(2 x nbfe+1)
        bin_size = 128
        n_max = 32
        m_max = 32
        num_grids = 4
        n_sub = n_max//num_grids
        m_sub = m_max//num_grids

        ##=======================================================================
        ##     solve boundary shfit kernel aX components
        ##=======================================================================
        a_area = self.df['BFE'][:,:,:,:] #5x5x32x32
        a_components = np.zeros( (4, 2*nbfe+1, 2*nbfe+1, n_max, m_max) ) #4x5x5x32x32

        ##solve aR aT aL aB for each a
        for n in range(n_max): #m_max and n_max = 32 (binned in 128x128)
            for m in range(m_max):
                a = a_area[:,:, n, m] ## a in (2 x nbfe+1)*(2 x nbfe+1)

                ## assume two parity symmetries
                a = ( a + np.fliplr(a) + np.flipud(a) + np.flip(a)  )/4.

                r = 0.5* ( 3.25/4.25  )**(1.5) / 1.5   ## source-boundary projection
                B = (a[2,2], a[3,2], a[2,3], a[3,3],
                     a[4,2], a[2,4], a[3,4], a[4,4] )

                A = np.array( [ [ -2 , -2 ,  0 ,  0 ,  0 ,  0 ,  0 ],
                                [  0 ,  1 ,  0 , -1 , -2 ,  0 ,  0 ],
                                [  1 ,  0 , -1 ,  0 , -2 ,  0 ,  0 ],
                                [  0 ,  0 ,  0 ,  0 ,  2 , -2 ,  0 ],
                                [  0 ,  0 ,  0 ,  1 ,  0 ,-2*r,  0 ],
                                [  0 ,  0 ,  1 ,  0 ,  0 ,-2*r,  0 ],
                                [  0 ,  0 ,  0 ,  0 ,  0 , 1+r, -1 ],
                                [  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  2 ]  ])


                s1,s2,s3,s4,s5,s6,s7 = np.linalg.lstsq(A, B, rcond=None)[0]

                aR = np.array( [[ 0.   , -s7  ,-r*s6 , r*s6 ,  s7  ],
                                [ 0.   , -s6  , -s5  ,  s5  ,  s6  ],
                                [ 0.   , -s3  , -s1  ,  s1  ,  s3  ],
                                [ 0.   , -s6  , -s5  ,  s5  ,  s6  ],
                                [ 0.   , -s7  ,-r*s6 , r*s6 ,  s7  ],])


                aT = np.array( [[   0.  ,  0. ,  0.  ,   0. ,   0.   ],
                                [  -s7  , -s6 , -s4  , -s6  ,  -s7   ],
                                [ -r*s6 , -s5 , -s2  , -s5  , -r*s6  ],
                                [  r*s6 ,  s5 ,  s2  ,  s5  ,  r*s6  ],
                                [   s7  ,  s6 ,  s4  ,  s6  ,   s7   ],])


                aL = aR[::-1, ::-1]
                aB = aT[::-1, ::-1]




                a_components[0, :,:, n, m] = aR[:,:]
                a_components[1, :,:, n, m] = aT[:,:]
                a_components[2, :,:, n, m] = aL[:,:]
                a_components[3, :,:, n, m] = aB[:,:]

        ##=============================
        ## Apply bfe to image
        ##=============================

        ## pad and expand kernels
        ## The img is clipped by the saturation level here to cap the brighter fatter effect and avoid unphysical behavior

        array_pad = self.saturate(im.copy()).array[4:-4,4:-4] # img of interest 4088x4088
        array_pad = np.pad(array_pad, [(4+nbfe,4+nbfe),(4+nbfe,4+nbfe)], mode='symmetric') #4100x4100 array


        dQ_components = np.zeros( (4, bin_size*n_max, bin_size*m_max) )   #(4, 4096, 4096) in order of [aR, aT, aL, aB]


        ### run in sub grids to reduce memory

        ## pad and expand kernels
        t = np.zeros((bin_size*n_sub, n_sub))
        for row in range(t.shape[0]):
            t[row, row//(bin_size) ] =1



        for gj in range(num_grids):
            for gi in range(num_grids):

                a_components_pad = np.zeros( (4, 2*nbfe+1, 2*nbfe+1, bin_size*n_sub+2*nbfe, bin_size*m_sub+2*nbfe)  ) #(4,5,5,sub_grid,sub_grid)


                for comp in range(4):
                    for j in range(2*nbfe+1):
                        for i in range(2*nbfe+1):
                            tmp = (t.dot(  a_components[comp,j,i,gj*n_sub:(gj+1)*n_sub,gi*m_sub:(gi+1)*m_sub]  ) ).dot(t.T) #sub_grid*sub_grid
                            a_components_pad[comp, j, i, :, :] = np.pad(tmp, [(nbfe,nbfe),(nbfe,nbfe)], mode='symmetric')

                #convolve aX_ij with Q_ij
                for comp in range(4):
                    for dy in range(-nbfe, nbfe+1):
                        for dx in range(-nbfe, nbfe+1):
                            dQ_components[comp, gj*bin_size*n_sub : (gj+1)*bin_size*n_sub , gi*bin_size*m_sub : (gi+1)*bin_size*m_sub]\
                         += a_components_pad[comp, nbfe+dy, nbfe+dx,  nbfe-dy:nbfe-dy+bin_size*n_sub, nbfe-dx:nbfe-dx+bin_size*m_sub ]\
                            *array_pad[  -dy + nbfe + gj*bin_size*n_sub :  -dy + nbfe+ (gj+1)*bin_size*n_sub  ,  -dx + nbfe + gi*bin_size*m_sub : -dx + nbfe + (gi+1)*bin_size*m_sub ]

                    dj = int(np.sin(comp*np.pi/2))
                    di = int(np.cos(comp*np.pi/2))

                    dQ_components[comp, gj*bin_size*n_sub : (gj+1)*bin_size*n_sub , gi*bin_size*m_sub : (gi+1)*bin_size*m_sub]\
                    *= 0.5*(array_pad[   nbfe + gj*bin_size*n_sub :    nbfe+ (gj+1)*bin_size*n_sub  ,    nbfe + gi*bin_size*m_sub :    nbfe + (gi+1)*bin_size*m_sub ] +\
                            array_pad[dj+nbfe + gj*bin_size*n_sub : dj+nbfe+ (gj+1)*bin_size*n_sub  , di+nbfe + gi*bin_size*m_sub : di+nbfe + (gi+1)*bin_size*m_sub]  )

        im.array[:,:]  -= dQ_components.sum(axis=0)
        im.array[:,1:] += dQ_components[0][:,:-1]
        im.array[1:,:] += dQ_components[1][:-1,:]
        im.array[:,:-1] += dQ_components[2][:,1:]
        im.array[:-1,:] += dQ_components[3][1:,:]


        return im


    def get_eff_sky_bg(self,pointing,radec):
        """
        Calculate effective sky background per pixel for nominal roman pixel scale.

        Input
        pointing            : Pointing object
        radec               : World coordinate position of image
        """

        sky_level = roman.getSkyLevel(pointing.bpass, world_pos=radec, date=pointing.date)
        sky_level *= (1.0 + roman.stray_light_fraction)*roman.pixel_scale**2

        return sky_level

    def setup_sky(self,im,pointing,rng):
        """
        Setup sky

        First we get the amount of zodaical light for a position corresponding to the position of
        the object. The results are provided in units of e-/arcsec^2, using the default Roman
        exposure time since we did not explicitly specify one. Then we multiply this by a factor
        >1 to account for the amount of stray light that is expected. If we do not provide a date
        for the observation, then it will assume that it's the vernal equinox (sun at (0,0) in
        ecliptic coordinates) in 2025.

        Input
        im                  : Image
        pointing            : Pointing object
        radec               : World coordinate position of image
        local_wcs           : Local WCS
        """

        self.rng       = rng
        self.rng_np    = np.random.default_rng(self.params['random_seed'])
        self.noise     = galsim.PoissonNoise(self.rng)
        self.dark_current_ = self.df['DARK'][:,:]*roman.exptime
        self.gain      = self.df['GAIN'][:,:]
        self.read_noise = galsim.GaussianNoise(self.rng, sigma=roman.read_noise)

        # Build current specification sky level if sky level not given
        sky_level = roman.getSkyLevel(pointing.bpass, world_pos=pointing.radec, date=pointing.date)
        sky_level *= (1.0 + roman.stray_light_fraction)
        # Make a image of the sky that takes into account the spatially variable pixel scale. Note
        # that makeSkyImage() takes a bit of time. If you do not care about the variable pixel
        # scale, you could simply compute an approximate sky level in e-/pix by multiplying
        # sky_level by roman.pixel_scale**2, and add that to final_image.

        # Create sky image
        self.sky = galsim.Image(bounds=im.bounds, wcs=pointing.WCS)
        pointing.WCS.makeSkyImage(self.sky, sky_level)

        # This image is in units of e-/pix. Finally we add the expected thermal backgrounds in this
        # band. These are provided in e-/pix/s, so we have to multiply by the exposure time.
        self.sky += roman.thermal_backgrounds[pointing.filter]*roman.exptime

        # Median of dark current is used here instead of mean since hot pixels contribute significantly to the mean.
        # Stastistics of dark current for the current test detector file: (mean, std, median, max) ~ (35, 3050, 0.008, 1.2E6)  (e-/p)
        # Hot pixels could be removed in further analysis using the dq array.
        self.sky_mean = np.mean(np.round((np.round(self.sky.array)+round(np.median(self.dark_current_)))/self.gain.mean()))

        self.sky.addNoise(self.noise)

    def add_background(self,im):
        """
        Add backgrounds to image (sky, thermal).

        First we get the amount of zodaical light for a position corresponding to the position of
        the object. The results are provided in units of e-/arcsec^2, using the default Roman
        exposure time since we did not explicitly specify one. Then we multiply this by a factor
        >1 to account for the amount of stray light that is expected. If we do not provide a date
        for the observation, then it will assume that it's the vernal equinox (sun at (0,0) in
        ecliptic coordinates) in 2025.

        Input
        im                  : Image
        """

        # If requested, dump an initial fits image to disk for diagnostics
        if self.params['save_diff']:
            orig = im.copy()
            orig.write('orig.fits', dir=self.params['diff_dir'])

        # If effect is turned off, return image unchanged
        if not self.params['use_background']:
            return im

        # Adding sky level to the image.
        im += self.sky[self.sky.bounds&im.bounds]

        # If requested, dump a post-change fits image to disk for diagnostics
        if self.params['save_diff']:
            prev = im.copy()
            diff = prev-orig
            diff.write('sky_a.fits', dir=self.params['diff_dir'])

        return im

    def recip_failure(self,im,exptime=roman.exptime,alpha=roman.reciprocity_alpha,base_flux=1.0):
        """
        Introduce reciprocity failure to image.

        Reciprocity, in the context of photography, is the inverse relationship between the
        incident flux (I) of a source object and the exposure time (t) required to produce a given
        response(p) in the detector, i.e., p = I*t. However, in NIR detectors, this relation does
        not hold always. The pixel response to a high flux is larger than its response to a low
        flux. This flux-dependent non-linearity is known as 'reciprocity failure', and the
        approximate amount of reciprocity failure for the Roman detectors is known, so we can
        include this detector effect in our images.

        Input
        im        : image
        exptime   : Exposure time
        alpha     : Reciprocity alpha
        base_flux : Base flux
        """

        # If effect is turned off, return image unchanged
        if not self.params['use_recip_failure']:
            return im

        # Add reciprocity effect
        im.addReciprocityFailure(exp_time=exptime, alpha=alpha, base_flux=base_flux)

        # If requested, dump a post-change fits image to disk for diagnostics. Both cumulative and iterative delta.
        if self.params['save_diff']:
            diff = im-prev
            diff.write('recip_a.fits')
            diff = im-orig
            diff.write('recip_b.fits')
            prev = im.copy()

        return im

    def dark_current(self,im):
        """
        Adding dark current to the image.

        Even when the detector is unexposed to any radiation, the electron-hole pairs that
        are generated within the depletion region due to finite temperature are swept by the
        high electric field at the junction of the photodiode. This small reverse bias
        leakage current is referred to as 'dark current'. It is specified by the average
        number of electrons reaching the detectors per unit time and has an associated
        Poisson noise since it is a random event.

        Input
        im               : image
        DARK[4096,4096]  : map of dark current in unit of e-/s
        """

        # If effect is turned off, return image unchanged
        if not self.params['use_dark_current']:
            return im


        dark_current_ = self.df['DARK'][:,:].flatten()*roman.exptime  #flattened 4096x4096 array
        dark_current_ = dark_current_.clip(0) #remove negative mean

        # This implementation using Galsim random functions is extremely slow
        # devs = [galsim.PoissonDeviate(rng, i) for i in dark_current_]
        # noise_array = [i() for i in devs]
        # im.array[:,:] += noise_array.reshape(im.array.shape).astype(im.dtype)

        # opt for numpy random geneator instead
        noise_array = self.rng_np.poisson(dark_current_)
        im.array[:,:] += noise_array.reshape(im.array.shape).astype(im.dtype)


        # NOTE: Sky level and dark current might appear like a constant background that can be
        # simply subtracted. However, these contribute to the shot noise and matter for the
        # non-linear effects that follow. Hence, these must be included at this stage of the
        # image generation process. We subtract these backgrounds in the end.

        return im

    def saturate(self, im):
        """
        Clip the saturation level

        Input
        im                     : image
        SATURATE[4096,4096]    : saturation map
        """

        if not self.params['use_saturate']:
            return im

        saturation_array = self.df['SATURATE'][:,:] #4096x4096 array
        where_sat = np.where(im.array > saturation_array)
        im.array[ where_sat ] = saturation_array[ where_sat ]

        return im

    def deadpix(self, im):
        """
        Apply dead pixel mask

        Input
        im                   : image
        BADPIX[4096,4096]    : bit mask with the first bit flags dead pixels
        """

        if not self.params['use_dead_pixel']:
            return im

        dead_mask = self.df['BADPIX'][:,:]&1 #4096x4096 array
        im.array[ dead_mask>0 ]=0

        return im

    def vtpe(self, im):
        """
        Apply vertical trailing pixel effect.
        The vertical trailing pixel effect (VTPE) is a non-linear effect that is
        related to readout patterns.

        Q'[j,i] = Q[j,i] + f(  Q[j,i] - Q[j-1, i]  ),

        where f( dQ ) = dQ ( a + b * ln(1 + |dQ|/dQ0) )


        Input
        im           : image
        VTPE[0,512,512]  : coefficient a binned in 8x8
        VTPE[1,512,512]  : coefficient a
        VTPE[2,512,512]  : coefficient dQ0
        """

        if not self.params['use_vtpe']:
            return im

        # expand 512x512 arrays to 4096x4096

        t = np.zeros((4096, 512))
        for row in range(t.shape[0]):
            t[row, row//8] =1
        a_vtpe = t.dot(self.df['VTPE'][0,:,:][0]).dot(t.T)
        b_vtpe = t.dot(self.df['VTPE'][1,:,:][0]).dot(t.T)
        dQ0 = t.dot(self.df['VTPE'][2,:,:][0]).dot(t.T)

        dQ = im.array - np.roll(im.array, 1, axis=0)
        dQ[0,:] *= 0

        im.array[:,:] += dQ * ( a_vtpe + b_vtpe * np.log( 1. + np.abs(dQ)/dQ0 ))
        return im

    def add_persistence(self, im, pointing):
        """
        Applying the persistence effect.

        Even after reset, some charges from prior illuminations are trapped in defects of semiconductors.
        Trapped charges are gradually released and generate the flux-dependent persistence signal.
        Here we adopt the same fermi-linear model to describe the illumination dependence and time dependence
        of the persistence effect for all SCAs.

        Input
        im                    : image
        PERSIST[6,4096,4096]  : persistence at six stimulus levels (in units of e-/pixel)
        """
        if not self.params['use_persistence']:
            return im

        #setup parameters for persistence
        Q01 = self.df['PERSIST'].read_header()['Q01']
        Q02 = self.df['PERSIST'].read_header()['Q02']
        Q03 = self.df['PERSIST'].read_header()['Q03']
        Q04 = self.df['PERSIST'].read_header()['Q04']
        Q05 = self.df['PERSIST'].read_header()['Q05']
        Q06 = self.df['PERSIST'].read_header()['Q06']
        alpha = self.df['PERSIST'].read_header()['ALPHA']


        # load the dithers of sky images that were simulated
        dither_sca_array=np.loadtxt(self.params['dither_from_file']).astype(int)

        # select adjacent exposures for the same sca (within 10*roman.exptime)
        dither_list_selected = dither_sca_array[dither_sca_array[:,1]==pointing.sca, 0]
        dither_list_selected = dither_list_selected[ np.abs(dither_list_selected-pointing.dither)<10  ]
        p_list = np.array([Pointing(self.params,None,filter_=None,sca=pointing.sca,dither=i) for i in dither_list_selected])
        dt_list = np.array([(pointing.date-p.date).total_seconds() for p in p_list])
        p_pers = p_list[ np.where((dt_list>0) & (dt_list < roman.exptime*10))]

        #iterate over previous exposures
        for p in p_pers:
            dt = (pointing.date-p.date).total_seconds() - roman.exptime/2 ##avg time since end of exposures
            fac_dt = (roman.exptime/2.)/dt  ##linear time dependence (approximate until we get t1 and Delat t of the data)
            fn = get_filename(self.params['out_path'],
                            'images',
                            self.params['output_meds'],
                            var=p.filter+'_'+str(p.dither),
                            name2=str(p.sca),
                            ftype='fits.gz',
                            overwrite=False)

            ## apply all the effects that occured before persistence on the previouse exposures
            ## since max of the sky background is of order 100, it is thus negligible for persistence
            ## same for brighter fatter effect
            bound_pad = galsim.BoundsI( xmin=1, ymin=1,
                                        xmax=4096, ymax=4096)
            x = galsim.Image(bound_pad)
            x.array[4:-4, 4:-4] = galsim.Image(fio.FITS(fn)['SCI'].read()).array[:,:]
            x = self.qe(x).array[:,:]

            x = x.clip(0) ##remove negative stimulus

            ## Do linear interpolation
            a = np.zeros(x.shape)
            a += ((x < Q01)) * x/Q01
            a += ((x >= Q01) & (x < Q02)) * (Q02-x)/(Q02-Q01)
            im.array[:,:] += a*self.df['PERSIST'][0,:,:][0]*fac_dt


            a = np.zeros(x.shape)
            a += ((x >= Q01) & (x < Q02)) * (x-Q01)/(Q02-Q01)
            a += ((x >= Q02) & (x < Q03)) * (Q03-x)/(Q03-Q02)
            im.array[:,:] += a*self.df['PERSIST'][1,:,:][0]*fac_dt

            a = np.zeros(x.shape)
            a += ((x >= Q02) & (x < Q03)) * (x-Q02)/(Q03-Q02)
            a += ((x >= Q03) & (x < Q04)) * (Q04-x)/(Q04-Q03)
            im.array[:,:] += a*self.df['PERSIST'][2,:,:][0]*fac_dt

            a = np.zeros(x.shape)
            a += ((x >= Q03) & (x < Q04)) * (x-Q03)/(Q04-Q03)
            a += ((x >= Q04) & (x < Q05)) * (Q05-x)/(Q05-Q04)
            im.array[:,:] += a*self.df['PERSIST'][3,:,:][0]*fac_dt

            a = np.zeros(x.shape)
            a += ((x >= Q04) & (x < Q05)) * (x-Q04)/(Q05-Q04)
            a += ((x >= Q05) & (x < Q06)) * (Q06-x)/(Q06-Q05)
            im.array[:,:] += a*self.df['PERSIST'][4,:,:][0]*fac_dt

            a = np.zeros(x.shape)
            a += ((x >= Q05) & (x < Q06)) * (x-Q05)/(Q06-Q05)
            a += ((x >= Q06)) * (x/Q06)**alpha       ##avoid fractional power of negative values
            im.array[:,:] += a*self.df['PERSIST'][5,:,:][0]*fac_dt


        return im

    def nonlinearity(self,im):
        """
        Applying a quadratic classical non-linearity described by the polynomial:
        Q' = Q - b_2 Q^2 - b_3 Q^3 - b_4 Q^4 to the fourth order.
        The coefficients b_i are provided in the "CNL" extionsion of the detector file.



        Input
        im                    : Image
        CNL[0, 4096, 4096]    : b_2
        CNL[1, 4096, 4096]    : b_3
        CNL[2, 4096, 4096]    : b_4
        """

        # If effect is turned off, return image unchanged
        if not self.params['use_nonlinearity']:
            return im

        # Apply the Roman nonlinearity effect
        im.array[:,:] -= self.df['CNL'][0,:,:][0] * im.array**2 +\
                         self.df['CNL'][1,:,:][0] * im.array**3 +\
                         self.df['CNL'][2,:,:][0] * im.array**4


        return im

    def interpix_cap(self,im):
        """
        Including Interpixel capacitance

        The voltage read at a given pixel location is influenced by the charges present in the
        neighboring pixel locations due to capacitive coupling of sense nodes. This interpixel
        capacitance effect is modeled as a linear effect that is described as a convolution of a
        3x3 kernel with the image.

        Q'(x,y) = \sum_{(Delta x,Delta y)} K_{Delta x, Delta y}(x-Delta x, y-Delta y) Q(x-Delta x, y-Delta y)


        Input
        im                             : image
        K[1+Delta y, 1+Delta x, y, x]  : IPC kernel with Delta x/y belongs to {-1, 0, 1}
        """

        # If effect is turned off, return image unchanged
        if not self.params['use_interpix_cap']:
            return im

        # Apply interpixel capacitance
        # pad the array by one pixel at the four edges


        num_grids = 4  ### num_grids <= 8
        grid_size = 4096//num_grids

        array_pad = im.array[4:-4,4:-4] #it's an array instead of img
        array_pad = np.pad(array_pad, [(5,5),(5,5)], mode='symmetric') #4098x4098 array

        K = self.df['IPC'][:,:,:,:]  ##3,3,512, 512

        t = np.zeros((grid_size, 512))
        for row in range(t.shape[0]):
            t[row, row//( grid_size//512) ] =1

        array_out = np.zeros( (4096, 4096))
        ##split job in sub_grids to reduce memory
        for gj in range(num_grids):
            for gi in range(num_grids):
                K_pad = np.zeros( (3,3, grid_size+2, grid_size+2) )

                for j in range(3):
                    for i in range(3):
                        tmp = (t.dot(K[j,i,:,:])).dot(t.T) #grid_sizexgrid_size
                        K_pad[j,i,:,:] = np.pad(tmp, [(1,1),(1,1)], mode='symmetric')

                for dy in range(-1, 2):
                    for dx in range(-1,2):

                        array_out[ gj*grid_size: (gj+1)*grid_size, gi*grid_size:(gi+1)*grid_size]\
                      +=K_pad[ 1+dy, 1+dx, 1-dy: 1-dy+grid_size, 1-dx:1-dx+grid_size ] \
                        *array_pad[1-dy+gj*grid_size: 1-dy+(gj+1)*grid_size, 1-dx+gi*grid_size:1-dx+(gi+1)*grid_size]

        im.array[:,:] = array_out

        return im

    def add_read_noise(self,im):
        """
        Adding read noise

        Read noise is the noise due to the on-chip amplifier that converts the charge into an
        analog voltage.  We already applied the Poisson noise due to the sky level, so read noise
        should just be added as Gaussian noise

        Input
        im                   : image
        READ[0, 4096, 4096]  : principal component for NGHXRG
        READ[1, 4096, 4096]  : CDS noise
        READ[2, 4096, 4096]  : total read noise
        """

        if not self.params['use_read_noise']:
            return im


        # use numpy random generator to draw 2-d noise map
        read_noise = self.df['READ'][2,:,:].flatten()  #flattened 4096x4096 array
        noise_array = self.rng_np.normal(loc=0., scale=read_noise)
        im.array[:,:] += noise_array.reshape(im.array.shape).astype(im.dtype)

        noise_array = self.rng_np.normal(loc=0., scale=read_noise)

        # 4088x4088 img
        self.sky.array[:,:] += noise_array.reshape(im.array.shape)[4:-4, 4:-4].astype(self.sky.dtype)
        return im

    def e_to_ADU(self,im):
        """
        We divide by the gain to convert from e- to ADU with gain and bias.

        Input
        im : image
        GAIN : 32x32 float img in unit of e-/adu, mean(GAIN)~ 1.6
        BIAS : 4096x4096 uint16 bias img (in unit of DN), mean(bias) ~ 6.7k
        """

        if not self.params['use_gain_bias']:
            return im

        gain = self.df['GAIN'][:,:] #32x32 img
        bias = self.df['BIAS'][:,:] #4096x4096 img

        t = np.zeros((4096, 32))
        for row in range(t.shape[0]):
            t[row, row//128] =1
        gain_expand = (t.dot(gain)).dot(t.T) #4096x4096 gain img
        im.array[:,:] = im.array/gain_expand + bias
        return im

    def add_gain(self,im):
        """
        We divide by the gain to convert from e- to ADU.

        Input
        im : image
        GAIN : 32x32 float img in unit of e-/adu, mean(GAIN)~ 1.6
        """

        if not self.params['use_gain']:
            return im

        gain = self.df['GAIN'][:,:] #32x32 img

        t = np.zeros((4096, 32))
        for row in range(t.shape[0]):
            t[row, row//128] =1
        gain_expand = (t.dot(gain)).dot(t.T) #4096x4096 gain img
        im.array[:,:] /= gain_expand
        return im

    def add_bias(self,im):
        """
        Add the voltage bias.

        Input
        im : image
        BIAS : 4096x4096 uint16 bias img (in unit of DN), mean(bias) ~ 6.7k
        """

        if not self.params['use_bias']:
            return im

        bias = self.df['BIAS'][:,:] #4096x4096 img

        im.array[:,:] +=  bias
        return im

    def finalize_sky_im(self,im):
        """
        Finalize sky background for subtraction from final image. Add dark current,
        convert to analog voltage, and quantize.

        Input
        im : sky image
        """

        if (self.params['sub_true_background'])&(self.params['use_dark_current']):
            im = (im + round(self.dark_current_))
        im = self.e_to_ADU(im)
        im.quantize()

        return im

    def finalize_background_subtract(self,im,sky):
        """
        Finalize background subtraction of image.

        Input
        im : image
        sky : sky image
        """

        # If effect is turned off, return image unchanged
        if not self.params['use_background']:
            return im,sky

        sky.quantize() # Quantize sky
        sky = self.finalize_sky_im(sky) # Finalize sky with dark current, convert to ADU, and quantize.
        im -= sky

        # If requested, dump a final fits image to disk for diagnostics.
        if self.params['save_diff']:
            im.write('final_a.fits')

        return im,sky
