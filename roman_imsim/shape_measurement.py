# from __future__ import division
# from __future__ import print_function

# from future import standard_library
# standard_library.install_aliases()
# from builtins import str
# from builtins import range
# from past.builtins import basestring
# from builtins import object
from re import I
import xxlimited
from ngmix.metacal import _make_symmetrized_image
from past.utils import old_div

import numpy as np
import healpy as hp
import sys, os, io
import logging
import yaml
import galsim as galsim
import galsim.roman as roman
import galsim.config.process as process
import ngmix
import fitsio as fio
import pickle as pickle
from astropy.time import Time
from mpi4py import MPI
# from mpi_pool import MPIPool
import cProfile, pstats, psutil
import glob
import shutil
import meds
from ngmix.jacobian import Jacobian
from ngmix.observation import Observation, ObsList, MultiBandObsList,make_kobs
from ngmix.galsimfit import GalsimRunner,GalsimSimple,GalsimTemplateFluxFitter
from ngmix.guessers import R50FluxGuesser
from ngmix.bootstrap import PSFRunner
from ngmix import priors, joint_prior
import psc
from skimage.measure import block_reduce

#from .telescope import pointing 
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

import roman_imsim

class shape_measurement(object):

    def __init__(self, param_file, filter_, pix, comm, ignore_missing_files = False, setup = False, condor_build=False, shape=False, shape_iter = None, shape_cnt = None):

        self.params     = yaml.load(open(param_file))
        self.param_file = param_file
        # Do some parsing
        for key in list(self.params.keys()):
            if self.params[key]=='None':
                self.params[key]=None
            if self.params[key]=='none':
                self.params[key]=None
            if self.params[key]=='True':
                self.params[key]=True
            if self.params[key]=='False':
                self.params[key]=False
        self.ditherfile = self.params['dither_file']
        logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
        self.logger = logging.getLogger('roman_sim')
        self.filter_ = filter_
        self.pointing   = roman_imsim.pointing(self.params,self.logger,filter_=self.filter_,sca=None,dither=None)
        self.pix = pix
        self.skip = False

        # Getting MPI ranks to parallelize the measurement. 
        self.comm = comm
        status = MPI.Status()
        if self.comm is None:
            self.rank = 0
            self.size = 1
        else:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

        # This is only necessary when running condor. This process can be replaces by running MPI. 
        # If not using condor, just set shape_iter=0 and shape_cnt=1. 
        self.shape_iter = shape_iter
        if self.shape_iter is None:
            self.shape_iter = 0
        self.shape_cnt  = shape_cnt
        if self.shape_cnt is None:
            self.shape_cnt = 1

        condor = False
        print('mpi check',self.rank,self.size)
        if not condor:
            os.chdir(os.environ['TMPDIR'].replace('[','[').replace(']',']'))

        # -> moved the production of roman PSF dictionary to simulate_roman_psfs() function.   
        if shape and not self.params['drizzle_coadd']:

            if ('output_meds' not in self.params) or ('psf_meds' not in self.params):
                raise ParamError('Must define both output_meds and psf_meds in yaml')
            if (self.params['output_meds'] is None) or (self.params['psf_meds'] is None):
                raise ParamError('Must define both output_meds and psf_meds in yaml')
            print('shape',self.shape_iter,self.shape_cnt)

            self.meds_filename = get_filename(  self.params['out_path'],
                                                'meds',
                                                self.params['output_meds'],
                                                var=self.pointing.filter+'_'+str(self.pix),
                                                ftype='fits.gz',
                                                overwrite=False)

            self.local_meds = get_filename( '/scratch/',
                                            '',
                                            self.params['output_meds'],
                                            var=self.pointing.filter+'_'+str(self.pix),
                                            ftype='fits',
                                            overwrite=False)
            # When doing multiband measurement, those meds files also need to be copied to /scratch and read. 
            if self.params['multiband']:
                self.meds_Jfilename = '/hpc/group/cosmology/phy-lsst/my137/roman_J129/'+self.params['sim_set']+'/meds/fiducial_J129_'+str(self.pix)+'.fits.gz'
                self.local_Jmeds = '/scratch/fiducial_J129_'+str(self.pix)+'.fits'
                if self.params['multiband_filter'] == 3:
                    self.meds_Ffilename = '/hpc/group/cosmology/phy-lsst/my137/roman_F184/'+self.params['sim_set']+'/meds/fiducial_F184_'+str(self.pix)+'.fits.gz'
                    self.local_Fmeds = '/scratch/fiducial_F184_'+str(self.pix)+'.fits'

            # PSFs are not saved in meds files for the current version, so these are the same as self.meds_filename and self.local_meds. 
            self.meds_psf = get_filename(   self.params['psf_path'],
                                            'meds',
                                            self.params['psf_meds'],
                                            var=self.pointing.filter+'_'+str(self.pix),
                                            ftype='fits.gz',
                                            overwrite=False)

            self.local_meds_psf = get_filename( '/scratch/',
                                                '',
                                                self.params['psf_meds'],
                                                var=self.pointing.filter+'_'+str(self.pix),
                                                ftype='fits',
                                                overwrite=False)

            # Only rank=0 does the copy operations. 
            if self.rank==0:
                shutil.copy(self.meds_filename,self.local_meds+'.gz')

                os.system( 'gunzip '+self.local_meds+'.gz')
                if self.params['multiband']:
                    shutil.copy(self.meds_Jfilename,self.local_Jmeds+'.gz')
                    os.system( 'gunzip '+self.local_Jmeds+'.gz')

                    shutil.copy(self.meds_Ffilename,self.local_Fmeds+'.gz')
                    os.system( 'gunzip '+self.local_Fmeds+'.gz')
                

                if self.local_meds != self.local_meds_psf:
                    shutil.copy(self.meds_psf,self.local_meds_psf+'.gz')
                    if os.path.exists(self.local_meds_psf):
                        os.remove(self.local_meds_psf)
                    os.system( 'gunzip '+self.local_meds_psf+'.gz')

            self.comm.Barrier()

            return
        elif shape and self.params['drizzle_coadd']:
            self.tilename = fio.read(os.path.join(self.params['out_path'], 'truth/fiducial_coaddlist.fits.gz'))['tilename'][pix]
            self.drizzle_cutout_filename = get_filename(self.params['out_path'],
                                                        'images/coadd/coadd_cutouts',
                                                        self.params['output_meds'],
                                                        var=self.pointing.filter+'_'+str(self.tilename)+'_cutouts',
                                                        ftype='pickle',
                                                        overwrite=False)
            if not os.path.exists(self.drizzle_cutout_filename):
                print('No cutout images found. Quiting...')
                sys.exit()
            self.local_drizzle_cutout = get_filename('/scratch',
                                                    '',
                                                    self.params['output_meds'],
                                                    var=self.pointing.filter+'_'+str(self.tilename)+'_cutouts',
                                                    ftype='pickle',
                                                    overwrite=False)
            if self.rank == 0:
                shutil.copy(self.drizzle_cutout_filename,self.local_drizzle_cutout)
            self.comm.Barrier()

            return 
        else:
            self.file_exists = False


    def __del__(self):

        if self.params['drizzle_coadd']:
            try:
                os.remove(self.local_drizzle_cutout)
            except:
                pass
        else:
            try:
                os.remove(self.local_meds)
                #os.remove(self.local_meds_psf)
                os.remove(self.local_Jmeds)
                if self.params['multiband_filter'] == 3:
                    os.remove(self.local_Fmeds)
            except:
                pass

    def simulate_roman_psf(self):

        # Since we did not save the PSF in the meds files, we need to create and save the Roman PSF at the center of
        # each SCA for each bandpass. This process is not necessary if we have drizzled coadd, because the PSFs are saved
        # in the coadd files for the drizzle coadd.

        # Get PSFs for all SCAs
        all_scas = np.array([i for i in range(1,19)])
        if not self.params['multiband']:
            self.all_psfs = {self.filter_: []}
        else:
            self.all_psfs = {self.filter_: [], 'J129': [], 'F184': []}

        for sca in all_scas:
            # Make PSFs for whatever bandpass we're doing the measurement with. 
            psf_sca = roman.getPSF(sca, 
                                   self.filter_, 
                                   SCA_pos=None, 
                                   pupil_bin=4,
                                   wavelength=roman.getBandpasses(AB_zeropoint=True)[self.filter_].effective_wavelength)
            self.all_psfs[self.filter_].append(psf_sca)

            # When doing the multiband, self.filter is always set to H158 so we need to get the PSFs for F184 and J129. 
            if self.params['multiband']: 
                Jpsf_sca = roman.getPSF(sca, 
                                        'J129', 
                                        SCA_pos=None, 
                                        pupil_bin=4,
                                        wavelength=roman.getBandpasses(AB_zeropoint=True)['J129'].effective_wavelength)
                self.all_psfs['J129'].append(Jpsf_sca)

                Fpsf_sca = roman.getPSF(sca, 
                                    'F184', 
                                    SCA_pos=None, 
                                    pupil_bin=4,
                                    wavelength=roman.getBandpasses(AB_zeropoint=True)['F184'].effective_wavelength)
                self.all_psfs['F184'].append(Fpsf_sca)

        return self.all_psfs

    def get_cutout_psf2(self,m,m2,i,j):

        box_size = m['psf_box_size2'][i]
        start_row = m['psf_start_row2'][i, j]
        row_end = start_row + box_size*box_size

        imflat = m2['psf2'][start_row:row_end]
        im = imflat.reshape(box_size, box_size)
        return im

    def get_cutout_psf(self,m,m2,i,j):

        box_size = m['psf_box_size'][i]
        start_row = m['psf_start_row'][i, j]
        row_end = start_row + box_size*box_size

        #imflat = m2['psf'][start_row:row_end]
        imflat = m2['psf'][start_row:row_end]
        im = imflat.reshape(box_size, box_size)
        return im

    # Leaving get_exp_list() and get_exp_list_coadd() as it is, because the new version of the simulations has already produced the coadded images. 
    # We will not need to create coadds on the fly as Yamamoto et al. 2022 did with psc. Sice we will be able to measure the shapes directly on the coadds, 
    # we will probably not need these functions to create the coadds. 
    def get_exp_list(self,m,i,m2=None,size=None):

        m3=[0]
        for jj,psf_ in enumerate(m2):
            if jj==0:
                continue
            gal_stamp_center_row=m['orig_start_row'][i][jj] + m['box_size'][i]/2 
            gal_stamp_center_col=m['orig_start_col'][i][jj] + m['box_size'][i]/2
            
            b = galsim.BoundsI( xmin=(m['orig_start_col'][i][jj]+(m['box_size'][i]-32)/2.), 
                                xmax=(m['orig_start_col'][i][jj]+m['box_size'][i]-(m['box_size'][i]-32)/2.) - 1,
                                ymin=(m['orig_start_row'][i][jj]+(m['box_size'][i]-32)/2.),
                                ymax=(m['orig_start_row'][i][jj]+m['box_size'][i]-(m['box_size'][i]-32)/2.) - 1)
            
            wcs_ = self.make_jacobian(m.get_jacobian(i,jj)['dudcol'],
                                    m.get_jacobian(i,jj)['dudrow'],
                                    m.get_jacobian(i,jj)['dvdcol'],
                                    m.get_jacobian(i,jj)['dvdrow'],
                                    m['orig_col'][i][jj],
                                    m['orig_row'][i][jj]) 
            scale = galsim.PixelScale(roman.pixel_scale)
            psf_ = wcs_.toWorld(scale.toImage(psf_), image_pos=galsim.PositionD(roman.n_pix/2, roman.n_pix/2))
            
            #st_model = galsim.DeltaFunction(flux=1.)
            #st_model = st_model.evaluateAtWavelength(roman.getBandpasses(AB_zeropoint=True)[self.filter_].effective_wavelength)
            #st_model = st_model.withFlux(1.)
            #st_model = galsim.Convolve(st_model, psf_)
            psf_stamp = galsim.Image(b, wcs=wcs_) #scale=roman.pixel_scale/self.params['oversample']) 

            offset_x = m['orig_col'][i][jj] - gal_stamp_center_col 
            offset_y = m['orig_row'][i][jj] - gal_stamp_center_row 
            offset = galsim.PositionD(offset_x, offset_y)
            psf_.drawImage(image=psf_stamp)
            m3.append(psf_stamp.array)


        if m2 is None:
            m2 = m

        obs_list=ObsList()
        psf_list=ObsList()

        included = []
        w        = []
        # For each of these objects create an observation
        for j in range(m['ncutout'][i]):
            if j==0:
                continue
            # if j>1:
            #     continue
            im = m.get_cutout(i, j, type='image')
            weight = m.get_cutout(i, j, type='weight')

            im_psf = m3[j] #self.get_cutout_psf(m, m2, i, j)
            im_psf2 = im_psf #self.get_cutout_psf2(m, m2, i, j)
            if np.sum(im)==0.:
                print(self.local_meds, i, j, np.sum(im))
                print('no flux in image ',i,j)
                continue

            jacob = m.get_jacobian(i, j)
            gal_jacob=Jacobian(

                row=(m['orig_row'][i][j]-m['orig_start_row'][i][j]), 
                col=(m['orig_col'][i][j]-m['orig_start_col'][i][j]),

                dvdrow=jacob['dvdrow'],
                dvdcol=jacob['dvdcol'],
                dudrow=jacob['dudrow'],
                dudcol=jacob['dudcol'])

            psf_center = (32*self.params['oversample']/2.)+0.5
            psf_jacob2=Jacobian(
                row=(m['orig_row'][i][j]-m['orig_start_row'][i][j]-(m['box_size'][i]-32)/2.), 
                col=(m['orig_col'][i][j]-m['orig_start_col'][i][j]-(m['box_size'][i]-32)/2.),
                dvdrow=jacob['dvdrow'],
                dvdcol=jacob['dvdcol'],
                dudrow=jacob['dudrow'],
                dudcol=jacob['dudcol'])

            # Create an obs for each cutout
            mask = np.where(weight!=0)
            if 1.*len(weight[mask])/np.product(np.shape(weight))<0.8:
                continue
            w.append(np.mean(weight[mask]))
            # noise = np.ones_like(weight)/w[-1]

            mask_zero = np.where(weight==0)
            noise = galsim.Image(np.ones_like(weight)/weight, scale=galsim.roman.pixel_scale)
            p_noise = galsim.PoissonNoise(galsim.BaseDeviate(1234), sky_level=0.)
            noise.array[mask_zero] = np.mean(weight[mask])
            noise.addNoise(p_noise)
            noise -= (1/np.mean(weight[mask]))

            psf_obs = Observation(im_psf, jacobian=gal_jacob, meta={'offset_pixels':None,'file_id':None})  
            psf_obs2 = Observation(im_psf2, jacobian=psf_jacob2, meta={'offset_pixels':None,'file_id':None})
            obs = Observation(im, weight=weight, jacobian=gal_jacob, psf=psf_obs2, meta={'offset_pixels':None,'file_id':None})
            #obs = Observation(im, weight=weight, jacobian=psf_jacob2, psf=psf_obs2, meta={'offset_pixels':None,'file_id':None})
            # obs.set_noise(noise)
            obs.set_noise(noise.array)

            obs_list.append(obs)
            psf_list.append(psf_obs2)
            included.append(j)

        return obs_list,psf_list,np.array(included)-1,np.array(w)

    # Leaving get_exp_list() and get_exp_list_coadd() as it is, because the new version of the simulations has already produced the coadded images. 
    # We will not need to create coadds on the fly as Yamamoto et al. 2022 did with psc. Sice we will be able to measure the shapes directly on the coadds, 
    # we will probably not need these functions to create the coadds.
    def get_exp_list_coadd(self,m,i,m2=None,size=None):

        m3=[0]
        for jj,psf_ in enumerate(m2): # m2 has 18 psfs that are centered at each SCA. 
            if jj==0:
                continue
            gal_stamp_center_row=m['orig_start_row'][i][jj] + m['box_size'][i]/2 - 0.5 # m['box_size'] is the galaxy stamp size. 
            gal_stamp_center_col=m['orig_start_col'][i][jj] + m['box_size'][i]/2 - 0.5 # m['orig_start_row/col'] is in SCA coordinates. 
            psf_stamp_size=32
            
            # Make the bounds for the psf stamp. 
            b = galsim.BoundsI( xmin=(m['orig_start_col'][i][jj]+(m['box_size'][i]-32)/2. - 1)*self.params['oversample']+1, 
                                xmax=(m['orig_start_col'][i][jj]+(m['box_size'][i]-32)/2.+psf_stamp_size-1)*self.params['oversample'],
                                ymin=(m['orig_start_row'][i][jj]+(m['box_size'][i]-32)/2. - 1)*self.params['oversample']+1,
                                ymax=(m['orig_start_row'][i][jj]+(m['box_size'][i]-32)/2.+psf_stamp_size-1)*self.params['oversample'])
    
            # Make wcs for oversampled psf. 
            wcs_ = self.make_jacobian(m.get_jacobian(i,jj)['dudcol']/self.params['oversample'],
                                    m.get_jacobian(i,jj)['dudrow']/self.params['oversample'],
                                    m.get_jacobian(i,jj)['dvdcol']/self.params['oversample'],
                                    m.get_jacobian(i,jj)['dvdrow']/self.params['oversample'],
                                    m['orig_col'][i][jj]*self.params['oversample'],
                                    m['orig_row'][i][jj]*self.params['oversample']) 
            # Taken from galsim/roman_psfs.py line 266. Update each psf to an object-specific psf using the wcs. 
            scale = galsim.PixelScale(roman.pixel_scale/self.params['oversample'])
            psf_ = wcs_.toWorld(scale.toImage(psf_), image_pos=galsim.PositionD(roman.n_pix/2, roman.n_pix/2))
            
            # Convolve with the star model and get the psf stamp. 
            #st_model = galsim.DeltaFunction(flux=1.)
            #st_model = st_model.evaluateAtWavelength(roman.getBandpasses(AB_zeropoint=True)[self.filter_].effective_wavelength)
            #st_model = st_model.withFlux(1.)
            #st_model = galsim.Convolve(st_model, psf_)
            psf_ = galsim.Convolve(psf_, galsim.Pixel(roman.pixel_scale))
            psf_stamp = galsim.Image(b, wcs=wcs_) 

            # Galaxy is being drawn with some subpixel offsets, so we apply the offsets when drawing the psf too. 
            offset_x = m['orig_col'][i][jj] - gal_stamp_center_col 
            offset_y = m['orig_row'][i][jj] - gal_stamp_center_row 
            offset = galsim.PositionD(offset_x, offset_y)
            psf_.drawImage(image=psf_stamp, offset=offset, method='no_pixel') 
            m3.append(psf_stamp.array)

        if m2 is None:
            m2 = m

        obs_list=ObsList()
        psf_list=ObsList()

        included = []
        w        = []
        # For each of these objects create an observation
        for j in range(m['ncutout'][i]):
            if j==0:
                continue
            # if j>1:
            #     continue
            im = m.get_cutout(i, j, type='image')
            weight = m.get_cutout(i, j, type='weight')

            #m2[j] = psf_offset(i,j,m2[j])
            im_psf = m3[j] #self.get_cutout_psf(m, m2, i, j)
            im_psf2 = im_psf #self.get_cutout_psf2(m, m2, i, j)
            if np.sum(im)==0.:
                print(self.local_meds, i, j, np.sum(im))
                print('no flux in image ',i,j)
                continue

            jacob = m.get_jacobian(i, j)
            # Get a galaxy jacobian. 
            gal_jacob=Jacobian(
                row=(m['orig_row'][i][j]-m['orig_start_row'][i][j]),
                col=(m['orig_col'][i][j]-m['orig_start_col'][i][j]),
                dvdrow=jacob['dvdrow'],
                dvdcol=jacob['dvdcol'],
                dudrow=jacob['dudrow'],
                dudcol=jacob['dudcol']) 

            psf_center = (32/2.)+0.5 
            # Get a oversampled psf jacobian. 
            if self.params['oversample']==1:
                psf_jacob2=Jacobian(
                    row=15.5 + (m['orig_row'][i][j]-m['orig_start_row'][i][j]+1-(m['box_size'][i]/2.+0.5))*self.params['oversample'],
                    col=15.5 + (m['orig_col'][i][j]-m['orig_start_col'][i][j]+1-(m['box_size'][i]/2.+0.5))*self.params['oversample'], 
                    dvdrow=jacob['dvdrow']/self.params['oversample'],
                    dvdcol=jacob['dvdcol']/self.params['oversample'],
                    dudrow=jacob['dudrow']/self.params['oversample'],
                    dudcol=jacob['dudcol']/self.params['oversample']) 
            elif self.params['oversample']==4:
                psf_jacob2=Jacobian(
                    row=63.5 + (m['orig_row'][i][j]-m['orig_start_row'][i][j]+1-(m['box_size'][i]/2.+0.5))*self.params['oversample'],
                    col=63.5 + (m['orig_col'][i][j]-m['orig_start_col'][i][j]+1-(m['box_size'][i]/2.+0.5))*self.params['oversample'], 
                    dvdrow=jacob['dvdrow']/self.params['oversample'],
                    dvdcol=jacob['dvdcol']/self.params['oversample'],
                    dudrow=jacob['dudrow']/self.params['oversample'],
                    dudcol=jacob['dudcol']/self.params['oversample']) 


            # Create an obs for each cutout
            mask = np.where(weight!=0)
            if 1.*len(weight[mask])/np.product(np.shape(weight))<0.8:
                continue

            w.append(np.mean(weight[mask]))
            noise = np.ones_like(weight)/w[-1]

            psf_obs = Observation(im_psf, jacobian=gal_jacob, meta={'offset_pixels':None,'file_id':None})
            psf_obs2 = Observation(im_psf2, jacobian=psf_jacob2, meta={'offset_pixels':None,'file_id':None})
            #obs = Observation(im, weight=weight, jacobian=gal_jacob, psf=psf_obs, meta={'offset_pixels':None,'file_id':None})
            # oversampled PSF
            obs = Observation(im, weight=weight, jacobian=gal_jacob, psf=psf_obs2, meta={'offset_pixels':None,'file_id':None})
            obs.set_noise(noise)
            # obs.set_noise(noise.array)

            obs_list.append(obs)
            psf_list.append(psf_obs2)
            included.append(j)

        return obs_list,psf_list,np.array(included)-1,np.array(w)

    # This function is solely here to add noise in the images, since the old version of the simulations did not save the noise images in the meds files. 
    # This function will not be necessary in the future version. 
    def get_exp_list_coadd_with_noise_image(self,m,i,m2=None,size=None):

        m3=[0]
        for jj,psf_model in enumerate(m2): 
            # m2 contains 18 psfs that are centered at each SCA. Created at line 117. 
            # These PSFs are in image coordinates and have not rotated according to the wcs. These are merely templates. 
            # We want to rotate the PSF template according to the wcs, and oversample it.
            if jj==0:
                continue
            gal_stamp_center_row=m['orig_start_row'][i][jj] + m['box_size'][i]/2 - 0.5 # m['box_size'] is the galaxy stamp size. 
            gal_stamp_center_col=m['orig_start_col'][i][jj] + m['box_size'][i]/2 - 0.5 # m['orig_start_row/col'] is in SCA coordinates. 
            psf_stamp_size=32
            
            # Make the bounds for the psf stamp. 
            b = galsim.BoundsI( xmin=(m['orig_start_col'][i][jj]+(m['box_size'][i]-32)/2. - 1)*self.params['oversample']+1, 
                                xmax=(m['orig_start_col'][i][jj]+(m['box_size'][i]-32)/2.+psf_stamp_size-1)*self.params['oversample'],
                                ymin=(m['orig_start_row'][i][jj]+(m['box_size'][i]-32)/2. - 1)*self.params['oversample']+1,
                                ymax=(m['orig_start_row'][i][jj]+(m['box_size'][i]-32)/2.+psf_stamp_size-1)*self.params['oversample'])
    
            # Make wcs for oversampled psf. 
            wcs_ = self.make_jacobian(m.get_jacobian(i,jj)['dudcol']/self.params['oversample'],
                                    m.get_jacobian(i,jj)['dudrow']/self.params['oversample'],
                                    m.get_jacobian(i,jj)['dvdcol']/self.params['oversample'],
                                    m.get_jacobian(i,jj)['dvdrow']/self.params['oversample'],
                                    m['orig_col'][i][jj]*self.params['oversample'],
                                    m['orig_row'][i][jj]*self.params['oversample']) 
            # Taken from galsim/roman_psfs.py line 266. Update each psf to an object-specific psf using the wcs. 
            # Apply WCS.
            # The PSF is in arcsec units, but oriented parallel to the image coordinates.
            # So to apply the right WCS, project to pixels using the Roman mean pixel_scale, then
            # project back to world coordinates with the provided wcs.
            scale = galsim.PixelScale(roman.pixel_scale/self.params['oversample'])
            # Image coordinates to world coordinates. PSF models were drawn at the center of the SCA. 
            psf_ = wcs_.toWorld(scale.toImage(psf_model), image_pos=galsim.PositionD(roman.n_pix/2, roman.n_pix/2))
            # Convolve the psf with oversampled pixel scale. Note that we should convolve with galsim.Pixel(self.params['oversample']), not galsim.Pixel(1.0)
            psf_ = wcs_.toWorld(galsim.Convolve(wcs_.toImage(psf_), galsim.Pixel(self.params['oversample'])))
            psf_stamp = galsim.Image(b, wcs=wcs_) 

            # Galaxy is being drawn with some subpixel offsets, so we apply the offsets when drawing the psf too. 
            offset_x = m['orig_col'][i][jj] - gal_stamp_center_col 
            offset_y = m['orig_row'][i][jj] - gal_stamp_center_row 
            offset = galsim.PositionD(offset_x, offset_y)
            psf_.drawImage(image=psf_stamp, offset=offset, method='no_pixel') 
            m3.append(psf_stamp.array)
            

        if m2 is None:
            m2 = m

        obs_list=ObsList()
        psf_list=ObsList()

        included = []
        w        = []
        # For each of these objects create an observation
        for j in range(m['ncutout'][i]):
            if j==0:
                continue
            # if j>1:
            #     continue
            im = m.get_cutout(i, j, type='image')
            weight = m.get_cutout(i, j, type='weight')

            #m2[j] = psf_offset(i,j,m2[j])
            im_psf = m3[j] #self.get_cutout_psf(m, m2, i, j)
            im_psf2 = im_psf #self.get_cutout_psf2(m, m2, i, j)
            if np.sum(im)==0.:
                print(self.local_meds, i, j, np.sum(im))
                print('no flux in image ',i,j)
                continue

            jacob = m.get_jacobian(i, j)
            # Get a galaxy jacobian. 
            gal_jacob=Jacobian(
                row=(m['orig_row'][i][j]-m['orig_start_row'][i][j]),
                col=(m['orig_col'][i][j]-m['orig_start_col'][i][j]),
                dvdrow=jacob['dvdrow'],
                dvdcol=jacob['dvdcol'],
                dudrow=jacob['dudrow'],
                dudcol=jacob['dudcol']) 

            psf_center = (32/2.)+0.5 
            # Get a oversampled psf jacobian. 
            if self.params['oversample']==1:
                psf_jacob2=Jacobian(
                    row=15.5 + (m['orig_row'][i][j]-m['orig_start_row'][i][j]+1-(m['box_size'][i]/2.+0.5))*self.params['oversample'],
                    col=15.5 + (m['orig_col'][i][j]-m['orig_start_col'][i][j]+1-(m['box_size'][i]/2.+0.5))*self.params['oversample'], 
                    dvdrow=jacob['dvdrow']/self.params['oversample'],
                    dvdcol=jacob['dvdcol']/self.params['oversample'],
                    dudrow=jacob['dudrow']/self.params['oversample'],
                    dudcol=jacob['dudcol']/self.params['oversample']) 
            elif self.params['oversample']==4:
                psf_jacob2=Jacobian(
                    row=63.5 + (m['orig_row'][i][j]-m['orig_start_row'][i][j]+1-(m['box_size'][i]/2.+0.5))*self.params['oversample'],
                    col=63.5 + (m['orig_col'][i][j]-m['orig_start_col'][i][j]+1-(m['box_size'][i]/2.+0.5))*self.params['oversample'], 
                    dvdrow=jacob['dvdrow']/self.params['oversample'],
                    dvdcol=jacob['dvdcol']/self.params['oversample'],
                    dudrow=jacob['dudrow']/self.params['oversample'],
                    dudcol=jacob['dudcol']/self.params['oversample']) 

            # Create an obs for each cutout
            mask = np.where(weight!=0)
            if 1.*len(weight[mask])/np.product(np.shape(weight))<0.8:
                continue
            w.append(np.mean(weight[mask]))
            # noise = np.ones_like(weight)/w[-1]
            
            mask_zero = np.where(weight==0)
            noise = galsim.Image(np.ones_like(weight)/weight, scale=galsim.roman.pixel_scale)
            p_noise = galsim.PoissonNoise(galsim.BaseDeviate(1234), sky_level=0.)
            noise.array[mask_zero] = np.mean(weight[mask])
            noise.addNoise(p_noise)
            noise -= (1/np.mean(weight[mask]))

            psf_obs = Observation(im_psf, jacobian=gal_jacob, meta={'offset_pixels':None,'file_id':None})
            psf_obs2 = Observation(im_psf2, jacobian=psf_jacob2, meta={'offset_pixels':None,'file_id':None})
            #obs = Observation(im, weight=weight, jacobian=gal_jacob, psf=psf_obs, meta={'offset_pixels':None,'file_id':None})
            # oversampled PSF
            obs = Observation(im, weight=weight, jacobian=gal_jacob, psf=psf_obs2, meta={'offset_pixels':None,'file_id':None})
            # obs.set_noise(noise)
            obs.set_noise(noise.array)

            obs_list.append(obs)
            psf_list.append(psf_obs2)
            included.append(j)

        return obs_list,psf_list,np.array(included)-1,np.array(w)

    # This function will need to be re-written according to the file format of the drizzle coadd and the new meds-like files. 
    # This function create the exposure list from the meds-like files for the single-epoch and coadd measurement, in order to 
    # follow the format of ngmix. 
    def get_exp_list_coadd_drizzle(self,m,i,m2=None,size=None):

        #def psf_offset(i,j,star_):
        m3=[0]
        #relative_offset=[0]
        for jj,psf_ in enumerate(m2): # m2 has 18 psfs that are centered at each SCA. Created at line 117. 
            if jj==0:
                continue
            gal_stamp_center_row=m['orig_start_row'][i][jj] + m['box_size'][i]/2 - 0.5 # m['box_size'] is the galaxy stamp size. 
            gal_stamp_center_col=m['orig_start_col'][i][jj] + m['box_size'][i]/2 - 0.5 # m['orig_start_row/col'] is in SCA coordinates. 
            psf_stamp_size=32
            
            # Make the bounds for the psf stamp. 
            b = galsim.BoundsI( xmin=(m['orig_start_col'][i][jj]+(m['box_size'][i]-32)/2. - 1)*self.params['oversample']+1, 
                                xmax=(m['orig_start_col'][i][jj]+(m['box_size'][i]-32)/2.+psf_stamp_size-1)*self.params['oversample'],
                                ymin=(m['orig_start_row'][i][jj]+(m['box_size'][i]-32)/2. - 1)*self.params['oversample']+1,
                                ymax=(m['orig_start_row'][i][jj]+(m['box_size'][i]-32)/2.+psf_stamp_size-1)*self.params['oversample'])
    
            # Make wcs for oversampled psf. 
            wcs_ = self.make_jacobian(m.get_jacobian(i,jj)['dudcol']/self.params['oversample'],
                                    m.get_jacobian(i,jj)['dudrow']/self.params['oversample'],
                                    m.get_jacobian(i,jj)['dvdcol']/self.params['oversample'],
                                    m.get_jacobian(i,jj)['dvdrow']/self.params['oversample'],
                                    m['orig_col'][i][jj]*self.params['oversample'],
                                    m['orig_row'][i][jj]*self.params['oversample']) 
            # Taken from galsim/roman_psfs.py line 266. Update each psf to an object-specific psf using the wcs. 
            scale = galsim.PixelScale(roman.pixel_scale/self.params['oversample'])
            psf_ = wcs_.toWorld(scale.toImage(psf_), image_pos=galsim.PositionD(roman.n_pix/2, roman.n_pix/2))
            
            # Convolve with the star model and get the psf stamp. 
            #st_model = galsim.DeltaFunction(flux=1.)
            #st_model = st_model.evaluateAtWavelength(roman.getBandpasses(AB_zeropoint=True)[self.filter_].effective_wavelength)
            #st_model = st_model.withFlux(1.)
            #st_model = galsim.Convolve(st_model, psf_)
            psf_ = galsim.Convolve(psf_, galsim.Pixel(roman.pixel_scale))
            psf_stamp = galsim.Image(b, wcs=wcs_) 

            # Galaxy is being drawn with some subpixel offsets, so we apply the offsets when drawing the psf too. 
            offset_x = m['orig_col'][i][jj] - gal_stamp_center_col 
            offset_y = m['orig_row'][i][jj] - gal_stamp_center_row 
            offset = galsim.PositionD(offset_x, offset_y)
            psf_.drawImage(image=psf_stamp, offset=offset, method='no_pixel') 
            m3.append(psf_stamp.array)

        if m2 is None:
            m2 = m

        obs_list=ObsList()
        psf_list=ObsList()

        included = []
        w        = []
        # For each of these objects create an observation
        for j in range(m['ncutout'][i]):
            if j==0:
                continue
            # if j>1:
            #     continue
            im = m.get_cutout(i, j, type='image')
            weight = m.get_cutout(i, j, type='weight')

            #m2[j] = psf_offset(i,j,m2[j])
            im_psf = m3[j] #self.get_cutout_psf(m, m2, i, j)
            im_psf2 = im_psf #self.get_cutout_psf2(m, m2, i, j)
            if np.sum(im)==0.:
                print(self.local_meds, i, j, np.sum(im))
                print('no flux in image ',i,j)
                continue

            jacob = m.get_jacobian(i, j)
            # Get a galaxy jacobian. 
            gal_jacob=Jacobian(
                row=(m['orig_row'][i][j]-m['orig_start_row'][i][j]),
                col=(m['orig_col'][i][j]-m['orig_start_col'][i][j]),
                dvdrow=jacob['dvdrow'],
                dvdcol=jacob['dvdcol'],
                dudrow=jacob['dudrow'],
                dudcol=jacob['dudcol']) 

            psf_center = (32/2.)+0.5 
            # Get a oversampled psf jacobian. 
            if self.params['oversample']==1:
                psf_jacob2=Jacobian(
                    row=15.5 + (m['orig_row'][i][j]-m['orig_start_row'][i][j]+1-(m['box_size'][i]/2.+0.5))*self.params['oversample'],
                    col=15.5 + (m['orig_col'][i][j]-m['orig_start_col'][i][j]+1-(m['box_size'][i]/2.+0.5))*self.params['oversample'], 
                    dvdrow=jacob['dvdrow']/self.params['oversample'],
                    dvdcol=jacob['dvdcol']/self.params['oversample'],
                    dudrow=jacob['dudrow']/self.params['oversample'],
                    dudcol=jacob['dudcol']/self.params['oversample']) 
            elif self.params['oversample']==4:
                psf_jacob2=Jacobian(
                    row=63.5 + (m['orig_row'][i][j]-m['orig_start_row'][i][j]+1-(m['box_size'][i]/2.+0.5))*self.params['oversample'],
                    col=63.5 + (m['orig_col'][i][j]-m['orig_start_col'][i][j]+1-(m['box_size'][i]/2.+0.5))*self.params['oversample'], 
                    dvdrow=jacob['dvdrow']/self.params['oversample'],
                    dvdcol=jacob['dvdcol']/self.params['oversample'],
                    dudrow=jacob['dudrow']/self.params['oversample'],
                    dudcol=jacob['dudcol']/self.params['oversample']) 


            # Create an obs for each cutout
            mask = np.where(weight!=0)
            if 1.*len(weight[mask])/np.product(np.shape(weight))<0.8:
                continue

            w.append(np.mean(weight[mask]))
            noise = np.ones_like(weight)/w[-1]

            psf_obs = Observation(im_psf, jacobian=gal_jacob, meta={'offset_pixels':None,'file_id':None})
            psf_obs2 = Observation(im_psf2, jacobian=psf_jacob2, meta={'offset_pixels':None,'file_id':None})
            #obs = Observation(im, weight=weight, jacobian=gal_jacob, psf=psf_obs, meta={'offset_pixels':None,'file_id':None})
            # oversampled PSF
            obs = Observation(im, weight=weight, jacobian=gal_jacob, psf=psf_obs2, meta={'offset_pixels':None,'file_id':None})
            obs.set_noise(noise)
            # obs.set_noise(noise.array)

            obs_list.append(obs)
            psf_list.append(psf_obs2)
            included.append(j)

        return obs_list,psf_list,np.array(included)-1,np.array(w)

    def get_snr(self,obs_list,res,res_full):

        if res_full['flags']!=0:
            return -1

        size = res['pars'][4]
        flux = res['flux']

        model_ = galsim.Sersic(1, half_light_radius=1.*size, flux=flux*(1.-res['pars'][5])) + galsim.Sersic(4, half_light_radius=1.*size, flux=flux*res['pars'][5])
        for i in range(len(obs_list)):
            obs = obs_list[i]
            im = obs.psf.image.copy()
            im *= 1.0/im.sum()/len(obs_list)
            psf_gsimage = galsim.Image(im,wcs=obs.psf.jacobian.get_galsim_wcs())
            psf_ii = galsim.InterpolatedImage(psf_gsimage,x_interpolant='lanczos15')

            model = galsim.Convolve(model_,psf_ii)
            gal_stamp = galsim.Image(np.shape(obs.image)[0],np.shape(obs.image)[1], wcs=obs.jacobian.get_galsim_wcs())

            model.drawImage(image=gal_stamp)
            if i==0:
                image = gal_stamp.array*obs.weight
            else:
                image += gal_stamp.array*obs.weight

        return image.sum()

    def measure_shape_mof(self,obs_list,T,flux=1000.0,fracdev=None,use_e=None,model='exp'):
        # model in exp, bdf

        pix_range = galsim.roman.pixel_scale/10.
        e_range = 0.1
        fdev = 1.
        def pixe_guess(n):
            return 2.*n*np.random.random() - n

        # possible models are 'exp','dev','bdf' galsim.wfirst.pixel_scale
        cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.roman.pixel_scale, galsim.roman.pixel_scale)
        gp = ngmix.priors.GPriorBA(0.3)
        hlrp = ngmix.priors.FlatPrior(1.0e-4, 1.0e2)
        fracdevp = ngmix.priors.Normal(0.5, 0.1, bounds=[0., 1.])
        fluxp = ngmix.priors.FlatPrior(0, 1.0e5) # not sure what lower bound should be in general

        # center1 + center2 + shape + hlr + fracdev + fluxes for each object
        # guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,0.5+pixe_guess(fdev),100.])
        if model=='bdf':
            if fracdev is None:
                fracdev = pixe_guess(fdev)
            if use_e is None:
                e1 = pixe_guess(e_range)
                e2 = pixe_guess(e_range)
            else:
                e1 = use_e[0]
                e2 = use_e[1]
            prior = joint_prior.PriorBDFSep(cp, gp, hlrp, fracdevp, fluxp)
            guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),e1,e2,T,fracdev,flux])
        elif model=='exp':
            prior = joint_prior.PriorSimpleSep(cp, gp, hlrp, fluxp)
            guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,500.])
        else:
            raise ParamError('Bad model choice.')

        if not self.params['avg_fit']:
            multi_obs_list=MultiBandObsList()
            multi_obs_list.append(obs_list)

            fitter = mof.GSMOF([multi_obs_list], model, prior)
            # center1 + center2 + shape + hlr + fracdev + fluxes for each object
            # guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,0.5+pixe_guess(fdev),100.])
            fitter.go(guess)

            res_ = fitter.get_object_result(0)
            res_full_  = fitter.get_result()
            if model=='exp':
                res_['flux'] = res_['pars'][5]
            else:
                res_['flux'] = res_['pars'][6]

            res_['s2n_r'] = self.get_snr(obs_list,res_,res_full_)

            return res_,res_full_

        else:

            out = []
            out_obj = []
            for i in range(len(obs_list)):
                multi_obs_list = MultiBandObsList()
                tmp_obs_list = ObsList()
                tmp_obs_list.append(obs_list[i])
                multi_obs_list.append(tmp_obs_list)

                fitter = mof.KGSMOF([multi_obs_list], 'bdf', prior)
                # center1 + center2 + shape + hlr + fracdev + fluxes for each object
                # guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,0.5+pixe_guess(fdev),100.])
                guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,0.5+pixe_guess(fdev),100.])
                fitter.go(guess)

                out_obj.append(fitter.get_object_result(0))
                out.append(fitter.get_result())

            return out_obj,out

    def measure_shape_gmix(self,obs_list,T,flux=1000.0,fracdev=None,use_e=None,model='exp'):
        # model in exp, bdf

        pix_range = galsim.roman.pixel_scale/10.
        e_range = 0.1
        fdev = 1.
        def pixe_guess(n):
            return 2.*n*np.random.random() - n

        # possible models are 'exp','dev','bdf' galsim.wfirst.pixel_scale
        cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.roman.pixel_scale, galsim.roman.pixel_scale)
        gp = ngmix.priors.GPriorBA(0.3)
        hlrp = ngmix.priors.FlatPrior(1.0e-4, 1.0e2)
        fracdevp = ngmix.priors.Normal(0.5, 0.1, bounds=[0., 1.])
        fluxp = ngmix.priors.FlatPrior(0, 1.0e5) # not sure what lower bound should be in general

        # center1 + center2 + shape + hlr + fracdev + fluxes for each object
        # guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,0.5+pixe_guess(fdev),100.])
        if model=='bdf':
            if fracdev is None:
                fracdev = pixe_guess(fdev)
            if use_e is None:
                e1 = pixe_guess(e_range)
                e2 = pixe_guess(e_range)
            else:
                e1 = use_e[0]
                e2 = use_e[1]
            prior = joint_prior.PriorBDFSep(cp, gp, hlrp, fracdevp, fluxp)
            guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),e1,e2,T,fracdev,flux])
        elif model=='exp':
            prior = joint_prior.PriorSimpleSep(cp, gp, hlrp, fluxp)
            guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,500.])
        else:
            raise ParamError('Bad model choice.')

            multi_obs_list=MultiBandObsList()
            multi_obs_list.append(obs_list)

            fitter = mof.GSMOF([multi_obs_list], model, prior)
            # center1 + center2 + shape + hlr + fracdev + fluxes for each object
            # guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,0.5+pixe_guess(fdev),100.])
            fitter.go(guess)

            res_ = fitter.get_object_result(0)
            res_full_  = fitter.get_result()
            if model=='exp':
                res_['flux'] = res_['pars'][5]
            else:
                res_['flux'] = res_['pars'][6]

            res_['s2n_r'] = self.get_snr(obs_list,res_,res_full_)

            return res_,res_full_

    def measure_shape_ngmix(self,obs_list,T,flux=1000.0,model='exp'):

        pix_range = galsim.roman.pixel_scale/10.
        e_range = 0.1
        fdev = 1.
        def pixe_guess(n):
            return 2.*n*np.random.random() - n

        # possible models are 'exp','dev','bdf' galsim.wfirst.pixel_scale
        cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.roman.pixel_scale, galsim.roman.pixel_scale)
        gp = ngmix.priors.GPriorBA(0.3)
        hlrp = ngmix.priors.FlatPrior(1.0e-4, 1.0e2)
        fracdevp = ngmix.priors.TruncatedGaussian(0.5, 0.5, -0.5, 1.5)
        fluxp = ngmix.priors.FlatPrior(0, 1.0e5) # not sure what lower bound should be in general

        prior = joint_prior.PriorBDFSep(cp, gp, hlrp, fracdevp, fluxp)
        # center1 + center2 + shape + hlr + fracdev + fluxes for each object
        # guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,0.5+pixe_guess(fdev),100.])
        guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,pixe_guess(fdev),300.])

        if not self.params['avg_fit']:

            guesser           = R50FluxGuesser(T,flux)
            ntry              = 5
            runner            = GalsimRunner(obs_list,model,guesser=guesser)
            runner.go(ntry=ntry)
            fitter            = runner.get_fitter()

            res_ = fitter.get_result()
            if model=='exp':
                res_['flux'] = res_['pars'][5]
            else:
                res_['flux'] = res_['pars'][6]

            return res_,res_

        else:

            out = []
            out_obj = []
            for i in range(len(obs_list)):

                tmp_obs_list = ObsList()
                tmp_obs_list.append(obs_list[i])
                guesser           = R50FluxGuesser(T,flux)
                ntry              = 5
                runner            = GalsimRunner(tmp_obs_list,model,guesser=guesser)
                runner.go(ntry=ntry)
                fitter            = runner.get_fitter()
                out.append(fitter.get_result())
                out_obj.append(fitter.get_result())

            return out_obj,out

    def measure_shape_metacal(self, obs_list, T, method='bootstrap', flux_=1000.0, fracdev=None, use_e=None):
        if method=='ngmix_fitter':
            mcal_keys=['noshear', '1p', '1m', '2p', '2m']
            obsdict = ngmix.metacal.get_all_metacal(obs_list, psf='gauss')
            results_metacal = {}
            for key in mcal_keys:
                mobs = obsdict[key]
                res_= self.measure_shape_ngmix(mobs,flux_)
                results_metacal[key] = res_
            return results_metacal

        elif method=='bootstrap':
            metacal_pars = {'types': ['noshear', '1p', '1m', '2p', '2m'], 'psf': 'gauss'}
            #T = self.hlr
            pix_range = old_div(galsim.roman.pixel_scale,10.)
            e_range = 0.1
            fdev = 1.
            def pixe_guess(n):
                return 2.*n*np.random.random() - n

            cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.roman.pixel_scale, galsim.roman.pixel_scale)
            gp = ngmix.priors.GPriorBA(0.3)
            hlrp = ngmix.priors.FlatPrior(1.0e-4, 1.0e2) # -> need to be the same units as Tguess, which has arcsec^2
            fracdevp = ngmix.priors.Normal(0.5, 0.1, bounds=[0., 1.])
            fluxp = ngmix.priors.FlatPrior(0, 1.0e5)

            prior = joint_prior.PriorSimpleSep(cp, gp, hlrp, fluxp)
            guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,500.])

            boot = ngmix.bootstrap.MaxMetacalBootstrapper(obs_list, use_noise_image=True)
            psf_model = "gauss"
            gal_model = "gauss"

            lm_pars={'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
            max_pars={'method': 'lm', 'lm_pars':lm_pars}

            Tguess=(0.178 / 2.35482)**2 * 2. # arcsec^2
            ntry=2
            try:
                boot.fit_metacal(psf_model, gal_model, max_pars, Tguess, prior=prior, ntry=ntry, metacal_pars=metacal_pars) 
                res_ = boot.get_metacal_result()
            except (ngmix.gexceptions.BootGalFailure, ngmix.gexceptions.BootPSFFailure):
                print('Fit failed')
                res_ = 0
            return res_

        elif method=='multiband':
            metacal_pars = {'types': ['noshear', '1p', '1m', '2p', '2m'], 'psf': 'gauss'}
            #T = self.hlr
            pix_range = old_div(galsim.roman.pixel_scale,10.)
            e_range = 0.1
            fdev = 1.
            def pixe_guess(n):
                return 2.*n*np.random.random() - n

            cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.roman.pixel_scale, galsim.roman.pixel_scale)
            gp = ngmix.priors.GPriorBA(0.3)
            hlrp = ngmix.priors.FlatPrior(1.0e-5, 1.0e4)
            fracdevp = ngmix.priors.Normal(0.5, 0.1, bounds=[0., 1.])
            fluxp = [ngmix.priors.FlatPrior(0, 1.0e6),ngmix.priors.FlatPrior(0, 1.0e6),ngmix.priors.FlatPrior(0, 1.0e6)]

            prior = joint_prior.PriorSimpleSep(cp, gp, hlrp, fluxp)
            guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,500.])

            boot = ngmix.bootstrap.MaxMetacalBootstrapper(obs_list, use_noise_image=True)
            psf_model = "gauss"
            gal_model = "gauss"

            lm_pars={'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
            max_pars={'method': 'lm', 'lm_pars':lm_pars}

            Tguess=(0.178 / 2.35482)**2 * 2. #T**2/(2*np.log(2))
            ntry=2
            try:
                boot.fit_metacal(psf_model, gal_model, max_pars, Tguess, prior=prior, ntry=ntry, metacal_pars=metacal_pars) 
                res_ = boot.get_metacal_result()
            except (ngmix.gexceptions.BootGalFailure, ngmix.gexceptions.BootPSFFailure):
                print('multiband fit failed')
                res_ = 0

            return res_


    def make_jacobian(self,dudx,dudy,dvdx,dvdy,x,y):
        j = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)
        return j.withOrigin(galsim.PositionD(x,y))

    def measure_psf_shape(self,obs_list,T_guess=0.16):
        # doesn't work

        def make_ngmix_prior(T, pixel_scale):

            # centroid is 1 pixel gaussian in each direction
            cen_prior=priors.CenPrior(0.0, 0.0, pixel_scale, pixel_scale)

            # g is Bernstein & Armstrong prior with sigma = 0.1
            gprior=priors.GPriorBA(0.1)

            # T is log normal with width 0.2
            Tprior=priors.LogNormal(T, 0.05)

            # flux is the only uninformative prior
            Fprior=priors.FlatPrior(-10.0, 1e3)

            prior=joint_prior.PriorSimpleSep(cen_prior, gprior, Tprior, Fprior)
            return prior

        T_guess = (T_guess / 2.35482)**2 * 2.

        cnt = dx = dy = e1 = e2 = T = flux = 0
        for ipsf,psf in enumerate(obs_list):

            # try:

            obs = ngmix.Observation(image=psf.psf.image, jacobian=psf.psf.jacobian)

            lm_pars = {'maxfev':4000}
            wcs = self.make_jacobian(psf.psf.jacobian.dudcol,
                                    psf.psf.jacobian.dudrow,
                                    psf.psf.jacobian.dvdcol,
                                    psf.psf.jacobian.dvdrow,
                                    psf.psf.jacobian.col0,
                                    psf.psf.jacobian.row0)
            prior = make_ngmix_prior(T_guess, wcs.minLinearScale())
            runner=PSFRunner(obs, 'gauss', T_guess, lm_pars, prior=prior)
            runner.go(ntry=5)

            flag = runner.fitter.get_result()['flags']
            gmix = runner.fitter.get_gmix()

            # except Exception as e:
            #     print 'exception'
            #     cnt+=1
            #     continue

            if flag != 0:
                print('flag',flag)
                cnt+=1
                continue

            e1_, e2_, T_ = gmix.get_g1g2T()
            dx_, dy_ = gmix.get_cen()
            if (np.abs(e1_) > 0.5) or (np.abs(e2_) > 0.5) or (dx_**2 + dy_**2 > MAX_CENTROID_SHIFT**2):
                print('g,xy',e1_,e2_,dx_,dy_)
                cnt+=1
                continue

            flux_ = gmix.get_flux() / wcs.pixelArea()

            dx   += dx_
            dy   += dy_
            e1   += e1_
            e2   += e2_
            T    += T_
            flux += flux_

        if cnt == len(obs_list):
            return None

        return cnt, dx/(len(obs_list)-cnt), dy/(len(obs_list)-cnt), e1/(len(obs_list)-cnt), e2/(len(obs_list)-cnt), T/(len(obs_list)-cnt), flux/(len(obs_list)-cnt)

    def measure_psf_shape_moments(self,obs_list,method='coadd'):

        BAD_MEASUREMENT = 1
        CENTROID_SHIFT  = 2
        MAX_CENTROID_SHIFT = 1.

        def make_psf_image(self,obs,method):
            if method == "single":
                wcs = self.make_jacobian(obs.jacobian.dudcol,
                                        obs.jacobian.dudrow,
                                        obs.jacobian.dvdcol,
                                        obs.jacobian.dvdrow,
                                        obs.jacobian.col0,
                                        obs.jacobian.row0)
                return galsim.Image(obs.image, xmin=1, ymin=1, wcs=wcs)
            elif method == "coadd":
                wcs = self.make_jacobian(obs.jacobian.dudcol,
                                        obs.jacobian.dudrow,
                                        obs.jacobian.dvdcol,
                                        obs.jacobian.dvdrow,
                                        obs.jacobian.col0,
                                        obs.jacobian.row0)
                return galsim.Image(obs.image, xmin=1, ymin=1, wcs=wcs)
            elif method == "multiband":
                wcs = self.make_jacobian(obs.jacobian.dudcol,
                                        obs.jacobian.dudrow,
                                        obs.jacobian.dvdcol,
                                        obs.jacobian.dvdrow,
                                        obs.jacobian.col0,
                                        obs.jacobian.row0)
                return galsim.Image(obs.image, xmin=1, ymin=1, wcs=wcs)

        out = np.zeros(len(obs_list),dtype=[('e1','f4')]+[('e2','f4')]+[('T','f4')]+[('dx','f4')]+[('dy','f4')]+[('flag','i2')])
        for iobs,obs in enumerate(obs_list):
            M = e1 = e2= 0
            im = make_psf_image(self,obs,method)

            try:
                shape_data = im.FindAdaptiveMom(weight=None, strict=False)
            except:
                out['flag'][iobs] |= BAD_MEASUREMENT
                continue

            if shape_data.moments_status != 0:
                out['flag'][iobs] |= BAD_MEASUREMENT
                continue

            out['dx'][iobs] = shape_data.moments_centroid.x - im.true_center.x
            out['dy'][iobs] = shape_data.moments_centroid.y - im.true_center.y
            if out['dx'][iobs]**2 + out['dy'][iobs]**2 > MAX_CENTROID_SHIFT**2:
                out['flag'][iobs] |= CENTROID_SHIFT
                continue

            # Account for the image wcs
            if im.wcs.isPixelScale():
                out['e1'][iobs] = shape_data.observed_shape.g1
                out['e2'][iobs] = shape_data.observed_shape.g2
                out['T'][iobs]  = 2 * shape_data.moments_sigma**2 * im.scale**2
            else:
                e1    = shape_data.observed_shape.e1
                e2    = shape_data.observed_shape.e2
                s     = shape_data.moments_sigma
                jac   = im.wcs.jacobian(im.true_center)
                M     = np.matrix( [[ 1 + e1, e2 ], [ e2, 1 - e1 ]] ) * s*s
                J     = jac.getMatrix()
                M     = J * M * J.T
                scale = np.sqrt(M/2./s/s)
                e1    = (M[0,0] - M[1,1]) / (M[0,0] + M[1,1])
                e2    = (2.*M[0,1]) / (M[0,0] + M[1,1])
                shear = galsim.Shear(e1=e1, e2=e2)
                out['T'][iobs]  = M[0,0] + M[1,1]
                out['e1'][iobs] = shear.g1
                out['e2'][iobs] = shear.g2

        return out

    # Let's make this function a general way to measure shapes with metacalibration. No creating coadds on the fly. 
    def get_coadd_shape_mcal(self):

        def get_flux(obs_list):
            flux = 0.
            for obs in obs_list:
                flux += obs.image.sum()
            flux /= len(obs_list)
            if flux<0:
                flux = 10.
            return flux

        print('mpi check 2',self.rank,self.size)

        filename = get_filename(self.params['out_path'],
                                'truth',
                                self.params['output_truth'],
                                name2='truth_gal',
                                overwrite=False)
        truth = fio.FITS(filename)[-1]
        m  = meds.MEDS(self.local_meds)
        #m2 = fio.FITS(self.local_meds_psf)
        if self.shape_iter is not None:
            indices = np.array_split(np.arange(len(m['number'][:])),self.shape_cnt)[self.shape_iter]
        else:
            indices = np.arange(len(m['number'][:]))

        print('rank in coadd_shape', self.rank)
        
        metacal_keys=['noshear', '1p', '1m', '2p', '2m']
        res_noshear=np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),])#('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_1p=np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),])#('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_1m=np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),])#('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_2p=np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),])#('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_2m=np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),])#('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_tot=[res_noshear, res_1p, res_1m, res_2p, res_2m]

        for i,ii in enumerate(indices):
            if i%self.size!=self.rank:
                continue
            if i%100==0:
                print('made it to object',i)
            try_save = False

            ind = m['number'][ii]
            t   = truth[ind]

            sca_list = m[ii]['sca']
            roman_psfs = self.simulate_roman_psfs()
            if not self.params['multiband']: 
                m2 = [roman_psfs[self.filter_][j-1] for j in sca_list[:m['ncutout'][i]]] ## first entry is taken care by the first function in get_exp_list. 
            else:
                print('Need some work on the multiband case. ')
                sys.exit()
            obs_list,psf_list,included,w = self.get_exp_list(m,ii,m2=m2,size=t['size']) # -> work on how to create the exposure list for the new version of sim.
            if len(included)==0:
                continue

            if self.params['shape_code']=='mof':
                res_,res_full_      = self.measure_shape_mof(obs_list,t['size'],flux=get_flux(obs_list),fracdev=t['bflux'],use_e=[t['int_e1'],t['int_e2']],model=self.params['ngmix_model'])
            elif self.params['shape_code']=='ngmix':
                res_,res_full_      = self.measure_shape_ngmix(obs_list,t['size'],model=self.params['ngmix_model'])
            elif self.params['shape_code']=='metacal':
                res_ = self.measure_shape_metacal(obs_list, t['size'], method='bootstrap', flux_=get_flux(obs_list), fracdev=t['bflux'],use_e=[t['int_e1'],t['int_e2']])
            else:
                raise ParamError('unknown shape code request')
            
            # Require new flagging system. 
            for k in metacal_keys:
                if res_[k]['flags'] !=0:
                    print('failed',i,ii,get_flux(obs_list))

            iteration=0
            for key in metacal_keys:
                res_tot[iteration]['ind'][i]                       = ind
                res_tot[iteration]['ra'][i]                        = t['ra']
                res_tot[iteration]['dec'][i]                       = t['dec']
                res_tot[iteration]['nexp_tot'][i]                  = m['ncutout'][ii]-1
                res_tot[iteration]['stamp'][i]                     = m['box_size'][ii]
                res_tot[iteration]['g1'][i]                        = t['g1']
                res_tot[iteration]['g2'][i]                        = t['g2']
                res_tot[iteration]['int_e1'][i]                    = t['int_e1']
                res_tot[iteration]['int_e2'][i]                    = t['int_e2']
                res_tot[iteration]['rot'][i]                       = t['rot']
                res_tot[iteration]['size'][i]                      = t['size']
                res_tot[iteration]['redshift'][i]                  = t['z']
                res_tot[iteration]['mag_'+self.pointing.filter][i] = t[self.pointing.filter]
                res_tot[iteration]['pind'][i]                      = t['pind']
                res_tot[iteration]['bulge_flux'][i]                = t['bflux']
                res_tot[iteration]['disk_flux'][i]                 = t['dflux']

                res_tot[iteration]['nexp_used'][i]                 = len(included)
                res_tot[iteration]['flags'][i]                     = res_[key]['flags']
                if res_==0:
                    res_tot[iteration]['flags'][i]                     = 2 # flag 2 means the object didnt pass shape fit. 
                if res_[key]['flags']==0:
                    res_tot[iteration]['px'][i]                        = res_[key]['pars'][0]
                    res_tot[iteration]['py'][i]                        = res_[key]['pars'][1]
                    res_tot[iteration]['flux'][i]                      = res_[key]['flux']
                    res_tot[iteration]['snr'][i]                       = res_[key]['s2n_r']
                    res_tot[iteration]['e1'][i]                        = res_[key]['pars'][2]
                    res_tot[iteration]['e2'][i]                        = res_[key]['pars'][3]
                    res_tot[iteration]['cov_11'][i]                    = res_[key]['pars_cov'][2,2]
                    res_tot[iteration]['cov_22'][i]                    = res_[key]['pars_cov'][3,3]
                    res_tot[iteration]['cov_12'][i]                    = res_[key]['pars_cov'][2,3]
                    res_tot[iteration]['cov_21'][i]                    = res_[key]['pars_cov'][3,2]
                    res_tot[iteration]['hlr'][i]                       = res_[key]['pars'][4]
                    res_tot[iteration]['psf_e1'][i]                    = res_[key]['gpsf'][0]
                    res_tot[iteration]['psf_e2'][i]                    = res_[key]['gpsf'][1]
                    res_tot[iteration]['psf_T'][i]                     = res_[key]['Tpsf']
                iteration+=1
        # end of metacal key loop. 
        m.close()

        print('done measuring',self.rank)

        self.comm.Barrier()
        print('after first barrier')

        for j in range(5):
            if self.rank==0:
                for i in range(1,self.size):
                    print('getting',i)
                    tmp_res   = self.comm.recv(source=i)
                    mask      = tmp_res['size']!=0
                    res_tot[j][mask] = tmp_res[mask]
                    # coadd.update(self.comm.recv(source=i))

                print('before barrier',self.rank)
                self.comm.Barrier()
                # print coadd.keys()
                res = res_tot[j][np.argsort(res_tot[j]['ind'])]
                res['ra'] = np.degrees(res['ra'])
                res['dec'] = np.degrees(res['dec'])
                if self.shape_iter is None:
                    ilabel = 0
                else:
                    ilabel = self.shape_iter
                filename = get_filename(self.params['out_path'],
                                        self.params['out_dir'],
                                        self.params['output_meds'],
                                        var=self.pointing.filter+'_'+str(self.pix)+'_'+str(ilabel)+'_mcal_'+str(metacal_keys[j]),
                                        ftype='fits',
                                        overwrite=True)
                fio.write(filename,res)

            else:
                self.comm.send(res_tot[j], dest=0)
                print('before barrier',self.rank)
                self.comm.Barrier()

    def make_psc_coadd_and_get_shape_mcal(self):

        def get_flux(obs_list):
            flux = 0.
            for obs in obs_list:
                flux += obs.image.sum()
            flux /= len(obs_list)
            if flux<0:
                flux = 10.
            return flux

        print('mpi check 2',self.rank,self.size)

        filename = get_filename(self.params['out_path'],
                                'truth',
                                self.params['output_truth'],
                                name2='truth_gal',
                                overwrite=False)
        truth = fio.FITS(filename)[-1]
        m  = meds.MEDS(self.local_meds)
        #m2 = fio.FITS(self.local_meds_psf)
        if self.shape_iter is not None:
            indices = np.array_split(np.arange(len(m['number'][:])),self.shape_cnt)[self.shape_iter]
        else:
            indices = np.arange(len(m['number'][:]))

        print('rank in coadd_shape', self.rank)

        metacal_keys=['noshear', '1p', '1m', '2p', '2m']
        res_noshear=np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('T',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_T',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_1p=np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('T',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_T',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_1m=np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('T',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_T',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_2p=np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('T',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_T',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_2m=np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('T',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_T',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_tot=[res_noshear, res_1p, res_1m, res_2p, res_2m]

        for i,ii in enumerate(indices):
            if i%self.size!=self.rank:
                continue
            if i%100==0:
                print('made it to object',i)
            try_save = False

            ind = m['number'][ii]
            t   = truth[ind]

            sca_list = m[ii]['sca']
            roman_psfs = self.simulate_roman_psfs()
            m2 = [roman_psfs[self.filter_][j-1] for j in sca_list[:m['ncutout'][i]]] ## first entry is taken care by the first function in get_exp_list. 
            m2_coadd = [roman_psfs[self.filter_][j-1] for j in sca_list[:m['ncutout'][i]]]
            
            if self.params['coadds']=='single': # measure shapes of single-epoch images. joint-fit. 
                obs_list,psf_list,included,w = self.get_exp_list(m,ii,m2=m2,size=t['size'])
            elif self.params['coadds']=='coadds': # measure shapes of coadd images. single-fit for the single-band case. 
                obs_list,psf_list,included,w = self.get_exp_list_coadd_with_noise_image(m,ii,m2=m2_coadd,size=t['size'])
            if len(included)==0:
                for f in range(5):
                    res_tot[f]['flags'][i] = 5
                continue
            
            if self.params['coadds']=='coadds':
                cdpsf_list = ObsList()
                coadd            = psc.Coadder(obs_list,flat_wcs=True).coadd_obs
                coadd.psf.image[coadd.psf.image<0] = 0 # set negative pixels to zero. 
                coadd.set_meta({'offset_pixels':None,'file_id':None})
                ### when doing oversampling ###
                if self.params['oversample'] == 4:
                    
                    psf_wcs = self.make_jacobian(coadd.psf.jacobian.dudcol,
                                                 coadd.psf.jacobian.dudrow,
                                                 coadd.psf.jacobian.dvdcol,
                                                 coadd.psf.jacobian.dvdrow,
                                                 coadd.psf.jacobian.col0,
                                                 coadd.psf.jacobian.row0)
                    gal_wcs =self.make_jacobian(coadd.jacobian.dudcol,
                                                coadd.jacobian.dudrow,
                                                coadd.jacobian.dvdcol,
                                                coadd.jacobian.dvdrow,
                                                coadd.jacobian.col0,
                                                coadd.jacobian.row0)
                    # To downsample the coadded oversampled PSF, we need to subsample every 4th (since the sampling factor is 4) pixel, not sum 4x4 block. 
                    # Since sampling from the first pixel might be anisotropic, 
                    # we should test with sampling different pixels like 1::4, 2::4, 3::4 to make sure this does not cause any sampling bias.)
                    subsampled_image_array = coadd.psf.image[3::4,3::4]
                    new_coadd_psf_jacob = Jacobian( row=15.5, #(coadd.psf.jacobian.row0/self.params['oversample']),
                                                    col=15.5, #(coadd.psf.jacobian.col0/self.params['oversample']), 
                                                    dvdrow=(coadd.psf.jacobian.dvdrow*self.params['oversample']),
                                                    dvdcol=(coadd.psf.jacobian.dvdcol*self.params['oversample']),
                                                    dudrow=(coadd.psf.jacobian.dudrow*self.params['oversample']),
                                                    dudcol=(coadd.psf.jacobian.dudcol*self.params['oversample']))
                    coadd_psf_obs = Observation(subsampled_image_array, jacobian=new_coadd_psf_jacob, meta={'offset_pixels':None,'file_id':None})

                    # Instead of subsampling every 4th pixel, we can treat the oversampled PSF as a surface brightness profile with interpolatedimage, and draw from the image.
                    # subsampled_coadd_psf = galsim.InterpolatedImage(galsim.Image(coadd.psf.image, wcs=psf_wcs))
                    # im_psf = galsim.Image(32, 32, wcs=gal_wcs)
                    # subsampled_coadd_psf.drawImage(im_psf, method='no_pixel')
                    # subsampled_image_array = im_psf.array
                    # coadd_psf_obs = Observation(subsampled_image_array, jacobian=coadd.jacobian, meta={'offset_pixels':None,'file_id':None})
                    coadd.psf = coadd_psf_obs

                    # For moments measurement of the PSF.
                    cdpsf_list.append(coadd_psf_obs)
                    if i == 100:
                        out = self.measure_psf_shape_moments(cdpsf_list, method='coadd')
                        mask = (out['flag']==0)
                        out = out[mask]
                        print('psf measurement', out['e1'], out['e2'], out['T'])
                elif self.params['oversample'] == 1:
                    cdpsf_list.append(coadd.psf)
            
            if self.params['shape_code']=='mof':
                res_,res_full_      = self.measure_shape_mof(obs_list,t['size'],flux=get_flux(obs_list),fracdev=t['bflux'],use_e=[t['int_e1'],t['int_e2']],model=self.params['ngmix_model'])
            elif self.params['shape_code']=='ngmix':
                res_,res_full_      = self.measure_shape_ngmix(obs_list,t['size'],model=self.params['ngmix_model'])
            elif self.params['shape_code']=='metacal':
                if self.params['coadds']=='single':
                    res_ = self.measure_shape_metacal(obs_list, t['size'], method='bootstrap', flux_=get_flux(obs_list), fracdev=t['bflux'],use_e=[t['int_e1'],t['int_e2']])
                    out = self.measure_psf_shape_moments(psf_list, method='single')
                    mask = (out['flag']==0)
                    out = out[mask]
                    w = w[mask]              
                elif self.params['coadds']=='coadds':
                    obs_list = ObsList()
                    obs_list.append(coadd)
                    res_ = self.measure_shape_metacal(obs_list, t['size'], method='bootstrap', flux_=get_flux(obs_list), fracdev=t['bflux'],use_e=[t['int_e1'],t['int_e2']])
                    out = self.measure_psf_shape_moments(cdpsf_list, method='coadd')
                    mask = (out['flag']==0)
                    out = out[mask]
            else:
                raise ParamError('unknown shape code request')
                                    
            for k in metacal_keys:
                if self.params['coadds']=='single':
                    if res_[k]['flags'] !=0:
                        print('failed',i,ii,get_flux(obs_list))

            wcs = self.make_jacobian(obs_list[0].jacobian.dudcol,
                                    obs_list[0].jacobian.dudrow,
                                    obs_list[0].jacobian.dvdcol,
                                    obs_list[0].jacobian.dvdrow,
                                    obs_list[0].jacobian.col0,
                                    obs_list[0].jacobian.row0)

            iteration=0
            for key in metacal_keys:
                res_tot[iteration]['ind'][i]                       = ind
                res_tot[iteration]['ra'][i]                        = t['ra']
                res_tot[iteration]['dec'][i]                       = t['dec']
                res_tot[iteration]['nexp_tot'][i]                  = m['ncutout'][ii]-1
                res_tot[iteration]['stamp'][i]                     = m['box_size'][ii]
                res_tot[iteration]['g1'][i]                        = t['g1']
                res_tot[iteration]['g2'][i]                        = t['g2']
                res_tot[iteration]['int_e1'][i]                    = t['int_e1']
                res_tot[iteration]['int_e2'][i]                    = t['int_e2']
                res_tot[iteration]['rot'][i]                       = t['rot']
                res_tot[iteration]['size'][i]                      = t['size']
                res_tot[iteration]['redshift'][i]                  = t['z']
                res_tot[iteration]['mag_'+self.pointing.filter][i] = t[self.pointing.filter]
                res_tot[iteration]['pind'][i]                      = t['pind']
                res_tot[iteration]['bulge_flux'][i]                = t['bflux']
                res_tot[iteration]['disk_flux'][i]                 = t['dflux']
                res_tot[iteration]['nexp_used'][i]                 = len(included)

                if self.params['coadds']=='single':
                    res_tot[iteration]['flags'][i]                     = res_[key]['flags']
                    if res_==0:
                        res_tot[iteration]['flags'][i]                     = 2 # flag 2 means the object didnt pass shape fit. 
                    if res_[key]['flags']==0:
                        res_tot[iteration]['px'][i]                        = res_[key]['pars'][0]
                        res_tot[iteration]['py'][i]                        = res_[key]['pars'][1]
                        res_tot[iteration]['flux'][i]                      = res_[key]['flux']
                        res_tot[iteration]['snr'][i]                       = res_[key]['s2n_r']
                        res_tot[iteration]['e1'][i]                        = res_[key]['pars'][2]
                        res_tot[iteration]['e2'][i]                        = res_[key]['pars'][3]
                        res_tot[iteration]['cov_11'][i]                    = res_[key]['pars_cov'][2,2]
                        res_tot[iteration]['cov_22'][i]                    = res_[key]['pars_cov'][3,3]
                        res_tot[iteration]['cov_12'][i]                    = res_[key]['pars_cov'][2,3]
                        res_tot[iteration]['cov_21'][i]                    = res_[key]['pars_cov'][3,2]
                        res_tot[iteration]['T'][i]                       = res_[key]['pars'][4]
                        
                        if len(out)!=0:
                            res_tot[iteration]['psf_e1'][i]        = np.average(out['e1'],weights=w)
                            res_tot[iteration]['psf_e2'][i]        = np.average(out['e2'],weights=w)
                            res_tot[iteration]['psf_T'][i]         = np.average(out['T'],weights=w)
                            res_tot[iteration]['psf_nexp_used'][i] = len(out)
                        else:
                            res_tot[iteration]['psf_e1'][i]        = -9999
                            res_tot[iteration]['psf_e2'][i]        = -9999
                            res_tot[iteration]['psf_T'][i]         = -9999
                elif self.params['coadds']=='coadds':
                    if res_[key]['flags']==0:
                        res_tot[iteration]['coadd_px'][i]                  = res_[key]['pars'][0]
                        res_tot[iteration]['coadd_py'][i]                  = res_[key]['pars'][1]
                        res_tot[iteration]['coadd_flux'][i]                = res_[key]['pars'][5] / wcs.pixelArea()
                        res_tot[iteration]['coadd_snr'][i]                 = res_[key]['s2n']
                        res_tot[iteration]['coadd_e1'][i]                  = res_[key]['pars'][2]
                        res_tot[iteration]['coadd_e2'][i]                  = res_[key]['pars'][3]
                        res_tot[iteration]['coadd_T'][i]                   = res_[key]['pars'][4]
                        # res_tot[iteration]['coadd_psf_e1'][i]              = res_[key]['gpsf'][0]
                        # res_tot[iteration]['coadd_psf_e2'][i]              = res_[key]['gpsf'][1]
                        # res_tot[iteration]['coadd_psf_T'][i]               = res_[key]['Tpsf']

                    if len(out)!=0:
                        res_tot[iteration]['coadd_psf_e1'][i]        = out['e1']
                        res_tot[iteration]['coadd_psf_e2'][i]        = out['e2']
                        res_tot[iteration]['coadd_psf_T'][i]         = out['T']
                    else:
                        res_tot[iteration]['coadd_psf_e1'][i]        = -9999
                        res_tot[iteration]['coadd_psf_e2'][i]        = -9999
                        res_tot[iteration]['coadd_psf_T'][i]         = -9999
                iteration+=1
            
        # end of metacal key loop. 
        m.close()

        print('done measuring',self.rank)

        self.comm.Barrier()
        print('after first barrier')

        for j in range(5):
            if self.rank==0:
                for i in range(1,self.size):
                    print('getting',i)
                    tmp_res   = self.comm.recv(source=i)
                    mask      = tmp_res['size']!=0
                    res_tot[j][mask] = tmp_res[mask]
                    # coadd.update(self.comm.recv(source=i))

                print('before barrier',self.rank)
                self.comm.Barrier()
                # print coadd.keys()
                res = res_tot[j][np.argsort(res_tot[j]['ind'])]
                res['ra'] = np.degrees(res['ra'])
                res['dec'] = np.degrees(res['dec'])
                if self.shape_iter is None:
                    ilabel = 0
                else:
                    ilabel = self.shape_iter
                filename = get_filename(self.params['out_path'],
                                        self.params['out_dir'],
                                        self.params['output_meds'],
                                        var=self.pointing.filter+'_'+str(self.pix)+'_'+str(ilabel)+'_mcal_'+str(metacal_keys[j]),
                                        ftype='fits',
                                        overwrite=True)
                fio.write(filename,res)

            else:

                self.comm.send(res_tot[j], dest=0)
                #self.comm.send(coadd, dest=0)
                #coadd = None
                print('before barrier',self.rank)
                self.comm.Barrier()

    def save_coadd_images(self):

        filename = get_filename(self.params['out_path'],
                                'truth',
                                self.params['output_truth'],
                                name2='truth_gal',
                                overwrite=False)
        truth = fio.FITS(filename)[-1]
        m  = meds.MEDS(self.local_meds)
        #m2 = fio.FITS(self.local_meds_psf)
        if self.shape_iter is not None:
            indices = np.array_split(np.arange(len(m['number'][:])),self.shape_cnt)[self.shape_iter]
        else:
            indices = np.arange(len(m['number'][:]))

        print('rank in coadd_shape', self.rank)

        # randomly select 50 objects for each meds file. -> This will end up in 24,000 objects in total for 480 meds files. -> a rate of 1 PSF per 1 arcmin x 1 arcmin. 
        rand_obj_list = np.random.choice(indices, size=50, replace=False)
        for i,ii in enumerate(rand_obj_list):
            
            ind = m['number'][ii]
            t   = truth[ind]
            sca_Hlist = m[ii]['sca'] # List of SCAs for the same object in multiple observations. 
            m2_coadd = [self.all_psfs[j-1] for j in sca_Hlist[:m['ncutout'][ii]]]

            obs_list,psf_list,included,w = self.get_exp_list_coadd_with_noise_image(m,ii,m2=m2_coadd,size=t['size'])
            res = np.zeros(1, dtype=[('ind', int), ('ra', float), ('dec', float), ('mag', float), ('nexp_tot', int)])
            res['ind']                       = ind
            res['ra']                        = t['ra']
            res['dec']                       = t['dec']
            res['mag']                       = t['H158']
            res['nexp_tot']                  = m['ncutout'][ii]-1

            # coadd images
            coadd_H            = psc.Coadder(obs_list,flat_wcs=True).coadd_obs
            coadd_H.psf.image[coadd_H.psf.image<0] = 0 # set negative pixels to zero. 
            coadd_H.set_meta({'offset_pixels':None,'file_id':None})

            print('writing out coadd files.')
            fits = fio.FITS('/hpc/group/cosmology/phy-lsst/public/psc_coadd_psf/fiducial_H158_oversampled/fiducial_H158_'+str(self.pix)+'_oversampled_'+str(ii)+'.fits','rw')
            # save coadd PSF
            fits.write(coadd_H.psf.image)
            # save single exposure PSF
            for exp in range(len(obs_list)):
                fits.write(obs_list[exp].psf.image)
            # save object info
            fits.write(res)
            fits.close()
        m.close()


    def make_psc_coadd_and_get_shape_mcal_multiband(self):

        def get_flux(obs_list):
            flux = 0.
            for obs in obs_list:
                flux += obs.image.sum()
            flux /= len(obs_list)
            if flux<0:
                flux = 10.
            return flux

        print('mpi check 2',self.rank,self.size)
        filename = get_filename(self.params['out_path'],
                                'truth',
                                self.params['output_truth'],
                                name2='truth_gal',
                                overwrite=False)
        truth = fio.FITS(filename)[-1]
        m_H158  = meds.MEDS(self.local_meds)
        m_J129  = meds.MEDS(self.local_Jmeds)
        m_F184  = meds.MEDS(self.local_Fmeds)
        if self.shape_iter is not None:
            indices_H = np.array_split(np.arange(len(m_H158['number'][:])),self.shape_cnt)[self.shape_iter]
            indices_J = np.array_split(np.arange(len(m_J129['number'][:])),self.shape_cnt)[self.shape_iter]
            indices_F = np.array_split(np.arange(len(m_F184['number'][:])),self.shape_cnt)[self.shape_iter]
        else:
            indices_H = np.arange(len(m_H158['number'][:]))
            indices_J = np.arange(len(m_J129['number'][:]))
            indices_F = np.arange(len(m_F184['number'][:]))

        print('rank in coadd_shape', self.rank)
 
        metacal_keys=['noshear', '1p', '1m', '2p', '2m']
        res_noshear=np.zeros(len(m_H158['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('T',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_T',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_1p=np.zeros(len(m_H158['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('T',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_T',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_1m=np.zeros(len(m_H158['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('T',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_T',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_2p=np.zeros(len(m_H158['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('T',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_T',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_2m=np.zeros(len(m_H158['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('T',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_T',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_tot=[res_noshear, res_1p, res_1m, res_2p, res_2m]

        for i,ii in enumerate(indices_H):
            if i%self.size!=self.rank:
                continue
            if i%100==0:
                print('made it to object',i)

            ind = m_H158['number'][ii]
            t   = truth[ind]

            ## use only objects that have 3 filters. check by galaxy ids.
            if (ind not in m_J129['number']) or (ind not in m_F184['number']):
                for f in range(5):
                    res_tot[f]['flags'][i]                     = 3 # flag 3 means the object does not have all 3 filters. 
                continue

            roman_psfs = self.simulate_roman_psfs()
            sca_Hlist = m_H158[ii]['sca'] # List of SCAs for the same object in multiple observations. 
            m2_H158_coadd = [roman_psfs[self.filter_][j-1] for j in sca_Hlist[:m_H158['ncutout'][ii]]]

            ii_J = m_J129[m_J129['number']==ind]['id'][0]
            sca_Jlist = m_J129[ii_J]['sca']
            m2_J129_coadd = [roman_psfs['J129'][j-1] for j in sca_Jlist[:m_J129['ncutout'][ii_J]]]

            ii_F = m_F184[m_F184['number']==ind]['id'][0]
            sca_Flist = m_F184[ii_F]['sca']
            m2_F184_coadd = [roman_psfs['F184'][j-1] for j in sca_Flist[:m_F184['ncutout'][ii_F]]]

            if self.params['coadds']=='single':
                obs_Hlist,psf_Hlist,included_H,w_H = self.get_exp_list(m_H158,ii,m2=m2_H158_coadd,size=t['size'])
                obs_Jlist,psf_Jlist,included_J,w_J = self.get_exp_list(m_J129,ii_J,m2=m2_J129_coadd,size=t['size'])
                obs_Flist,psf_Flist,included_F,w_F = self.get_exp_list(m_F184,ii_F,m2=m2_F184_coadd,size=t['size'])
                if len(obs_Hlist)==0 or len(obs_Jlist)==0 or len(obs_Flist)==0:
                    for f in range(5):
                        res_tot[f]['flags'][i]                     = 4 # flag 4 means the object masking is more than 20%.  
                    continue
            elif self.params['coadds']=='coadds':
                obs_Hlist,psf_Hlist,included_H,w_H = self.get_exp_list_coadd_with_noise_image(m_H158,ii,m2=m2_H158_coadd,size=t['size'])
                obs_Jlist,psf_Jlist,included_J,w_J = self.get_exp_list_coadd_with_noise_image(m_J129,ii_J,m2=m2_J129_coadd,size=t['size'])
                obs_Flist,psf_Flist,included_F,w_F = self.get_exp_list_coadd_with_noise_image(m_F184,ii_F,m2=m2_F184_coadd,size=t['size'])
                # check if masking is less than 20%
                if len(obs_Hlist)==0 or len(obs_Jlist)==0 or len(obs_Flist)==0:
                    for f in range(5):
                        res_tot[f]['flags'][i]                     = 4 # flag 4 means the object masking is more than 20%.  
                    continue
                coadd_F            = psc.Coadder(obs_Flist,flat_wcs=True).coadd_obs
                coadd_F.psf.image[coadd_F.psf.image<0] = 0 # set negative pixels to zero. 
                coadd_F.set_meta({'offset_pixels':None,'file_id':None})

                coadd_H            = psc.Coadder(obs_Hlist,flat_wcs=True).coadd_obs
                coadd_H.psf.image[coadd_H.psf.image<0] = 0 # set negative pixels to zero. 
                coadd_H.set_meta({'offset_pixels':None,'file_id':None})
                
                coadd_J            = psc.Coadder(obs_Jlist,flat_wcs=True).coadd_obs
                coadd_J.psf.image[coadd_J.psf.image<0] = 0 # set negative pixels to zero. 
                coadd_J.set_meta({'offset_pixels':None,'file_id':None})
                

            if len(included_H)==0:
                for f in range(5):
                    res_tot[f]['flags'][i] = 5 # flag 5 means no flux in the image. 
                continue

            if self.params['coadds']=='single':
                single_mb = [obs_Hlist, obs_Jlist, obs_Flist]
                mbpsf_list = psf_Hlist+psf_Jlist+psf_Flist
                mb_obs_list = MultiBandObsList()
                for band in range(3):
                    mb_obs_list.append(single_mb[band])

                wcs = self.make_jacobian(obs_Hlist[0].jacobian.dudcol,
                                        obs_Hlist[0].jacobian.dudrow,
                                        obs_Hlist[0].jacobian.dvdcol,
                                        obs_Hlist[0].jacobian.dvdrow,
                                        obs_Hlist[0].jacobian.col0,
                                        obs_Hlist[0].jacobian.row0)

            ### when doing oversampling ###
            if self.params['coadds']=='coadds':
                coadd = [coadd_H, coadd_J, coadd_F] 
                mb_obs_list = MultiBandObsList()
                mbpsf_list = ObsList()
                for band in range(3): 
                    obs_list = ObsList()
                    if self.params['oversample'] == 4: # This is probably wrong. Need to make a change when using oversampling PSF. 
                        new_coadd_psf_block = block_reduce(coadd[band].psf.image, block_size=(4,4), func=np.sum)
                        new_coadd_psf_jacob = Jacobian( row=15.5,
                                                        col=15.5, 
                                                        dvdrow=(coadd[band].psf.jacobian.dvdrow*self.params['oversample']),
                                                        dvdcol=(coadd[band].psf.jacobian.dvdcol*self.params['oversample']),
                                                        dudrow=(coadd[band].psf.jacobian.dudrow*self.params['oversample']),
                                                        dudcol=(coadd[band].psf.jacobian.dudcol*self.params['oversample']))
                        coadd_psf_obs = Observation(new_coadd_psf_block, jacobian=new_coadd_psf_jacob, meta={'offset_pixels':None,'file_id':None})
                        coadd[band].psf = coadd_psf_obs
                        mbpsf_list.append(coadd_psf_obs)
                    elif self.params['oversample'] == 1:
                        mbpsf_list.append(coadd[band].psf)
                    obs_list.append(coadd[band])
                    mb_obs_list.append(obs_list)

                wcs = self.make_jacobian(coadd_H.jacobian.dudcol,
                                        coadd_H.jacobian.dudrow,
                                        coadd_H.jacobian.dvdcol,
                                        coadd_H.jacobian.dvdrow,
                                        coadd_H.jacobian.col0,
                                        coadd_H.jacobian.row0)

            iteration=0
            for key in metacal_keys:
                res_tot[iteration]['ind'][i]                       = ind
                res_tot[iteration]['ra'][i]                        = t['ra']
                res_tot[iteration]['dec'][i]                       = t['dec']
                #res_tot[iteration]['nexp_tot'][i]                  = m['ncutout'][ii]-1
                #res_tot[iteration]['stamp'][i]                     = m['box_size'][ii]
                res_tot[iteration]['g1'][i]                        = t['g1']
                res_tot[iteration]['g2'][i]                        = t['g2']
                res_tot[iteration]['int_e1'][i]                    = t['int_e1']
                res_tot[iteration]['int_e2'][i]                    = t['int_e2']
                res_tot[iteration]['rot'][i]                       = t['rot']
                res_tot[iteration]['size'][i]                      = t['size']
                res_tot[iteration]['redshift'][i]                  = t['z']
                res_tot[iteration]['mag_'+self.pointing.filter][i] = t[self.pointing.filter]
                res_tot[iteration]['pind'][i]                      = t['pind']
                res_tot[iteration]['bulge_flux'][i]                = t['bflux']
                res_tot[iteration]['disk_flux'][i]                 = t['dflux']

                iteration+=1
                
            res_ = self.measure_shape_metacal(mb_obs_list, t['size'], method='multiband', flux_=1000.0, fracdev=t['bflux'],use_e=[t['int_e1'],t['int_e2']])
            out = self.measure_psf_shape_moments(mbpsf_list, method='multiband')
            mask = (out['flag']==0)
            out = out[mask]

            iteration=0
            for key in metacal_keys:
                if res_==0:
                    #res_tot[iteration]['ind'][i]                       = 0
                    res_tot[iteration]['flags'][i]                     = 2 # flag 2 means the object didnt pass shape fit. 
                elif res_[key]['flags']==0:
                    res_tot[iteration]['flags'][i]                     = res_[key]['flags']
                    res_tot[iteration]['coadd_px'][i]                  = res_[key]['pars'][0]
                    res_tot[iteration]['coadd_py'][i]                  = res_[key]['pars'][1]
                    res_tot[iteration]['coadd_flux'][i]                = res_[key]['pars'][5] / wcs.pixelArea()
                    res_tot[iteration]['coadd_snr'][i]                 = res_[key]['s2n']
                    res_tot[iteration]['coadd_e1'][i]                  = res_[key]['pars'][2]
                    res_tot[iteration]['coadd_e2'][i]                  = res_[key]['pars'][3]
                    res_tot[iteration]['coadd_T'][i]                 = res_[key]['pars'][4]
                    #res_tot[iteration]['coadd_psf_e1'][i]              = res_[key]['gpsf'][0]
                    #res_tot[iteration]['coadd_psf_e2'][i]              = res_[key]['gpsf'][1]
                    #res_tot[iteration]['coadd_psf_T'][i]               = res_[key]['Tpsf']

                if len(out)!=0:
                    res_tot[iteration]['coadd_psf_e1'][i]        = np.average(out['e1'])
                    res_tot[iteration]['coadd_psf_e2'][i]        = np.average(out['e2'])
                    res_tot[iteration]['coadd_psf_T'][i]         = np.average(out['T'])
                    res_tot[iteration]['psf_nexp_used'][i]       = len(out)
                else:
                    res_tot[iteration]['coadd_psf_e1'][i]        = -9999
                    res_tot[iteration]['coadd_psf_e2'][i]        = -9999
                    res_tot[iteration]['coadd_psf_T'][i]         = -9999
                iteration+=1
        # end of metacal key loop. 
        m_H158.close()
        m_J129.close()
        m_F184.close()

        print('done measuring',self.rank)

        self.comm.Barrier()
        print('after first barrier')

        for j in range(5):
            if self.rank==0:
                for i in range(1,self.size):
                    print('getting',i)
                    tmp_res   = self.comm.recv(source=i)
                    mask      = tmp_res['size']!=0
                    res_tot[j][mask] = tmp_res[mask]
                    # coadd.update(self.comm.recv(source=i))

                print('before barrier',self.rank)
                self.comm.Barrier()
                # print coadd.keys()
                res = res_tot[j][np.argsort(res_tot[j]['ind'])]
                res['ra'] = np.degrees(res['ra'])
                res['dec'] = np.degrees(res['dec'])
                if self.shape_iter is None:
                    ilabel = 0
                else:
                    ilabel = self.shape_iter
                filename = get_filename(self.params['out_path'],
                                        self.params['out_dir'],
                                        self.params['output_meds'],
                                        var=self.pointing.filter+'_'+str(self.pix)+'_'+str(ilabel)+'_mcal_multiband_'+str(metacal_keys[j]),
                                        ftype='fits',
                                        overwrite=True)
                fio.write(filename,res)

            else:

                self.comm.send(res_tot[j], dest=0)
                #self.comm.send(coadd, dest=0)
                #coadd = None
                print('before barrier',self.rank)
                self.comm.Barrier()

    def get_coadd_shape_drizzle(self):

        def get_flux(obs_list):
            flux = 0.
            for obs in obs_list:
                flux += obs.image.sum()
            flux /= len(obs_list)
            if flux<0:
                flux = 10.
            return flux

        print('mpi check 2',self.rank,self.size)
        
        with open(self.local_drizzle_cutout, 'rb') as f:
            f_cutouts = pickle.load(f)
        indices = f_cutouts.keys()

        print('rank in coadd_shape', self.rank)
        
        metacal_keys=['noshear', '1p', '1m', '2p', '2m']
        res_noshear=np.zeros(len(indices),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_1p=np.zeros(len(indices),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_1m=np.zeros(len(indices),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_2p=np.zeros(len(indices),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_2m=np.zeros(len(indices),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
        res_tot=[res_noshear, res_1p, res_1m, res_2p, res_2m]

        for i,ii in enumerate(indices):
            if i%self.size!=self.rank:
                continue
            if i%100==0:
                print('made it to object',i)
            try_save = False

            t = f_cutouts[ii]['object_data']
            m = f_cutouts[ii]['image_cutouts']
            m_noise = f_cutouts[ii]['noise_cutouts']
            m_weight = f_cutouts[ii]['weight_cutouts']
            m2 = f_cutouts[ii]['psf_cutouts']

            obs_list=ObsList()
            psf_list=ObsList()
            w        = []
            im = m
            im_psf = m2.array
            weight = m_weight
            if np.sum(im)==0.:
                print('no flux in image ',i)
                continue

            # Get a galaxy jacobian. 
            gal_jacob=Jacobian( x=t['cutout_x'],
                                y=t['cutout_y'],
                                dudx=t['dudx'],
                                dudy=t['dudy'],
                                dvdx=t['dvdx'],
                                dvdy=t['dvdy']) 

            psf_obs = Observation(im_psf, jacobian=gal_jacob, meta={'offset_pixels':None,'file_id':None})
            obs = Observation(im, weight=weight, jacobian=gal_jacob, psf=psf_obs, meta={'offset_pixels':None,'file_id':None})
            obs.set_noise(m_noise)
            # obs.set_noise(noise.array)
            obs_list.append(obs)
            psf_list.append(psf_obs)

            iteration=0
            for key in metacal_keys:
                res_tot[iteration]['ind'][i]                       = ii
                res_tot[iteration]['ra'][i]                        = t['ra']
                res_tot[iteration]['dec'][i]                       = t['dec']
                res_tot[iteration]['stamp'][i]                     = t['stamp']
                res_tot[iteration]['g1'][i]                        = t['g1']
                res_tot[iteration]['g2'][i]                        = t['g2']
                res_tot[iteration]['int_e1'][i]                    = t['int_e1']
                res_tot[iteration]['int_e2'][i]                    = t['int_e2']
                res_tot[iteration]['rot'][i]                       = t['rot']
                res_tot[iteration]['size'][i]                      = t['size']
                res_tot[iteration]['redshift'][i]                  = t['redshift']
                res_tot[iteration]['mag_'+self.pointing.filter][i] = t['mag']
                res_tot[iteration]['pind'][i]                      = t['pind']
                res_tot[iteration]['bulge_flux'][i]                = t['bulge_flux']
                res_tot[iteration]['disk_flux'][i]                 = t['disk_flux']

                iteration+=1

            wcs = self.make_jacobian(obs_list[0].jacobian.dudcol,
                                    obs_list[0].jacobian.dudrow,
                                    obs_list[0].jacobian.dvdcol,
                                    obs_list[0].jacobian.dvdrow,
                                    obs_list[0].jacobian.col0,
                                    obs_list[0].jacobian.row0)
            
            res_ = self.measure_shape_metacal(obs_list, t['size'], method='bootstrap', flux_=get_flux(obs_list), fracdev=t['bulge_flux'],use_e=[t['int_e1'],t['int_e2']])
            out = self.measure_psf_shape_moments(psf_list, method='coadd')
            mask = (out['flag']==0)
            out = out[mask]
            iteration=0
            for key in metacal_keys:
                if res_==0:
                    res_tot[iteration]['flags'][i]                     = 2 # flag 2 means the object didnt pass shape fit. 
                elif res_[key]['flags']==0:
                    res_tot[iteration]['coadd_px'][i]                  = res_[key]['pars'][0]
                    res_tot[iteration]['coadd_py'][i]                  = res_[key]['pars'][1]
                    res_tot[iteration]['coadd_flux'][i]                = res_[key]['pars'][5] / wcs.pixelArea()
                    res_tot[iteration]['coadd_snr'][i]                 = res_[key]['s2n']
                    res_tot[iteration]['coadd_e1'][i]                  = res_[key]['pars'][2]
                    res_tot[iteration]['coadd_e2'][i]                  = res_[key]['pars'][3]
                    res_tot[iteration]['coadd_hlr'][i]                 = res_[key]['pars'][4]

                if len(out)!=0:
                    res_tot[iteration]['coadd_psf_e1'][i]        = np.average(out['e1'])
                    res_tot[iteration]['coadd_psf_e2'][i]        = np.average(out['e2'])
                    res_tot[iteration]['coadd_psf_T'][i]         = np.average(out['T'])
                    res_tot[iteration]['psf_nexp_used'][i]       = len(out)
                else:
                    res_tot[iteration]['coadd_psf_e1'][i]        = -9999
                    res_tot[iteration]['coadd_psf_e2'][i]        = -9999
                    res_tot[iteration]['coadd_psf_T'][i]         = -9999
                iteration+=1

        print('done measuring',self.rank)

        self.comm.Barrier()
        print('after first barrier')

        for j in range(5):
            if self.rank==0:
                for i in range(1,self.size):
                    print('getting',i)
                    tmp_res   = self.comm.recv(source=i)
                    mask      = tmp_res['size']!=0
                    res_tot[j][mask] = tmp_res[mask]
                    # coadd.update(self.comm.recv(source=i))

                print('before barrier',self.rank)
                self.comm.Barrier()
                # print coadd.keys()
                res = res_tot[j][np.argsort(res_tot[j]['ind'])]
                res['ra'] = np.degrees(res['ra'])
                res['dec'] = np.degrees(res['dec'])
                if self.shape_iter is None:
                    ilabel = 0
                else:
                    ilabel = self.shape_iter
                filename = get_filename(self.params['out_path'],
                                        self.params['out_dir'],
                                        self.params['output_meds'],
                                        var=self.pointing.filter+'_'+str(self.pix)+'_'+str(ilabel)+'_mcal_drizzle_'+str(metacal_keys[j]),
                                        ftype='fits',
                                        overwrite=True)
                fio.write(filename,res)

            else:

                self.comm.send(res_tot[j], dest=0)
                #self.comm.send(coadd, dest=0)
                #coadd = None
                print('before barrier',self.rank)
                self.comm.Barrier()

    def cleanup(self):

        filenames = get_filenames(self.params['out_path'],
                                    'ngmix',
                                    self.params['output_meds'],
                                    var=self.pointing.filter,
                                    ftype='fits')
        filename = get_filename(self.params['out_path'],
                    'ngmix',
                    self.params['output_meds'],
                    var=self.pointing.filter+'_combined',
                    ftype='fits',
                    overwrite=True)

        length = 0
        for f_ in filenames:
            if length==0:
                tmp = fio.FITS(f_)[-1].read()
            length += fio.FITS(f_)[-1].read_header()['NAXIS2']


        l = 0
        out = np.zeros(length,dtype=tmp.dtype)
        for f_ in filenames:
            tmp = fio.FITS(f_)[-1].read()
            for name in tmp.dtype.names:
                out[name][l:l+len(tmp)] = tmp[name]
            l+=len(tmp)

        out = out[np.argsort(out['ind'])]

        fio.write(filename,out)

