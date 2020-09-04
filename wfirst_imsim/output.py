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
import galsim.wfirst as wfirst
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
import meds
from ngmix.jacobian import Jacobian
from ngmix.observation import Observation, ObsList, MultiBandObsList,make_kobs
from ngmix.galsimfit import GalsimRunner,GalsimSimple,GalsimTemplateFluxFitter
from ngmix.guessers import R50FluxGuesser
from ngmix.bootstrap import PSFRunner
from ngmix import priors, joint_prior

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

import wfirst_imsim

class accumulate_output_disk(object):

    def __init__(self, param_file, filter_, pix, comm, ignore_missing_files = False, setup = False,condor_build=False, shape=False, shape_iter = None, shape_cnt = None):

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
        self.logger = logging.getLogger('wfirst_sim')
        self.pointing   = wfirst_imsim.pointing(self.params,self.logger,filter_=filter_,sca=None,dither=None)
        self.pix = pix
        self.skip = False

        self.comm = comm
        status = MPI.Status()
        if self.comm is None:
            self.rank = 0
            self.size = 1
        else:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

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

        if shape:
            self.file_exists = True

            # Get PSFs for all SCAs
            all_scas = np.array([i for i in range(1,19)])
            self.all_psfs = []
            filter_ = 'H158'
            b = galsim.BoundsI( xmin=1,
                                xmax=32,
                                ymin=1,
                                ymax=32)
            for sca in all_scas:
                psf_stamp = galsim.Image(b, scale=wfirst.pixel_scale)
                psf_sca = wfirst.getPSF(sca, filter_, SCA_pos=None, approximate_struts=True, wavelength=wfirst.getBandpasses(AB_zeropoint=True)[filter_].effective_wavelength, high_accuracy=False)
                st_model = galsim.DeltaFunction(flux=1.)
                st_model = st_model.evaluateAtWavelength(wfirst.getBandpasses(AB_zeropoint=True)[filter_].effective_wavelength)
                st_model = st_model.withFlux(1.)
                st_model = galsim.Convolve(st_model, psf_sca)
                st_model.drawImage(image=psf_stamp)
                self.all_psfs.append(psf_stamp)
            #print(self.all_psfs)

            #if not condor:
            #    raise ParamError('Not intended to work outside condor.')
            if ('output_meds' not in self.params) or ('psf_meds' not in self.params):
                raise ParamError('Must define both output_meds and psf_meds in yaml')
            if (self.params['output_meds'] is None) or (self.params['psf_meds'] is None):
                raise ParamError('Must define both output_meds and psf_meds in yaml')
            print('shape',self.shape_iter,self.shape_cnt)
            self.load_index()
            #self.local_meds = get_filename('./',
            #        '',
            #        self.params['output_meds'],
            #        var=self.pointing.filter+'_'+str(self.pix),
            #        ftype='fits',
            #        overwrite=False)
            self.local_meds = get_filename(self.params['out_path'],
                    'meds',
                    self.params['output_meds'],
                    var=self.pointing.filter+'_'+str(self.pix),
                    ftype='fits',
                    overwrite=False)
            #self.local_meds_psf = get_filename('./',
            #        '',
            #        self.params['psf_meds'],
            #        var=self.pointing.filter+'_'+str(self.pix),
            #        ftype='fits',
            #        overwrite=False)

            os.system( 'gunzip '+self.local_meds+'.gz')
            print(self.local_meds)

            #if self.local_meds != self.local_meds_psf:
            #    os.system( 'gunzip '+self.local_meds_psf+'.gz')

            return
        else:
            self.file_exists = False

        if (not setup)&(not condor_build):
            if self.rank==0:
                make = True
            else:
                make = False

            self.meds_filename = get_filename(self.params['out_path'],
                                'meds',
                                self.params['output_meds'],
                                var=self.pointing.filter+'_'+str(self.pix),
                                ftype='fits.gz',
                                overwrite=False,
                                make=make)
            self.local_meds = get_filename('/scratch/',
                                'meds',
                                self.params['output_meds'],
                                var=self.pointing.filter+'_'+str(self.pix),
                                ftype='fits',
                                overwrite=False,
                                make=make)

            self.local_meds_psf = self.local_meds
            if 'psf_meds' in self.params:
                if self.params['psf_meds'] is not None:
                    self.meds_psf = get_filename(self.params['psf_path'],
                            'meds',
                            self.params['psf_meds'],
                            var=self.pointing.filter+'_'+str(self.pix),
                            ftype='fits.gz',
                            overwrite=False,make=False)
                    if self.meds_psf!=self.meds_filename:
                        self.local_meds_psf = get_filename('./',
                                    '',
                                    self.params['psf_meds'],
                                    var=self.pointing.filter+'_'+str(self.pix),
                                    ftype='fits',
                                    overwrite=False)
                    if not condor:
                        if self.meds_psf!=self.meds_filename:
                            shutil.copy(self.meds_psf,self.local_meds_psf+'.gz')
                            os.system( 'gunzip '+self.local_meds_psf+'.gz')

        if self.rank>0:
            return

        print('to before setup')
        if setup:
            self.accumulate_index_table()
            return

        if condor_build:
            self.load_index(full=True)
            self.condor_build()
            return

        self.load_index()
        tmp = self.EmptyMEDS()
        if tmp is None:
            self.skip = True
            return
        if tmp:
            shutil.copy(self.meds_filename,self.local_meds+'.gz')
            os.system( 'gunzip '+self.local_meds+'.gz')
            self.file_exists = True
            return
        self.accumulate_dithers(condor=False)


    def accumulate_index_table(self):

        print('inside accumulate')

        index_filename = get_filename(self.params['out_path'],
                            'truth',
                            self.params['output_meds'],
                            var=self.pointing.filter+'_index_sorted',
                            ftype='fits.gz',
                            overwrite=False)

        if (os.path.exists(index_filename)) and (not self.params['overwrite']):

            print('break accumulate')
            return

        else:
            setup=True
            if not setup:
                raise ParamError('Trying to setup index file in potentially parallel run. Run with setup first.')

            print('good accumulate')
            index_files = get_filenames(self.params['out_path'],
                                        'truth',
                                        self.params['output_meds'],
                                        var='index'+'_'+self.pointing.filter,
                                        ftype='fits',
                                        exclude='_star.fits')

            print('good2 accumulate',index_files)
            length = 0
            for filename in index_files:
                print('length ',filename)
                length+=fio.FITS(filename)[-1].read_header()['NAXIS2']

            print('tmp')

            self.index = np.zeros(length,dtype=fio.FITS(index_files[0])[-1].read().dtype)
            length = 0
            for filename in index_files:
                print('reading ',filename)
                f = fio.FITS(filename)[-1].read()
                self.index[length:length+len(f)] = f
                length += len(f)

            self.index = self.index[np.argsort(self.index, order=['ind','dither'])]

            steps = np.where(np.roll(self.index['ind'],1)!=self.index['ind'])[0]
            self.index_ = np.zeros(len(self.index)+len(np.unique(self.index['ind'])),dtype=self.index.dtype)
            for name in self.index.dtype.names:
                if name=='dither':
                    self.index_[name] = np.insert(self.index[name],steps,np.ones(len(steps))*-1)
                else:
                    self.index_[name] = np.insert(self.index[name],steps,self.index[name][steps])

            self.index = self.index_
            self.index_= None
            self.index['ra']  = np.degrees(self.index['ra'])
            self.index['dec'] = np.degrees(self.index['dec'])
            fio.write(index_filename,self.index,clobber=True)

    def condor_build(self):

        if not self.params['condor']:
            return

        a = """#-*-shell-script-*-

universe     = vanilla
Requirements = OSGVO_OS_VERSION == "7" && CVMFS_oasis_opensciencegrid_org_REVISION >= 10686 && (HAS_CVMFS_sw_lsst_eu =?= True)

+ProjectName = "duke.lsst"
+WantsCvmfsStash = true
request_memory = 4G

should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
Executable     = ../run_osg.sh
transfer_output_files   = meds
Initialdir     = /stash/user/troxel/wfirst_sim_%s/
log            = %s_meds_log_$(MEDS).log
Arguments = %s_osg.yaml H158 meds $(MEDS)
Output         = %s_meds_$(MEDS).log
Error          = %s_meds_$(MEDS).log


""" % (self.params['output_meds'],self.params['output_tag'],self.params['output_tag'],self.params['output_tag'],self.params['output_tag'])

        a2 = """#-*-shell-script-*-

universe     = vanilla
Requirements = OSGVO_OS_VERSION == "7" && CVMFS_oasis_opensciencegrid_org_REVISION >= 10686 && (HAS_CVMFS_sw_lsst_eu =?= True)

+ProjectName = "duke.lsst"
+WantsCvmfsStash = true
request_memory = 2G

should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
Executable     = ../run_osg.sh
transfer_output_files   = ngmix
Initialdir     = /stash/user/troxel/wfirst_sim_%s/
log            = %s_shape_log_$(MEDS)_$(ITER).log
Arguments = %s_osg.yaml H158 meds shape $(MEDS) $(ITER) 5
Output         = %s_shape_$(MEDS)_$(ITER).log
Error          = %s_shape_$(MEDS)_$(ITER).log


""" % (self.params['output_meds'],self.params['output_tag'],self.params['output_tag'],self.params['output_tag'],self.params['output_tag'])

        b = """transfer_input_files    = /home/troxel/wfirst_stack/wfirst_stack.tar.gz, \
/home/troxel/wfirst_imsim_paper1/code/osg_runs/%s/%s_osg.yaml, \
/home/troxel/wfirst_imsim_paper1/code/meds_pix_list.txt, \
/stash/user/troxel/wfirst_sim_%s/run.tar""" % (self.params['output_meds'],self.params['output_tag'],self.params['output_meds'])

        # print(self.index)
        pix0 = self.get_index_pix()
        # print(pix0)
        p = np.unique(pix0)
        p2 = np.array_split(p,10)
        for ip2,p2_ in enumerate(p2):
            script = a+"""
"""
            print(p)
            for ip,p_ in enumerate(p2_):
                # if ip>3:
                #     continue
                meds_psf = get_filename(self.params['psf_path'],
                'meds',
                self.params['psf_meds'],
                var=self.pointing.filter+'_'+str(p_),
                ftype='fits.gz',
                overwrite=False,make=False)
                file_list = ''
                stamps_used = np.unique(self.index[['dither','sca']][pix0==p_])
                for i in range(len(stamps_used)):
                    if stamps_used['dither'][i]==-1:
                        continue
                    print(p_,i)
                    # filename = '/stash/user/troxel/wfirst_sim_fiducial/stamps/fiducial_H158_'+str(stamps_used['dither'][i])+'/'+str(stamps_used['sca'][i])+'_0.cPickle'
                    filename = get_filename(self.params['condor_zip_dir'],
                                            'stamps',
                                            self.params['output_meds'],
                                            var=self.pointing.filter+'_'+str(stamps_used['dither'][i]),
                                            name2=str(stamps_used['sca'][i])+'_0',
                                            ftype='cPickle.gz',
                                            overwrite=False,make=False)
                    file_list+=', '+filename
                d = """MEDS=%s
Queue

""" % (str(p_))
                script+="""
"""+b
                if 'psf_meds' in self.params:
                    if self.params['psf_meds'] is not None:
                        if self.params['psf_meds']!=self.params['output_meds']:
                            script+=', '+meds_psf
                script+=file_list+"""
"""+d

            # print(script)
            # print(self.params['psf_meds'])
            f = open(self.params['output_tag']+'_meds_run_osg_'+str(ip2)+'.sh','w')
            f.write(script)
            f.close()

        script = a2+"""
"""
        meds_psf = get_filename(self.params['psf_path'],
                            'meds',
                            self.params['psf_meds'],
                            var=self.pointing.filter+'_$(MEDS)',
                            ftype='fits.gz',
                            overwrite=False,make=False)
        meds = get_filename(self.params['condor_zip_dir'],
                            'meds',
                            self.params['output_meds'],
                            var=self.pointing.filter+'_$(MEDS)',
                            ftype='fits.gz',
                            overwrite=False,make=False)
        script+="""
"""+b
        script+=', '+meds
        if 'psf_meds' in self.params:
            if self.params['psf_meds'] is not None:
                if self.params['psf_meds']!=self.params['output_meds']:
                    script+=', '+meds_psf

        for ip,p_ in enumerate(p):
            d = """MEDS=%s
Queue ITER from seq 0 1 4 |

""" % (str(p_))
            script+="""
"""+d

        f = open(self.params['output_tag']+'_meds_shape_osg.sh','w')
        f.write(script)
        f.close()

    def load_index(self,full=False):

        index_filename = get_filename(self.params['out_path'],
                            'truth',
                            self.params['output_meds'],
                            var=self.pointing.filter+'_index_sorted',
                            ftype='fits.gz',
                            overwrite=False)

        self.index = fio.FITS(index_filename)[-1].read()

        if full:
            self.index = self.index[self.index['stamp']!=0]
        else:
            self.index = self.index[(self.index['stamp']!=0) & (self.get_index_pix()==self.pix)]

        # print 'debugging here'
        # self.index = self.index[self.index['ind']<np.unique(self.index['ind'])[5]]
        # print self.index
        self.steps = np.where(np.roll(self.index['ind'],1)!=self.index['ind'])[0]
        # print self.steps
        # print 'debugging here'

    def mask_index(self,pix):

        return self.index[self.get_index_pix()==pix]

    def get_index_pix(self):

        return hp.ang2pix(self.params['nside'],np.pi/2.-np.radians(self.index['dec']),np.radians(self.index['ra']),nest=True)

    def EmptyMEDS(self):
        """
        Based on galsim.des.des_meds.WriteMEDS().
        """

        from galsim._pyfits import pyfits

        if len(self.index)==0:
            print('skipping due to no objects')
            return None

        if (os.path.exists(self.meds_filename+'.gz')) or (os.path.exists(self.meds_filename)):
            if not self.params['overwrite']:
                print('skipping due to file exists')
                return True
            os.remove(self.meds_filename+'.gz')
            if os.path.exists(self.meds_filename):
                os.remove(self.meds_filename)
        if os.path.exists(self.local_meds):
            os.remove(self.local_meds)
        if os.path.exists(self.local_meds+'.gz'):
            os.remove(self.local_meds+'.gz')

        print(self.local_meds)
        m = fio.FITS(self.local_meds,'rw',clobber=True)

        print('Starting empty meds pixel',self.pix)
        indices = self.index['ind']
        bincount = np.bincount(indices)
        indcheck = np.where(bincount>0)[0]
        bincount = bincount[bincount>0]
        MAX_NCUTOUTS = np.max(bincount)
        print('MAX_NCUTOUTS', MAX_NCUTOUTS)

        assert np.sum(bincount==1) == 0
        assert np.all(indcheck==np.unique(indices))
        assert np.all(indcheck==indices[self.steps])
        cum_exps = len(indices)
        # get number of objects
        n_obj = len(indcheck)

        # get the primary HDU
        primary = pyfits.PrimaryHDU()

        # second hdu is the object_data
        # cf. https://github.com/esheldon/meds/wiki/MEDS-Format
        dtype = [
            ('id', 'i8'),
            ('number', 'i8'),
            ('box_size', 'i8'),
            ('psf_box_size', 'i8'),
            ('psf_box_size2', 'i8'),
            ('ra','f8'),
            ('dec','f8'),
            ('ncutout', 'i8'),
            ('file_id', 'i8', (MAX_NCUTOUTS,)),
            ('start_row', 'i8', (MAX_NCUTOUTS,)),
            ('psf_start_row', 'i8', (MAX_NCUTOUTS,)),
            ('psf_start_row2', 'i8', (MAX_NCUTOUTS,)),
            ('orig_row', 'f8', (MAX_NCUTOUTS,)),
            ('orig_col', 'f8', (MAX_NCUTOUTS,)),
            ('orig_start_row', 'i8', (MAX_NCUTOUTS,)),
            ('orig_start_col', 'i8', (MAX_NCUTOUTS,)),
            ('cutout_row', 'f8', (MAX_NCUTOUTS,)),
            ('cutout_col', 'f8', (MAX_NCUTOUTS,)),
            ('dudrow', 'f8', (MAX_NCUTOUTS,)),
            ('dudcol', 'f8', (MAX_NCUTOUTS,)),
            ('dvdrow', 'f8', (MAX_NCUTOUTS,)),
            ('dvdcol', 'f8', (MAX_NCUTOUTS,)),
            ('dither', 'i8', (MAX_NCUTOUTS,)),
            ('sca', 'i8', (MAX_NCUTOUTS,)),
        ]

        data                 = np.zeros(n_obj,dtype)
        data['id']           = np.arange(n_obj)
        data['number']       = self.index['ind'][self.steps]
        data['ra']           = self.index['ra'][self.steps]
        data['dec']          = self.index['dec'][self.steps]
        data['ncutout']      = bincount
        for i in range(len(self.steps)-1):
            data['box_size'][i] = np.min(self.index['stamp'][self.steps[i]:self.steps[i+1]])
        data['box_size'][i+1]   = np.min(self.index['stamp'][self.steps[-1]:])
        data['psf_box_size'] = np.ones(n_obj)*self.params['psf_stampsize']
        data['psf_box_size2'] = np.ones(n_obj)*self.params['psf_stampsize']*self.params['oversample']
        m.write(data,extname='object_data')

        length = np.sum(bincount*data['box_size']**2)
        psf_length = np.sum(bincount*data['psf_box_size']**2)
        psf_length2 = np.sum(bincount*data['psf_box_size2']**2)
        # print 'lengths',length,psf_length,bincount,data['box_size']

        # third hdu is image_info
        dtype = [
            ('image_path', 'S256'),
            ('image_ext', 'i8'),
            ('weight_path', 'S256'),
            ('weight_ext', 'i8'),
            ('seg_path','S256'),
            ('seg_ext','i8'),
            ('bmask_path', 'S256'),
            ('bmask_ext', 'i8'),
            ('bkg_path', 'S256'),
            ('bkg_ext', 'i8'),
            ('image_id', 'i8'),
            ('image_flags', 'i8'),
            ('magzp', 'f8'),
            ('scale', 'f8'),
            ('position_offset', 'f8'),
        ]

        gstring             = 'generated_by_galsim'
        data                = np.zeros(n_obj,dtype)
        data['image_path']  = gstring
        data['weight_path'] = gstring
        data['seg_path']    = gstring
        data['bmask_path']  = gstring
        data['bkg_path']    = gstring
        data['magzp']       = 30
        m.write(data,extname='image_info')

        # fourth hdu is metadata
        # default values?
        dtype = [
            ('magzp_ref', 'f8'),
            ('DESDATA', 'S256'),
            ('cat_file', 'S256'),
            ('coadd_image_id', 'S256'),
            ('coadd_file','S256'),
            ('coadd_hdu','i8'),
            ('coadd_seg_hdu', 'i8'),
            ('coadd_srclist', 'S256'),
            ('coadd_wt_hdu', 'i8'),
            ('coaddcat_file', 'S256'),
            ('coaddseg_file', 'S256'),
            ('cutout_file', 'S256'),
            ('max_boxsize', 'S3'),
            ('medsconv', 'S3'),
            ('min_boxsize', 'S2'),
            ('se_badpix_hdu', 'i8'),
            ('se_hdu', 'i8'),
            ('se_wt_hdu', 'i8'),
            ('seg_hdu', 'i8'),
            ('psf_hdu', 'i8'),
            ('sky_hdu', 'i8'),
            ('fake_coadd_seg', 'f8'),
        ]

        data                   = np.zeros(n_obj,dtype)
        data['magzp_ref']      = 30
        data['DESDATA']        = gstring
        data['cat_file']       = gstring
        data['coadd_image_id'] = gstring
        data['coadd_file']     = gstring
        data['coadd_hdu']      = 9999
        data['coadd_seg_hdu']  = 9999
        data['coadd_srclist']  = gstring
        data['coadd_wt_hdu']   = 9999
        data['coaddcat_file']  = gstring
        data['coaddseg_file']  = gstring
        data['cutout_file']    = gstring
        data['max_boxsize']    = '-1'
        data['medsconv']       = 'x'
        data['min_boxsize']    = '-1'
        data['se_badpix_hdu']  = 9999
        data['se_hdu']         = 9999
        data['se_wt_hdu']      = 9999
        data['seg_hdu']        = 9999
        data['psf_hdu']        = 9999
        data['sky_hdu']        = 9999
        data['fake_coadd_seg'] = 9999
        m.write(data,extname='metadata')

        # rest of HDUs are image vectors
        print('Writing empty meds pixel',self.pix)
        m.write(np.zeros(length,dtype='f8'),extname='image_cutouts')
        m.write(np.zeros(length,dtype='f8'),extname='weight_cutouts')
        # m.write(np.zeros(length,dtype='f8'),extname='seg_cutouts')
        #m.write(np.zeros(psf_length,dtype='f8'),extname='psf')
        #m.write(np.zeros(psf_length2,dtype='f8'),extname='psf2')
        # m['image_cutouts'].write(np.zeros(1,dtype='f8'), start=[length])
        # m['weight_cutouts'].write(np.zeros(1,dtype='f8'), start=[length])
        # m['seg_cutouts'].write(np.zeros(1,dtype='f8'), start=[length])
        # m['psf'].write(np.zeros(1,dtype='f8'), start=[psf_length])

        m.close()
        print('Done empty meds pixel',self.pix)

        return False

    def dump_meds_start_info(self,object_data,i,j):

        #print(i, j, len(object_data['start_row']))
        #print(object_data['start_row'][i][j])
        object_data['start_row'][i][j] = np.sum((object_data['ncutout'][:i])*object_data['box_size'][:i]**2)+j*object_data['box_size'][i]**2
        # change here
        # object_data['psf_start_row'][i][j] = np.sum((object_data['ncutout'][:i])*object_data['box_size'][:i]**2)+j*object_data['box_size'][i]**2
        object_data['psf_start_row'][i][j] = np.sum((object_data['ncutout'][:i])*object_data['psf_box_size'][:i]**2)+j*object_data['psf_box_size'][i]**2
        object_data['psf_start_row2'][i][j] = np.sum((object_data['ncutout'][:i])*object_data['psf_box_size2'][:i]**2)+j*object_data['psf_box_size2'][i]**2
        # print 'starts',i,j,object_data['start_row'][i][j],object_data['psf_start_row'][i][j],object_data['box_size'][i],object_data['psf_box_size'][i]

    def dump_meds_wcs_info( self,
                            object_data,
                            i,
                            j,
                            x,
                            y,
                            origin_x,
                            origin_y,
                            dither,
                            sca,
                            dudx,
                            dudy,
                            dvdx,
                            dvdy,
                            wcsorigin_x=None,
                            wcsorigin_y=None):

        object_data['orig_row'][i][j]       = y
        object_data['orig_col'][i][j]       = x
        object_data['orig_start_row'][i][j] = origin_y
        object_data['orig_start_col'][i][j] = origin_x
        object_data['dither'][i][j]         = dither
        object_data['sca'][i][j]            = sca
        object_data['dudcol'][i][j]         = dudx
        object_data['dudrow'][i][j]         = dudy
        object_data['dvdcol'][i][j]         = dvdx
        object_data['dvdrow'][i][j]         = dvdy
        if wcsorigin_y is None:
            object_data['cutout_row'][i][j]     = y-origin_y
        else:
            object_data['cutout_row'][i][j]     = wcsorigin_y
        if wcsorigin_x is None:
            object_data['cutout_col'][i][j]     = x-origin_x
        else:
            object_data['cutout_col'][i][j]     = wcsorigin_x

    def dump_meds_pix_info(self,m,object_data,i,j,gal,weight):#,psf):#,psf2):

        #print(len(gal), object_data['box_size'][i]**2, i)
        assert len(gal)==object_data['box_size'][i]**2
        assert len(weight)==object_data['box_size'][i]**2
        # assert len(psf)==object_data['psf_box_size'][i]**2
        # change here
        m['image_cutouts'].write(gal, start=object_data['start_row'][i][j])
        m['weight_cutouts'].write(weight, start=object_data['start_row'][i][j])
        #m['psf'].write(psf, start=object_data['psf_start_row'][i][j])
        #m['psf2'].write(psf2, start=object_data['psf_start_row2'][i][j])

    def accumulate_dithers(self, condor=False):
        """
        Accumulate the written pickle files that contain the postage stamps for all objects, with SCA and dither ids.
        Write stamps to MEDS file, and SCA and dither ids to truth files.
        """


        print('mpi check',self.rank,self.size)

        print('Starting meds pixel',self.pix)
        m = fio.FITS(self.local_meds,'rw')
        object_data = m['object_data'].read()

        stamps_used = np.unique(self.index[['dither','sca']])
        print('number of files',stamps_used)
        for si,s in enumerate(range(len(stamps_used))):
            if stamps_used['dither'][s] == -1:
                continue

            if condor:
                filename = get_filename('./',
                                        '',
                                        self.params['output_meds'],
                                        var=self.pointing.filter+'_'+str(stamps_used['dither'][s]),
                                        name2=str(stamps_used['sca'][s])+'_0',
                                        ftype='cPickle',
                                        overwrite=False)
            else:
                filename1 = get_filenames(self.params['out_path'],
                                        'stamps',
                                        self.params['output_meds'],
                                        var=self.pointing.filter+'_'+str(stamps_used['dither'][s]),
                                        name2=str(stamps_used['sca'][s]),
                                        exclude='star',
                                        ftype='cPickle.gz')
                #shutil.copy(filename1,filename+'.gz')

            print(stamps_used['dither'][s],stamps_used['sca'][s])

            for f in filename1:
                filename=f.replace(self.params['out_path']+'stamps/', self.params['tmpdir'])
                #print(f, filename)
                shutil.copy(f, filename)
                os.system('gunzip '+filename)

                filename=filename.replace('.gz', '')
                with io.open(filename, 'rb') as p :
                    unpickler = pickle.Unpickler(p)
                    while p.peek(1) :
                        gal = unpickler.load()
                        i = np.where(gal['ind'] == object_data['number'])[0]
                        if len(i)==0:
                            continue
                        assert len(i)==1
                        # print gal
                        i = i[0]
                        j = np.nonzero(object_data['dither'][i])[0]

                        if len(j)==0:
                            j = 0
                        else:
                            print('max j', j)
                            j = np.max(j)+1
                        index_i = np.where((self.index['ind']==gal['ind'])&(self.index['dither']==gal['dither']))[0]
                        assert len(index_i)==1
                        index_i=index_i[0]

                        if j==0:
                            self.dump_meds_start_info(object_data,i,j)
                            j+=1
                        self.dump_meds_start_info(object_data,i,j)

                        #print(i,object_data['box_size'][i],index_i,self.index['stamp'][index_i])
                        if object_data['box_size'][i] > self.index['stamp'][index_i]:
                            pad_    = int((object_data['box_size'][i] - self.index['stamp'][index_i])/2)
                            gal_    = np.pad(gal['gal'].array,(pad_,pad_),'wrap').flatten()
                            weight_ = np.pad(gal['weight'].reshape(self.index['stamp'][index_i],self.index['stamp'][index_i]),(pad_,pad_),'wrap').flatten()
                        elif object_data['box_size'][i] < self.index['stamp'][index_i]:
                            pad_    = int((self.index['stamp'][index_i] - object_data['box_size'][i])/2)
                            gal_    = gal['gal'].array[pad_:-pad_,pad_:-pad_].flatten()
                            weight_ = gal['weight'].reshape(self.index['stamp'][index_i],self.index['stamp'][index_i])[pad_:-pad_,pad_:-pad_].flatten()
                        else:
                            gal_    = gal['gal'].array.flatten()
                            weight_ = gal['weight']

                        #print(len(gal['gal'].array.flatten()),len(gal_))

                        # orig_box_size = object_data['box_size'][i]
                        # if True:
                        #     object_data['box_size'][i] = int(orig_box_size*1.5)+int(orig_box_size*1.5)%2

                        # box_diff = object_data['box_size'][i] - self.index['stamp'][index_i]

                        # ====================
                        # this is a patch, remove later
                        # gal['x']+=0.5
                        # gal['y']+=0.5
                        # ===================
                        origin_x = gal['gal'].origin.x
                        origin_y = gal['gal'].origin.y
                        gal['gal'].setOrigin(0,0)
                        new_pos  = galsim.PositionD(gal['x']-origin_x,gal['y']-origin_y)
                        wcs = gal['gal'].wcs.affine(image_pos=new_pos)
                        self.dump_meds_wcs_info(object_data,
                                                i,
                                                j,
                                                gal['x'],
                                                gal['y'],
                                                origin_x,
                                                origin_y,
                                                self.index['dither'][index_i],
                                                self.index['sca'][index_i],
                                                wcs.dudx,
                                                wcs.dudy,
                                                wcs.dvdx,
                                                wcs.dvdy)
                        print(i)
                        self.dump_meds_pix_info(m,
                                                object_data,
                                                i,
                                                j,
                                                gal_,
                                                weight_)
                                                #gal['psf'],
                                                #gal['psf2'])
                        # print np.shape(gals[gal]['psf']),gals[gal]['psf']

        # object_data['psf_box_size'] = object_data['box_size']
        print('Writing meds pixel',self.pix)
        m['object_data'].write(object_data)
        m.close()
        print('Done meds pixel',self.pix)

    def finish(self,condor=False):

        if self.rank>0:
            return

        print('start meds finish')
        if not self.file_exists:
            os.system('gzip '+self.local_meds)
        if not condor and not self.file_exists:
            shutil.move(self.local_meds+'.gz',self.meds_filename)
            # if os.path.exists(self.local_meds+'.gz'):
            #     os.remove(self.local_meds+'.gz')
        print('done meds finish')

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

    def get_exp_list(self,m,i,m2=None,size=None):

        def get_stamp(size,box_size):
            hlp = size*10./wfirst.pixel_scale
            if hlp>box_size:
                return int(box_size)
            if hlp<32:
                return 32
            return int(2**(int(np.log2(100))+1))

        if m2 is None:
            m2 = m

        obs_list=ObsList()
        psf_list=ObsList()

        if size is not None:
            box_size = get_stamp(size,m['box_size'][i])

        included = []
        w        = []
        # For each of these objects create an observation
        for j in range(m['ncutout'][i]):
            if j==0:
                continue
            # if j>1:
            #     continue
            im = m.get_cutout(i, j, type='image')
            im = im[:,len(im)//2-box_size//2:len(im)//2+box_size//2][len(im)//2-box_size//2:len(im)//2+box_size//2,:]
            weight = m.get_cutout(i, j, type='weight')
            weight = weight[:,len(weight)//2-box_size//2:len(weight)//2+box_size//2][len(weight)//2-box_size//2:len(weight)//2+box_size//2,:]

            im_psf = m2[j] #self.get_cutout_psf(m, m2, i, j)
            im_psf2 = im_psf #self.get_cutout_psf2(m, m2, i, j)
            if np.sum(im)==0.:
                print(self.local_meds, i, j, np.sum(im))
                print('no flux in image ',i,j)
                continue

            jacob = m.get_jacobian(i, j)
            gal_jacob=Jacobian(
                row=(m['orig_row'][i][j]-m['orig_start_row'][i][j])-m['box_size'][i]/2+box_size/2,
                col=(m['orig_col'][i][j]-m['orig_start_col'][i][j])-m['box_size'][i]/2+box_size/2,
                dvdrow=jacob['dvdrow'],
                dvdcol=jacob['dvdcol'],
                dudrow=jacob['dudrow'],
                dudcol=jacob['dudcol'])

            psf_center = int((m['psf_box_size2'][i]-1)/2.)
            psf_jacob2=Jacobian(
                row=jacob['row0']*self.params['oversample'],
                col=jacob['col0']*self.params['oversample'],
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
            obs = Observation(im, weight=weight, jacobian=gal_jacob, psf=psf_obs, meta={'offset_pixels':None,'file_id':None})
            obs.set_noise(noise)

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

        pix_range = galsim.wfirst.pixel_scale/10.
        e_range = 0.1
        fdev = 1.
        def pixe_guess(n):
            return 2.*n*np.random.random() - n

        # possible models are 'exp','dev','bdf' galsim.wfirst.pixel_scale
        cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.wfirst.pixel_scale, galsim.wfirst.pixel_scale)
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

        pix_range = galsim.wfirst.pixel_scale/10.
        e_range = 0.1
        fdev = 1.
        def pixe_guess(n):
            return 2.*n*np.random.random() - n

        # possible models are 'exp','dev','bdf' galsim.wfirst.pixel_scale
        cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.wfirst.pixel_scale, galsim.wfirst.pixel_scale)
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

        pix_range = galsim.wfirst.pixel_scale/10.
        e_range = 0.1
        fdev = 1.
        def pixe_guess(n):
            return 2.*n*np.random.random() - n

        # possible models are 'exp','dev','bdf' galsim.wfirst.pixel_scale
        cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.wfirst.pixel_scale, galsim.wfirst.pixel_scale)
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

    def measure_shape_metacal(self, obs_list, method='bootstrap', flux_=1000.0, fracdev=None, use_e=None):
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
            T = self.hlr
            pix_range = old_div(galsim.wfirst.pixel_scale,10.)
            e_range = 0.1
            fdev = 1.
            def pixe_guess(n):
                return 2.*n*np.random.random() - n

            cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.wfirst.pixel_scale, galsim.wfirst.pixel_scale)
            gp = ngmix.priors.GPriorBA(0.3)
            hlrp = ngmix.priors.FlatPrior(1.0e-4, 1.0e2)
            fracdevp = ngmix.priors.Normal(0.5, 0.1, bounds=[0., 1.])
            fluxp = ngmix.priors.FlatPrior(0, 1.0e5)

            prior = joint_prior.PriorSimpleSep(cp, gp, hlrp, fluxp)
            guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,500.])

            boot = ngmix.bootstrap.MaxMetacalBootstrapper(obs_list)
            psf_model = "gauss"
            gal_model = "gauss"

            lm_pars={'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
            max_pars={'method': 'lm', 'lm_pars':lm_pars}

            Tguess=T**2/(2*np.log(2))
            ntry=2
            boot.fit_metacal(psf_model, gal_model, max_pars, Tguess, prior=prior, ntry=ntry, metacal_pars=metacal_pars) 
            res_ = boot.get_metacal_result()

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

    def measure_psf_shape_moments(self,obs_list):

        def make_psf_image(self,obs):

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
            im = make_psf_image(self,obs)

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

    def get_coadd_shape(self):


        def get_flux(obs_list):
            flux = 0.
            for obs in obs_list:
                flux += obs.image.sum()
            flux /= len(obs_list)
            if flux<0:
                flux = 10.
            return flux

        #tmp
        # self.psf_model = []
        # for i in range(1,19):
        #     self.pointing.sca = i
        #     self.pointing.get_psf()
        #     self.psf_model.append(self.pointing.PSF)
        #tmp

        print('mpi check 2',self.rank,self.size)

        filename = get_filename(self.params['out_path'],
                                'truth',
                                self.params['output_truth'],
                                name2='truth_gal',
                                overwrite=False)
        truth = fio.FITS(filename)[-1]
        m  = meds.MEDS(self.local_meds)
        m2 = fio.FITS(self.local_meds_psf)
        if self.shape_iter is not None:
            indices = np.array_split(np.arange(len(m['number'][:])),self.shape_cnt)[self.shape_iter]
        else:
            indices = np.arange(len(m['number'][:]))

        print('rank in coadd_shape', self.rank)
        coadd = {}
        res   = np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),])#, ('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])

        for i,ii in enumerate(indices):
            if i%self.size!=self.rank:
                continue
            if i%100==0:
                print('made it to object',i)
            try_save = False

            ind = m['number'][ii]
            t   = truth[ind]

            res['ind'][i]                       = ind
            res['ra'][i]                        = t['ra']
            res['dec'][i]                       = t['dec']
            res['nexp_tot'][i]                  = m['ncutout'][ii]-1
            res['stamp'][i]                     = m['box_size'][ii]
            res['g1'][i]                        = t['g1']
            res['g2'][i]                        = t['g2']
            res['int_e1'][i]                    = t['int_e1']
            res['int_e2'][i]                    = t['int_e2']
            res['rot'][i]                       = t['rot']
            res['size'][i]                      = t['size']
            res['redshift'][i]                  = t['z']
            res['mag_'+self.pointing.filter][i] = t[self.pointing.filter]
            res['pind'][i]                      = t['pind']
            res['bulge_flux'][i]                = t['bflux']
            res['disk_flux'][i]                 = t['dflux']

            obs_list,psf_list,included,w = self.get_exp_list(m,ii,m2=m2,size=t['size'])
            if len(included)==0:
                continue
            # coadd[i]            = psc.Coadder(obs_list).coadd_obs
            # coadd[i].set_meta({'offset_pixels':None,'file_id':None})
            if self.params['shape_code']=='mof':
                res_,res_full_      = self.measure_shape_mof(obs_list,t['size'],flux=get_flux(obs_list),fracdev=t['bflux'],use_e=[t['int_e1'],t['int_e2']],model=self.params['ngmix_model'])
            elif self.params['shape_code']=='ngmix':
                res_,res_full_      = self.measure_shape_ngmix(obs_list,t['size'],model=self.params['ngmix_model'])
            else:
                raise ParamError('unknown shape code request')
            if res_full_['flags'] !=0:
                print('failed',i,ii,get_flux(obs_list))

            wcs = self.make_jacobian(obs_list[0].jacobian.dudcol,
                                    obs_list[0].jacobian.dudrow,
                                    obs_list[0].jacobian.dvdcol,
                                    obs_list[0].jacobian.dvdrow,
                                    obs_list[0].jacobian.col0,
                                    obs_list[0].jacobian.row0)

            if not self.params['avg_fit']:
                res['nexp_used'][i]                 = len(included)
                res['flags'][i]                     = res_full_['flags']
                if res_full_['flags']==0:
                    res['px'][i]                        = res_['pars'][0]
                    res['py'][i]                        = res_['pars'][1]
                    res['flux'][i]                      = res_['flux']
                    res['snr'][i]                       = res_['s2n_r']
                    res['e1'][i]                        = res_['pars'][2]
                    res['e2'][i]                        = res_['pars'][3]
                    res['cov_11'][i]                    = res_['pars_cov'][2,2]
                    res['cov_22'][i]                    = res_['pars_cov'][3,3]
                    res['cov_12'][i]                    = res_['pars_cov'][2,3]
                    res['cov_21'][i]                    = res_['pars_cov'][3,2]
                    res['hlr'][i]                       = res_['pars'][4]
                else:
                    try_save = False
            else:
                mask = []
                for flag in res_full_:
                    if flag['flags']==0:
                        mask.append(True)
                    else:
                        mask.append(False)
                mask = np.array(mask)
                res['nexp_used'][i]                 = np.sum(mask)
                div = 0
                if np.sum(mask)==0:
                    res['flags'][i] = 999
                else:
                    for j in range(len(mask)):
                        if mask[j]:
                            print(i,j,res_[j]['pars'][0],res_[j]['pars'][1])
                            div                                 += w[j]
                            res['px'][i]                        += res_[j]['pars'][0]
                            res['py'][i]                        += res_[j]['pars'][1]
                            res['flux'][i]                      += res_[j]['flux'] * w[j]
                            if self.params['shape_code']=='mof':
                                res['snr'][i]                       = res_[j]['s2n'] * w[j]
                            elif self.params['shape_code']=='ngmix':
                                res['snr'][i]                       = res_[j]['s2n_r'] * w[j]
                            res['e1'][i]                        += res_[j]['pars'][2] * w[j]
                            res['e2'][i]                        += res_[j]['pars'][3] * w[j]
                            res['hlr'][i]                       += res_[j]['pars'][4] * w[j]
                    res['px'][i]                        /= div
                    res['py'][i]                        /= div
                    res['flux'][i]                      /= div
                    res['snr'][i]                       /= div
                    res['e1'][i]                        /= div
                    res['e2'][i]                        /= div
                    res['hlr'][i]                       /= div

            if try_save:
                mosaic = np.hstack((obs_list[i].image for i in range(len(obs_list))))
                psf_mosaic = np.hstack((obs_list[i].psf.image for i in range(len(obs_list))))
                mosaic = np.vstack((mosaic,np.hstack((obs_list[i].weight for i in range(len(obs_list))))))
                plt.imshow(mosaic)
                plt.tight_layout()
                plt.savefig('/users/PCON0003/cond0083/tmp_'+str(i)+'.png', bbox_inches='tight')#, dpi=400)
                plt.close()
                plt.imshow(psf_mosaic)
                plt.tight_layout()
                plt.savefig('/users/PCON0003/cond0083/tmp_psf_'+str(i)+'.png', bbox_inches='tight')#, dpi=400)
                plt.close()

            out = self.measure_psf_shape_moments(psf_list)
            mask = out['flag']==0
            out = out[mask]
            w = w[mask]
            res['psf_e1'][i]        = np.average(out['e1'],weights=w)
            res['psf_e2'][i]        = np.average(out['e2'],weights=w)
            res['psf_T'][i]         = np.average(out['T'],weights=w)
            if len(out)<len(obs_list):
                print('----------- bad psf measurement in ',i)
            res['psf_nexp_used'][i] = len(out)

            # obs_list = ObsList()
            # obs_list.append(coadd[i])
            # res_,res_full_     = self.measure_shape(obs_list,t['size'],model=self.params['ngmix_model'])

            # res['coadd_flags'][i]                   = res_full_['flags']
            # if res_full_['flags']==0:
            #     res['coadd_px'][i]                  = res_['pars'][0]
            #     res['coadd_py'][i]                  = res_['pars'][1]
            #     res['coadd_flux'][i]                = res_['pars'][5] / wcs.pixelArea()
            #     res['coadd_snr'][i]                 = res_['s2n']
            #     res['coadd_e1'][i]                  = res_['pars'][2]
            #     res['coadd_e2'][i]                  = res_['pars'][3]
            #     res['coadd_hlr'][i]                 = res_['pars'][4]

            # out = self.measure_psf_shape_moments([coadd[i]])
            # if out['flag']==0:
            #     res['coadd_psf_e1'][i]        = out['e1']
            #     res['coadd_psf_e2'][i]        = out['e2']
            #     res['coadd_psf_T'][i]         = out['T']
            # else:
            #     res['coadd_psf_e1'][i]        = -9999
            #     res['coadd_psf_e2'][i]        = -9999
            #     res['coadd_psf_T'][i]         = -9999

        m.close()

        print('done measuring',self.rank)

        self.comm.Barrier()
        print('after first barrier')

        if self.rank==0:
            for i in range(1,self.size):
                print('getting',i)
                tmp_res   = self.comm.recv(source=i)
                mask      = tmp_res['size']!=0
                res[mask] = tmp_res[mask]
                # coadd.update(self.comm.recv(source=i))

            print('before barrier',self.rank)
            self.comm.Barrier()
            # print coadd.keys()
            res = res[np.argsort(res['ind'])]
            res['ra'] = np.degrees(res['ra'])
            res['dec'] = np.degrees(res['dec'])
            if self.shape_iter is None:
                ilabel = 0
            else:
                ilabel = self.shape_iter
            filename = get_filename(self.params['out_path'],
                                'ngmix',
                                self.params['output_meds'],
                                var=self.pointing.filter+'_'+str(self.pix)+'_'+str(ilabel),
                                ftype='fits',
                                overwrite=True)
            fio.write(filename,res)
            #tmp
            # if os.path.exists(self.local_meds):
            #     os.remove(self.local_meds)
            #tmp


            # m        = fio.FITS(self.local_meds,'rw')
            # object_data = m['object_data'].read()

            # for i in coadd:
            #     self.dump_meds_wcs_info(object_data,
            #                             i,
            #                             0,
            #                             9999,
            #                             9999,
            #                             9999,
            #                             9999,
            #                             9999,
            #                             9999,
            #                             coadd[i].jacobian.dudcol,
            #                             coadd[i].jacobian.dudrow,
            #                             coadd[i].jacobian.dvdcol,
            #                             coadd[i].jacobian.dvdrow,
            #                             coadd[i].jacobian.col0,
            #                             coadd[i].jacobian.row0)

            #     self.dump_meds_pix_info(m,
            #                             object_data,
            #                             i,
            #                             0,
            #                             coadd[i].image.flatten(),
            #                             coadd[i].weight.flatten(),
            #                             coadd[i].psf.image.flatten())

            # m['object_data'].write(object_data)
            # m.close()

        else:

            self.comm.send(res, dest=0)
            res = None
            # self.comm.send(coadd, dest=0)
            # coadd = None
            print('before barrier',self.rank)
            self.comm.Barrier()

    def get_coadd_shape_mcal(self):

        def get_flux(obs_list):
            flux = 0.
            for obs in obs_list:
                flux += obs.image.sum()
            flux /= len(obs_list)
            if flux<0:
                flux = 10.
            return flux

        #tmp
        # self.psf_model = []
        # for i in range(1,19):
        #     self.pointing.sca = i
        #     self.pointing.get_psf()
        #     self.psf_model.append(self.pointing.PSF)
        #tmp

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
        coadd = {}
        res   = np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),])#, ('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])

        metacal_keys=['noshear', '1p', '1m', '2p', '2m']
        res_noshear=np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),])
        res_1p=np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),])
        res_1m=np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),])
        res_2p=np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),])
        res_2m=np.zeros(len(m['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('px',float), ('py',float), ('flux',float), ('snr',float), ('e1',float), ('e2',float), ('int_e1',float), ('int_e2',float), ('hlr',float), ('psf_e1',float), ('psf_e2',float), ('psf_T',float), ('psf_nexp_used',int), ('stamp',int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('mag_'+self.pointing.filter,float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('flags',int), ('coadd_flags',int), ('nexp_used',int), ('nexp_tot',int), ('cov_11',float), ('cov_12',float), ('cov_21',float), ('cov_22',float),])
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
            m2 = [self.all_psfs[i].array for i in sca_list]
            obs_list,psf_list,included,w = self.get_exp_list(m,ii,m2=m2,size=t['size'])
            if len(included)==0:
                continue
            # coadd[i]            = psc.Coadder(obs_list).coadd_obs
            # coadd[i].set_meta({'offset_pixels':None,'file_id':None})
            if self.params['shape_code']=='mof':
                res_,res_full_      = self.measure_shape_mof(obs_list,t['size'],flux=get_flux(obs_list),fracdev=t['bflux'],use_e=[t['int_e1'],t['int_e2']],model=self.params['ngmix_model'])
            elif self.params['shape_code']=='ngmix':
                res_,res_full_      = self.measure_shape_ngmix(obs_list,t['size'],model=self.params['ngmix_model'])
            elif self.params['shape_code']=='metacal':
                res_ = self.measure_shape_metacal(obs_list, method='bootstrap', flux_=get_flux(obs_list), fracdec=t['bflux'],use_e=[t['int_e1'],t['int_e2']])
            else:
                raise ParamError('unknown shape code request')
            if res_full_['flags'] !=0:
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

                if not self.params['avg_fit']:
                    res_tot[iteration]['nexp_used'][i]                 = len(included)
                    res_tot[iteration]['flags'][i]                     = res_[key]['flags']
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
                    else:
                        try_save = False
                
                else:
                    mask = []
                    for flag in res_full_:
                        if flag['flags']==0:
                            mask.append(True)
                        else:
                            mask.append(False)
                    mask = np.array(mask)
                    res['nexp_used'][i]                 = np.sum(mask)
                    div = 0
                    if np.sum(mask)==0:
                        res['flags'][i] = 999
                    else:
                        for j in range(len(mask)):
                            if mask[j]:
                                print(i,j,res_[j]['pars'][0],res_[j]['pars'][1])
                                div                                 += w[j]
                                res['px'][i]                        += res_[j]['pars'][0]
                                res['py'][i]                        += res_[j]['pars'][1]
                                res['flux'][i]                      += res_[j]['flux'] * w[j]
                                if self.params['shape_code']=='mof':
                                    res['snr'][i]                       = res_[j]['s2n'] * w[j]
                                elif self.params['shape_code']=='ngmix':
                                    res['snr'][i]                       = res_[j]['s2n_r'] * w[j]
                                res['e1'][i]                        += res_[j]['pars'][2] * w[j]
                                res['e2'][i]                        += res_[j]['pars'][3] * w[j]
                                res['hlr'][i]                       += res_[j]['pars'][4] * w[j]
                        res['px'][i]                        /= div
                        res['py'][i]                        /= div
                        res['flux'][i]                      /= div
                        res['snr'][i]                       /= div
                        res['e1'][i]                        /= div
                        res['e2'][i]                        /= div
                        res['hlr'][i]                       /= div
                
                if try_save:
                    mosaic = np.hstack((obs_list[i].image for i in range(len(obs_list))))
                    psf_mosaic = np.hstack((obs_list[i].psf.image for i in range(len(obs_list))))
                    mosaic = np.vstack((mosaic,np.hstack((obs_list[i].weight for i in range(len(obs_list))))))
                    plt.imshow(mosaic)
                    plt.tight_layout()
                    plt.savefig('/users/PCON0003/cond0083/tmp_'+str(i)+'.png', bbox_inches='tight')#, dpi=400)
                    plt.close()
                    plt.imshow(psf_mosaic)
                    plt.tight_layout()
                    plt.savefig('/users/PCON0003/cond0083/tmp_psf_'+str(i)+'.png', bbox_inches='tight')#, dpi=400)
                    plt.close()

                iteration+=1
            #print("if shape measurement is right, ", res_['1p']['pars'][2], res_['1m']['pars'][2])
            #print("assignment value is right? ", res_tot[1]['e1'][i], res_tot[2]['e1'][i])
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
                                    'ngmix',
                                    self.params['output_meds'],
                                    var=self.pointing.filter+'_'+str(self.pix)+'_'+str(ilabel)+'_mcal_'+str(metacal_keys[j]),
                                    ftype='fits',
                                    overwrite=True)
                fio.write(filename,res)

            else:

                self.comm.send(res, dest=0)
                res = None
                # self.comm.send(coadd, dest=0)
                # coadd = None
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


#class output_metacal(accumulate_output_disk):

