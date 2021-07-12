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
#import guppy

from .output import accumulate_output_disk
from .image import draw_image 
from .image import draw_detector 
from .detector import modify_image
from .universe import init_catalogs
from .universe import setupCCM_ab
from .universe import addDust
from .telescope import pointing 
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

# Converts galsim Roman filter names to indices in Chris' dither file.
filter_dither_dict = {
    'J129' : 3,
    'F184' : 1,
    'Y106' : 4,
    'H158' : 2
}

class roman_sim(object):
    """
    Roman image simulation.

    Input:
    param_file : File path for input yaml config file or yaml dict. Example located at: ./example.yaml.
    """

    def __init__(self, param_file, index=None):

        if isinstance(param_file, str):
            # Load parameter file
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
            if 'condor' not in self.params:
                self.params['condor']=False

        else:
            # Else use existing param dict
            self.params     = param_file


        self.index = index
        if 'tmpdir' in self.params:
            os.chdir(self.params['tmpdir'])

        # Set up some information on processes and MPI
        if self.params['mpi']:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            print('doing mpi')
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

        print('mpi',self.rank,self.size)

        # Set up logger. I don't really use this, but it could be used.
        logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
        self.logger = logging.getLogger('roman_sim')

        return

    def setup(self,filter_,dither,sca=1,setup=False,load_cats=True):
        """
        Set up initial objects.

        Input:
        filter_ : A filter name. 'None' to determine by dither.
        """
        filter_dither_dict = {
                             'J129' : 3,
                             'F184' : 1,
                             'Y106' : 4,
                             'H158' : 2}
        filter_flux_dict = {
                            'J129' : 'j_Roman',
                            'F184' : 'F184W_Roman',
                            'Y106' : 'y_Roman',
                            'H158' : 'h_Roman'}

        if filter_!='None':
            # Filter be present in filter_dither_dict{} (exists in survey strategy file).
            if filter_ not in list(filter_dither_dict.keys()):
                raise ParamError('Supplied invalid filter: '+filter_)

        # This sets up a mostly-unspecified pointing object in this filter. We will later specify a dither and SCA to complete building the pointing information.
        if filter_=='None':
            self.pointing = pointing(self.params,self.logger,filter_=None,sca=None,dither=None,rank=self.rank)
        else:
            self.pointing = pointing(self.params,self.logger,filter_=filter_,sca=None,dither=None,rank=self.rank)

        if not setup:
            # This updates the dither
            self.pointing.update_dither(dither)
            # This sets up a specific pointing for this SCA (things like WCS, PSF)
            self.pointing.update_sca(sca)

        self.gal_rng = galsim.UniformDeviate(self.params['random_seed'])
        # This checks whether a truth galaxy/star catalog exist. If it doesn't exist, it is created based on specifications in the yaml file. It then sets up links to the truth catalogs on disk.
        if load_cats:
            self.cats     = init_catalogs(self.params, self.pointing, self.gal_rng, self.rank, self.size, comm=self.comm, setup=setup)

        print('Done with init_catalogs')

        if setup:
            return False

        if load_cats:
            if len(self.cats.gal_ind)==0:
                print('skipping due to no objects near pointing',str(self.rank))
                return True

        return False

    def get_sca_list(self):
        """
        Generate list of SCAs to simulate based on input parameter file.
        """

        if hasattr(self.params,'sca'):
            if self.params['sca'] is None:
                sca_list = np.arange(1,19)
            elif self.params['sca'] == 'None':
                sca_list = np.arange(1,19)
            elif hasattr(self.params['sca'],'__len__'):
                if type(self.params['sca'])==str:
                    raise ParamError('Provided SCA list is not numeric.')
                sca_list = self.params['sca']
            else:
                sca_list = [self.params['sca']]
        else:
            sca_list = np.arange(1,19)

        return sca_list

    def get_inds(self):
        """
        Checks things are setup, cut out objects not near SCA, and distributes objects across procs.
        """

        # If something went wrong and there's no pointing defined, then crash.
        if not hasattr(self,'pointing'):
            raise ParamError('Sim object has no pointing - need to run sim.setup() first.')
        if self.pointing.dither is None:
            raise ParamError('Sim pointing object has no dither assigned - need to run sim.pointing.update_dither() first.')

        mask_sca      = self.pointing.in_sca(self.cats.gals['ra'][:],self.cats.gals['dec'][:])
        mask_sca_star = self.pointing.in_sca(self.cats.stars['ra'][:],self.cats.stars['dec'][:])
        if self.cats.supernovae is not None:
            mask_sca_supernova = self.pointing.in_sca(self.cats.supernovae['ra'][:],self.cats.supernovae['dec'][:])
        self.cats.add_mask(mask_sca,star_mask=mask_sca_star,supernova_mask=mask_sca_supernova)

    def iterate_image(self):
        """
        This is the main simulation. It instantiates the draw_image object, then iterates over all galaxies and stars. The output is then accumulated from other processes (if mpi is enabled), and saved to disk.
        """

        # Build file name path for stampe dictionary pickle
        if 'tmpdir' in self.params:
            filename = get_filename(self.params['tmpdir'],
                                    '',
                                    self.params['output_meds'],
                                    var=self.pointing.filter+'_'+str(self.pointing.dither),
                                    name2=str(self.pointing.sca)+'_'+str(self.rank),
                                    ftype='cPickle',
                                    overwrite=True)
            filename_ = get_filename(self.params['out_path'],
                                    'stamps',
                                    self.params['output_meds'],
                                    var=self.pointing.filter+'_'+str(self.pointing.dither),
                                    name2=str(self.pointing.sca)+'_'+str(self.rank),
                                    ftype='cPickle',
                                    overwrite=True)
            supernova_filename = get_filename(self.params['tmpdir'],
                                          '',
                                          self.params['output_meds'],
                                          var=self.pointing.filter+'_'+str(self.pointing.dither),
                                          name2=str(self.pointing.sca)+'_'+str(self.rank)+'_supernova',
                                          ftype='cPickle',
                                          overwrite=True)
            supernova_filename_ = get_filename(self.params['out_path'],
                                          'stamps',
                                          self.params['output_meds'],
                                          var=self.pointing.filter+'_'+str(self.pointing.dither),
                                          name2=str(self.pointing.sca)+'_'+str(self.rank)+'_supernova',
                                          ftype='cPickle',
                                          overwrite=True)
            star_filename = get_filename(self.params['tmpdir'],
                                          '',
                                          self.params['output_meds'],
                                          var=self.pointing.filter+'_'+str(self.pointing.dither),
                                          name2=str(self.pointing.sca)+'_'+str(self.rank)+'_star',
                                          ftype='cPickle',
                                          overwrite=True)
            star_filename_ = get_filename(self.params['out_path'],
                                          'stamps',
                                          self.params['output_meds'],
                                          var=self.pointing.filter+'_'+str(self.pointing.dither),
                                          name2=str(self.pointing.sca)+'_'+str(self.rank)+'_star',
                                          ftype='cPickle',
                                          overwrite=True)
        else:
            filename = get_filename(self.params['out_path'],
                                    'stamps',
                                    self.params['output_meds'],
                                    var=self.pointing.filter+'_'+str(self.pointing.dither),
                                    name2=str(self.pointing.sca)+'_'+str(self.rank),
                                    ftype='cPickle',
                                    overwrite=True)
            filename_ = None

            supernova_filename = get_filename(self.params['out_path'],
                                          'stamps',
                                          self.params['output_meds'],
                                          var=self.pointing.filter+'_'+str(self.pointing.dither),
                                          name2=str(self.pointing.sca)+'_'+str(self.rank)+'_supernova',
                                          ftype='cPickle',
                                          overwrite=True)
            supernova_filename_ = None
            
            star_filename = get_filename(self.params['out_path'],
                                          'stamps',
                                          self.params['output_meds'],
                                          var=self.pointing.filter+'_'+str(self.pointing.dither),
                                          name2=str(self.pointing.sca)+'_'+str(self.rank)+'_star',
                                          ftype='cPickle',
                                          overwrite=True)
            star_filename_ = None

        # Instantiate draw_image object. The input parameters, pointing object, modify_image object, truth catalog object, random number generator, logger, and galaxy & star indices are passed.
        # Instantiation defines some parameters, iterables, and image bounds, and creates an empty SCA image.
        self.draw_image = draw_image(self.params, self.pointing, self.modify_image, self.cats,  self.logger, rank=self.rank, comm=self.comm)

        t0 = time.time()
        t1 = time.time()
        index_table = None
        if self.cats.get_gal_length()!=0:#&(self.cats.get_star_length()==0):
            tmp,tmp_ = self.cats.get_gal_list()
            if len(tmp)!=0:
                # Build indexing table for MEDS making later
                index_table = np.empty(int(self.cats.get_gal_length()),dtype=[('ind',int), ('sca',int), ('dither',int), ('x',float), ('y',float), ('ra',float), ('dec',float), ('mag',float), ('stamp',int)])
                index_table['ind']=-999
                # Objects to simulate
                # Open pickler
                with io.open(filename, 'wb') as f :
                    i=0
                    pickler = pickle.Pickler(f,protocol=pickle.HIGHEST_PROTOCOL)
                    # gals = {}
                    # Empty storage dictionary for postage stamp information
                    print('Attempting to simulate '+str(len(tmp))+' galaxies for SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.')
                    gal_list = tmp
                    while True:
                        # Loop over all galaxies near pointing and attempt to simulate them.
                        g_ = None
                        self.draw_image.iterate_gal()
                        print(self.draw_image.ind,self.draw_image.gal_stamp_too_large)
                        if self.draw_image.gal_done:
                            break
                        # Store postage stamp output in dictionary
                        g_ = self.draw_image.retrieve_stamp()
                        #print(g_)
                        if g_ is not None:
                            # gals[self.draw_image.ind] = g_
                            #print(type(self.params['skip_stamps']),self.params['skip_stamps'])
                            if not self.params['skip_stamps']:
                                #print('test')
                                pickler.clear_memo()
                                pickler.dump(g_)
                            index_table['ind'][i]    = np.copy(g_['ind'])
                            index_table['x'][i]      = np.copy(g_['x'])
                            index_table['y'][i]      = np.copy(g_['y'])
                            index_table['ra'][i]     = np.copy(g_['ra'])
                            index_table['dec'][i]    = np.copy(g_['dec'])
                            index_table['mag'][i]    = np.copy(g_['mag'])
                            if g_['gal'] is not None:
                                print('.....yes',g_['ind'])
                                index_table['stamp'][i]  = np.copy(g_['stamp'])
                            else:
                                print('.....no',g_['ind'])
                                index_table['stamp'][i]  = 0
                            index_table['sca'][i]    = self.pointing.sca
                            index_table['dither'][i] = self.pointing.dither
                            i+=1
                            # if i%1000==0:
                            #     print('time',time.time()-t1)
                            #     t1 = time.time()
                            # g_.clear()

                    index_table = index_table[:i]
                if 'skip_stamps' in self.params:
                    if self.params['skip_stamps']:
                        os.remove(filename)
        print('galaxy time', time.time()-t0)


        t1 = time.time()
        index_table_star = None
        tmp,tmp_ = self.cats.get_star_list()
        if len(tmp)!=0:
            with io.open(star_filename, 'wb') as f :
                pickler = pickle.Pickler(f)
                index_table_star = np.empty(int(self.cats.get_star_length()),dtype=[('ind',int), ('sca',int), ('dither',int), ('x',float), ('y',float), ('ra',float), ('dec',float), ('mag',float), ('stamp',int)])
                index_table_star['ind']=-999
                print('Attempting to simulate '+str(len(tmp))+' stars for SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.')
                i=0
                while True:
                    # Loop over all stars near pointing and attempt to simulate them. Stars aren't saved in postage stamp form.
                    self.draw_image.iterate_star()
                    if self.draw_image.star_done:
                        break
                    s_ = self.draw_image.retrieve_star_stamp()
                    if s_ is not None:
                        pickler.clear_memo()
                        pickler.dump(s_)
                        index_table_star['ind'][i]    = s_['ind']
                        index_table_star['x'][i]      = s_['x']
                        index_table_star['y'][i]      = s_['y']
                        index_table_star['ra'][i]     = s_['ra']
                        index_table_star['dec'][i]    = s_['dec']
                        index_table_star['mag'][i]    = s_['mag']
                        index_table_star['sca'][i]    = self.pointing.sca
                        index_table_star['dither'][i] = self.pointing.dither
                        if s_['star'] is not None:
                            index_table_star['stamp'][i]  = s_['stamp']
                        else:
                            index_table_star['stamp'][i]  = 0
                        i+=1
                        s_.clear()
                index_table_star = index_table_star[:i]
        print('star time', time.time()-t1)

        index_table_sn = None
        if self.cats.supernovae is not None:
            tmp,tmp_ = self.cats.get_supernova_list()
            if tmp is not None:
                if len(tmp)!=0:
                    with io.open(supernova_filename, 'wb') as f :
                        pickler = pickle.Pickler(f)
                        index_table_sn = np.empty(int(self.cats.get_supernova_length()),dtype=[('ind',int), ('sca',int), ('dither',int), ('x',float), ('y',float), ('ra',float), ('dec',float), ('mag',float), ('hostid',int)])
                        index_table_sn['ind']=-999
                        print('Attempting to simulate '+str(len(tmp))+' supernovae for SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.')
                        i=0
                        while True:
                            # Loop over all supernovae near pointing and attempt to simulate them.
                            self.draw_image.iterate_supernova()
                            if self.draw_image.supernova_done:
                                break
                            s_ = self.draw_image.retrieve_supernova_stamp()
                            if s_ is not None:
                                pickler.dump(s_)
                                index_table_sn['ind'][i]    = s_['ind']
                                index_table_sn['x'][i]      = s_['x']
                                index_table_sn['y'][i]      = s_['y']
                                index_table_sn['ra'][i]     = s_['ra']
                                index_table_sn['dec'][i]    = s_['dec']
                                index_table_sn['mag'][i]    = s_['mag']
                                index_table_sn['sca'][i]    = self.pointing.sca
                                index_table_sn['dither'][i] = self.pointing.dither
                                index_table_sn['hostid'][i] = s_['hostid']
                                i+=1
                                s_.clear()
                        index_table_sn = index_table_sn[:i]

        self.comm.Barrier()

        if os.path.exists(filename):
            os.system('gzip '+filename)
            if filename_ is not None:
                shutil.copy(filename+'.gz',filename_+'.gz')
                os.remove(filename+'.gz')
        if os.path.exists(star_filename):
            os.system('gzip '+star_filename)
            if star_filename_ is not None:
                shutil.copy(star_filename+'.gz',star_filename_+'.gz')
                os.remove(star_filename+'.gz')
        if os.path.exists(supernova_filename):
            os.system('gzip '+supernova_filename)
            if supernova_filename_ is not None:
                shutil.copy(supernova_filename+'.gz',supernova_filename_+'.gz')
                os.remove(supernova_filename+'.gz')
        if self.rank == 0:
            # Build file name path for SCA image
            filename = get_filename(self.params['out_path'],
                                    'images',
                                    self.params['output_meds'],
                                    var=self.pointing.filter+'_'+str(self.pointing.dither),
                                    name2=str(self.pointing.sca),
                                    ftype='fits.gz',
                                    overwrite=True)

        self.comm.Barrier()
        print(self.rank,self.comm,flush=True)
        if self.comm is None:

            if (self.cats.get_gal_length()==0) and (len(gal_list)==0):
                return

            # No mpi, so just finalize the drawing of the SCA image and write it to a fits file.
            print('Saving SCA image to '+filename)
            write_fits(filename,self.draw_image.im,None,None,self.pointing.sca,self.params['output_meds'])
            #img = self.draw_image.finalize_sca()
            #write_fits(filename,img)

        else:

            # Send/receive all versions of SCA images across procs and sum them, then finalize and write to fits file.
            if self.rank == 0:
                for i in range(1,self.size):
                    self.draw_image.im = self.draw_image.im + self.comm.recv(source=i)

                if index_table is not None:
                    print('Saving SCA image to '+filename)
                    # self.draw_image.im.write(filename+'_raw.fits.gz')
                    write_fits(filename,self.draw_image.im,None,None,self.pointing.sca,self.params['output_meds'])

            else:

                self.comm.send(self.draw_image.im, dest=0)

            # Send/receive all parts of postage stamp dictionary across procs and merge them.
            # if self.rank == 0:

            #     for i in range(1,self.size):
            #         gals.update( self.comm.recv(source=i) )

            #     # Build file name path for stampe dictionary pickle
            #     filename = get_filename(self.params['out_path'],
            #                             'stamps',
            #                             self.params['output_meds'],
            #                             var=self.pointing.filter+'_'+str(self.pointing.dither),
            #                             name2=str(self.pointing.sca),
            #                             ftype='cPickle',
            #                             overwrite=True)

            #     if gals!={}:
            #         # Save stamp dictionary pickle
            #         print('Saving stamp dict to '+filename)
            #         save_obj(gals, filename )

            # else:

            #     self.comm.send(gals, dest=0)

        if self.rank == 0:

            filename = get_filename(self.params['out_path'],
                                    'truth',
                                    self.params['output_meds'],
                                    var='index',
                                    name2=self.pointing.filter+'_'+str(self.pointing.dither)+'_'+str(self.pointing.sca),
                                    ftype='fits',
                                    overwrite=True)
            filename_star = get_filename(self.params['out_path'],
                                    'truth',
                                    self.params['output_meds'],
                                    var='index',
                                    name2=self.pointing.filter+'_'+str(self.pointing.dither)+'_'+str(self.pointing.sca)+'_star',
                                    ftype='fits',
                                    overwrite=True)
            filename_sn = get_filename(self.params['out_path'],
                                    'truth',
                                    self.params['output_meds'],
                                    var='index',
                                    name2=self.pointing.filter+'_'+str(self.pointing.dither)+'_'+str(self.pointing.sca)+'_sn',
                                    ftype='fits',
                                    overwrite=True)  

            print('before index')
            for i in range(1,self.size):
                tmp = self.comm.recv(source=i)
                if tmp is not None:
                    index_table = np.append(index_table,tmp)
            for i in range(1,self.size):
                tmp = self.comm.recv(source=i)
                if tmp is not None:
                    index_table_star = np.append(index_table_star,tmp)
            for i in range(1,self.size):
                tmp = self.comm.recv(source=i)
                if tmp is not None:
                    index_table_sn = np.append(index_table_sn,tmp)

            if index_table is not None:
                print('Saving index to '+filename)
                fio.write(filename,index_table)
            else: 
                print('Not saving index, no objects in SCA')
            if index_table_star is not None:
                fio.write(filename_star,index_table_star)
            if index_table_sn is not None:
                fio.write(filename_sn,index_table_sn)
        else:
            self.comm.send(index_table, dest=0)            
            self.comm.send(index_table_star, dest=0)            
            self.comm.send(index_table_sn, dest=0)            

    def iterate_image_detector(self):
        """
        This is the main simulation. It instantiates the draw_image object, then iterates over all galaxies and stars. The output is then accumulated from other processes (if mpi is enabled), and saved to disk.
        """

        # Build file name path for stampe dictionary pickle
        if 'tmpdir' in self.params:
            filename = get_filename(self.params['tmpdir'],
                                    '',
                                    self.params['output_meds'],
                                    var=self.pointing.filter+'_'+str(self.pointing.dither),
                                    name2=str(self.pointing.sca),
                                    ftype='cPickle',
                                    overwrite=True)
            filename_ = get_filename(self.params['out_path'],
                                    'stamps/final',
                                    self.params['output_meds'],
                                    var=self.pointing.filter+'_'+str(self.pointing.dither),
                                    name2=str(self.pointing.sca),
                                    ftype='cPickle',
                                    overwrite=True)
            supernova_filename = get_filename(self.params['tmpdir'],
                                          '',
                                          self.params['output_meds'],
                                          var=self.pointing.filter+'_'+str(self.pointing.dither),
                                          name2=str(self.pointing.sca)+'_supernova',
                                          ftype='cPickle',
                                          overwrite=True)
            supernova_filename_ = get_filename(self.params['out_path'],
                                          'stamps/final',
                                          self.params['output_meds'],
                                          var=self.pointing.filter+'_'+str(self.pointing.dither),
                                          name2=str(self.pointing.sca)+'_supernova',
                                          ftype='cPickle',
                                          overwrite=True)
            star_filename = get_filename(self.params['tmpdir'],
                                          '',
                                          self.params['output_meds'],
                                          var=self.pointing.filter+'_'+str(self.pointing.dither),
                                          name2=str(self.pointing.sca)+'_star',
                                          ftype='cPickle',
                                          overwrite=True)
            star_filename_ = get_filename(self.params['out_path'],
                                          'stamps/final',
                                          self.params['output_meds'],
                                          var=self.pointing.filter+'_'+str(self.pointing.dither),
                                          name2=str(self.pointing.sca)+'_star',
                                          ftype='cPickle',
                                          overwrite=True)
        else:
            filename = get_filename(self.params['out_path'],
                                    'stamps/final',
                                    self.params['output_meds'],
                                    var=self.pointing.filter+'_'+str(self.pointing.dither),
                                    name2=str(self.pointing.sca),
                                    ftype='cPickle',
                                    overwrite=True)
            filename_ = None

            supernova_filename = get_filename(self.params['out_path'],
                                          'stamps/final',
                                          self.params['output_meds'],
                                          var=self.pointing.filter+'_'+str(self.pointing.dither),
                                          name2=str(self.pointing.sca)+'_supernova',
                                          ftype='cPickle',
                                          overwrite=True)
            supernova_filename_ = None
            
            star_filename = get_filename(self.params['out_path'],
                                          'stamps/final',
                                          self.params['output_meds'],
                                          var=self.pointing.filter+'_'+str(self.pointing.dither),
                                          name2=str(self.pointing.sca)+'_star',
                                          ftype='cPickle',
                                          overwrite=True)
            star_filename_ = None

        # Instantiate draw_detector object. The input parameters, pointing object, modify_image object, truth catalog object, random number generator, logger, and galaxy & star indices are passed.
        # Instantiation defines some parameters, iterables, and image bounds, and creates an empty SCA image.
        if self.rank == 0:
            # Build file name path for SCA image
            imfilename = get_filename(self.params['out_path'],
                                    'images',
                                    self.params['output_meds'],
                                    var=self.pointing.filter+'_'+str(self.pointing.dither),
                                    name2=str(self.pointing.sca),
                                    ftype='fits.gz',
                                    overwrite=False)
            im = fio.FITS(imfilename)['SCI'].read()
            imfilename = get_filename(self.params['out_path'],
                                    'images/final',
                                    self.params['output_meds'],
                                    var=self.pointing.filter+'_'+str(self.pointing.dither),
                                    name2=str(self.pointing.sca),
                                    ftype='fits.gz',
                                    overwrite=True)
            self.draw_image = draw_detector(self.params, self.pointing, self.modify_image,  self.logger, rank=self.rank, comm=self.comm, im=im)
            img,err,dq = self.draw_image.finalize_sca()
            write_fits(imfilename,img,err,dq,self.pointing.sca,self.params['output_meds'])
            print('done image detector stuff')

        self.comm.Barrier()

        t0 = time.time()
        with io.open(filename, 'wb') as f :
            i=0
            pickler = pickle.Pickler(f)
            while True:
                filename1 = get_filenames(self.params['out_path'],
                                            'stamps',
                                            self.params['output_meds'],
                                            var=self.pointing.filter+'_'+str(stamps_used['dither'][s]),
                                            name2=str(stamps_used['sca'][s])+'_',
                                            exclude='star',
                                            ftype='cPickle.gz')
                for f in filename1:
                    stampsfilename=f.replace(self.params['out_path']+'stamps/', self.params['tmpdir'])
                    shutil.copy(f, stampsfilename)
                    os.system('gunzip '+stampsfilename)
                    stampsfilename=stampsfilename.replace('.gz', '')
                    with io.open(stampsfilename, 'rb') as p :
                        unpickler = pickle.Unpickler(p)
                        while p.peek(1) :
                            gal = unpickler.load()
                            if (gal is None) or (gal['gal'] is None):
                                pickler.dump(gal)
                                continue

                            img,err,dq = self.draw_image.finalize_stamp(gal)
                            gal['gal']    = img
                            gal['weight'] = err
                            gal['dq']     = dq

                            pickler.dump(gal)

                break

        if os.path.exists(filename):
            os.system('gzip '+filename)
            if filename_ is not None:
                shutil.copy(filename+'.gz',filename_+'.gz')
                os.remove(filename+'.gz')


        t0 = time.time()
        with io.open(star_filename, 'wb') as f :
            i=0
            pickler = pickle.Pickler(f)
            while True:
                filename1 = get_filenames(self.params['out_path'],
                                            'stamps',
                                            self.params['output_meds'],
                                            var=self.pointing.filter+'_'+str(stamps_used['dither'][s]),
                                            name2=str(stamps_used['sca'][s])+'_',
                                            ftype='_star.cPickle.gz')
                for f in filename1:
                    stampsfilename=f.replace(self.params['out_path']+'stamps/', self.params['tmpdir'])
                    shutil.copy(f, stampsfilename)
                    os.system('gunzip '+stampsfilename)
                    stampsfilename=stampsfilename.replace('.gz', '')
                    with io.open(stampsfilename, 'rb') as p :
                        unpickler = pickle.Unpickler(p)
                        while p.peek(1) :
                            gal = unpickler.load()
                            if (gal is None) or (gal['star'] is None):
                                pickler.dump(gal)
                                continue

                            img,err,dq = self.draw_image.finalize_stamp(gal,obj='star')
                            gal['star']   = img
                            gal['weight'] = err
                            gal['dq']     = dq

                            pickler.dump(gal)

                break

        if os.path.exists(star_filename):
            os.system('gzip '+star_filename)
            if star_filename_ is not None:
                shutil.copy(star_filename+'.gz',star_filename_+'.gz')
                os.remove(star_filename+'.gz')


    def check_file(self,sca,dither,filter_):
        self.pointing = pointing(self.params,self.logger,filter_=None,sca=None,dither=int(dither),rank=self.rank)
        print(sca,dither,filter_)
        f = get_filename(self.params['out_path'],
                                    'truth',
                                    self.params['output_meds'],
                                    var='index',
                                    name2=self.pointing.filter+'_'+str(dither)+'_'+str(sca),
                                    ftype='fits',
                                    overwrite=False)
        print(f)
        return os.path.exists(f)