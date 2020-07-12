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
from astropy.io import fits
import pickle as pickle
import pickletools
from astropy.time import Time
from mpi4py import MPI
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

class init_catalogs(object):
    """
    Build truth catalogs if they don't exist from input galaxy and star catalogs.
    """


    def __init__(self, params, pointing, gal_rng, rank, size, comm=None, setup=False):
        """
        Initiate the catalogs

        Input
        params   : Parameter dictionary
        pointing : Pointing object
        gal_rng  : Random generator [0,1]
        rank     : Process rank
        comm     : MPI comm object
        """

        self.pointing = pointing
        self.rank = rank
        if rank == 0:
            # Set up file path. Check if output truth file path exists or if explicitly remaking galaxy properties
            filename = get_filename(params['out_path'],
                                    'truth',
                                    params['output_truth'],
                                    name2='truth_gal',
                                    overwrite=params['overwrite'])

            # Link to galaxy truth catalog on disk
            self.gals  = self.init_galaxy(filename,params,pointing,gal_rng,setup)
            # Link to star truth catalog on disk
            self.stars = self.init_star(params)
            # Link to supernova truth catalog on disk
            self.supernovae,self.lightcurves = self.init_supernova(params)
            if setup:
                comm.Barrier()
                return

            self.init_sed(params)

            # print 'gal check',len(self.gals['ra'][:]),len(self.stars['ra'][:]),np.degrees(self.gals['ra'][:].min()),np.degrees(self.gals['ra'][:].max()),np.degrees(self.gals['dec'][:].min()),np.degrees(self.gals['dec'][:].max())

            if comm is not None:
                # Pass gal_ind to other procs
                self.get_near_sca()
                # print 'gal check',len(self.gals['ra'][:]),len(self.stars['ra'][:]),np.degrees(self.gals['ra'][:].min()),np.degrees(self.gals['ra'][:].max()),np.degrees(self.gals['dec'][:].min()),np.degrees(self.gals['dec'][:].max())

                for i in range(1,size):
                    comm.send(self.gal_ind,  dest=i)
                    comm.send(self.gals,  dest=i)

                # Pass star_ind to other procs
                for i in range(1,size):
                    comm.send(self.star_ind,  dest=i)
                    comm.send(self.stars,  dest=i)

                # Pass seds to other procs
                for i in range(1,size):
                    comm.send(self.seds,  dest=i)

                # Pass sne to other procs
                for i in range(1,size):
                    comm.send(self.supernova_ind, dest=i)
                    comm.send(self.supernovae, dest=i)
                    comm.send(self.lightcurves, dest=i)
        else:
            if setup:
                comm.Barrier()
                return

            # Get gals
            self.gal_ind = comm.recv(source=0)
            self.gals = comm.recv(source=0)

            # Get stars
            self.star_ind = comm.recv(source=0)
            self.stars = comm.recv(source=0)

            # Get seds
            self.seds = comm.recv(source=0)

            # Get sne
            self.supernova_ind = comm.recv(source=0)
            self.supernovae = comm.recv(source=0)
            self.lightcurves = comm.recv(source=0)

        self.gal_ind  = self.gal_ind[rank::size]
        self.gals     = self.gals[rank::size]
        if rank>=params['starproc']:
            self.star_ind=[]
            self.stars=[]
        else:
            self.star_ind = self.star_ind[rank::params['starproc']]
            self.stars    = self.stars[rank::params['starproc']]
        if self.supernovae is not None:
            self.supernova_ind = self.supernova_ind[rank::size]
            self.supernovae = self.supernovae[rank::size]

    def close(self):

        self.gal_ind  = None
        self.gals     = None
        self.star_ind = None
        self.stars    = None
        self.supernova_ind = None
        self.supernovae = None
        self.lightcurves = None

    def get_near_sca(self):

        self.gal_ind  = self.pointing.near_pointing( self.gals['ra'][:], self.gals['dec'][:] )
        # print len(self.gal_ind),len(self.gals['ra'][:])
        if len(self.gal_ind)==0:
            self.gal_ind = []
            self.gals = []
        else:
            self.gals = self.gals[self.gal_ind]

        self.star_ind = self.pointing.near_pointing( self.stars['ra'][:], self.stars['dec'][:] )
        # print len(self.star_ind),len(self.stars['ra'][:])
        if len(self.star_ind)==0:
            self.star_ind = []
            self.stars = []
        else:
            self.stars = self.stars[self.star_ind]

        mask_sca      = self.pointing.in_sca(self.gals['ra'][:],self.gals['dec'][:])
        if len(mask_sca)==0:
            self.gal_ind = []
            self.gals = []
        else:
            self.gals    = self.gals[mask_sca]
            self.gal_ind = self.gal_ind[mask_sca]

        mask_sca_star = self.pointing.in_sca(self.stars['ra'][:],self.stars['dec'][:])
        if len(mask_sca_star)==0:
            self.star_ind = []
            self.stars = []
        else:   
            self.stars    = self.stars[mask_sca_star]
            self.star_ind = self.star_ind[mask_sca_star]

        if self.supernovae is not None:
            self.supernova_ind = self.pointing.near_pointing( self.supernovae['ra'][:], 
                                                            self.supernovae['dec'][:], 
                                                            min_date=self.lightcurves['mjd'][self.supernovae['ptrobs_min']][:], 
                                                            max_date=self.lightcurves['mjd'][self.supernovae['ptrobs_max'] - 1][:]) 
            self.supernovae = self.supernovae[self.supernova_ind]
        else: 
            self.supernova_ind = None

    def add_mask(self,gal_mask,star_mask=None,supernova_mask=None):

        if gal_mask.dtype == bool:
            self.gal_mask = np.where(gal_mask)[0]
        else:
            self.gal_mask = gal_mask

        if star_mask is None:
            self.star_mask = []
        elif star_mask.dtype == bool:
            self.star_mask = np.where(star_mask)[0]
        else:
            self.star_mask = star_mask
        if supernova_mask is None:
            self.supernova_mask = []
        elif supernova_mask.dtype == bool:
            self.supernova_mask = np.where(supernova_mask)[0]
        else:
            self.supernova_mask = supernova_mask

    def get_gal_length(self):

        return len(self.gal_ind)

    def get_star_length(self):

        return len(self.star_ind)

    def get_supernova_length(self):
        
        return len(self.supernova_ind)

    def get_gal_list(self):

        return self.gal_ind,self.gals
        return self.gal_ind[self.gal_mask],self.gals[self.gal_mask]

    def get_star_list(self):

        return self.star_ind,self.stars
        return self.star_ind[self.star_mask],self.stars[self.star_mask]

    def get_supernova_list(self):
        
        return self.supernova_ind,self.supernovae
        return self.supernova_ind[self.supernova_mask],self.supernovae[self.supernova_mask]

    def get_gal(self,ind):

        return self.gal_ind[ind],self.gals[ind]
        return self.gal_ind[self.gal_mask[ind]],self.gals[self.gal_mask[ind]]

    def get_star(self,ind):

        return self.star_ind[ind],self.stars[ind]
        return self.star_ind[self.star_mask[ind]],self.stars[self.star_mask[ind]]

    def get_supernova(self,ind):

        return self.supernova_ind[ind],self.supernovae[ind]
        return self.supernova_ind[self.supernova_mask[ind]],self.supernovae[self.supernova_mask[ind]]

    def dump_truth_gal(self,filename,store):
        """
        Write galaxy truth catalog to disk.

        Input
        filename    : Fits filename
        store       : Galaxy truth catalog
        """

        fio.write(filename,store,clobber=True)

        return fio.FITS(filename)[-1]

    def load_truth_gal(self,filename,params):
        """
        Load galaxy truth catalog from disk.

        Input
        filename    : Fits filename
        """

        if 'tmpdir' in params:
            filename2 = get_filename(params['tmpdir'],
                                    '',
                                    params['output_truth'],
                                    name2='truth_gal',
                                    overwrite=params['overwrite'])
            if not params['overwrite']:
                if not os.path.exists(filename2):
                    shutil.copy(filename,filename2, follow_symlinks=True)
            else:
                shutil.copy(filename,filename2, follow_symlinks=True)

        store = fio.FITS(filename2)[-1]

        return store

    def fwhm_to_hlr(self,fwhm):
        """
        Convert full-width half-maximum to half-light radius in units of arcseconds.

        Input
        fwhm : full-width half-maximum
        """

        radius = fwhm * 0.06 / 2. # 1 pix = 0.06 arcsec, factor 2 to convert to hlr

        return radius

    def init_galaxy(self,filename,params,pointing,gal_rng,setup):
        """
        Does the work to return a random, unique object property list (truth catalog).

        Input
        filname  : Filename of galaxy truth catalog.
        params   : Parameter dict
        pointing : pointing object
        gal_rng  : Random generator [0,1]
        """

        # This is a placeholder option to allow different galaxy simulatin methods later if necessary
        if params['gal_type'] == 0:
            # Analytic profile - sersic disk


            if not setup:
                if os.path.exists(filename):
                    # Truth file exists and no instruction to overwrite it, so load existing truth file with galaxy properties
                    return self.load_truth_gal(filename,params)
                else:
                    raise ParamError('No truth file to load.')

            if (not params['overwrite']) and (os.path.exists(filename)):
                print('Reusing existing truth file.')
                return None

            # Make sure galaxy distribution filename is well-formed and link to it
            if isinstance(params['gal_dist'],str):
                # Provided an ra,dec catalog of object positions.
                radec_file = fio.FITS(params['gal_dist'])[-1]
            else:
                raise ParamError('Bad gal_dist filename.')

            print('-----building truth catalog------')
            # Read in file with photometry/size/redshift distribution similar to WFIRST galaxies
            phot       = fio.FITS(params['gal_sample'])[-1].read(columns=['fwhm','redshift',filter_flux_dict['J129'],filter_flux_dict['F184'],filter_flux_dict['Y106'],filter_flux_dict['H158']])
            pind_list_ = np.ones(len(phot)).astype(bool) # storage list for original index of photometry catalog
            pind_list_ = pind_list_&(phot[filter_flux_dict['J129']]<99)&(phot[filter_flux_dict['J129']]>0) # remove bad mags
            pind_list_ = pind_list_&(phot[filter_flux_dict['F184']]<99)&(phot[filter_flux_dict['F184']]>0) # remove bad mags
            pind_list_ = pind_list_&(phot[filter_flux_dict['Y106']]<99)&(phot[filter_flux_dict['Y106']]>0) # remove bad mags
            pind_list_ = pind_list_&(phot[filter_flux_dict['H158']]<99)&(phot[filter_flux_dict['H158']]>0) # remove bad mags
            pind_list_ = pind_list_&(phot['redshift']>0)&(phot['redshift']<5) # remove bad redshifts
            pind_list_ = np.where(pind_list_)[0]

            n_gal = radec_file.read_header()['NAXIS2']

            # Create minimal storage array for galaxy properties
            store = np.ones(n_gal, dtype=[('gind','i4')]
                                        +[('ra',float)]
                                        +[('dec',float)]
                                        +[('g1','f4')]
                                        +[('g2','f4')]
                                        +[('int_e1','f4')]
                                        +[('int_e2','f4')]
                                        +[('rot','f4')]
                                        +[('size','f4')]
                                        +[('z','f4')]
                                        +[('J129','f4')]
                                        +[('F184','f4')]
                                        +[('Y106','f4')]
                                        +[('H158','f4')]
                                        +[('pind','i4')]
                                        +[('bflux','f4')]
                                        +[('dflux','f4')])
            store['gind']       = np.arange(n_gal) # Index array into original galaxy position catalog
            store['ra']         = radec_file['ra'][:]*np.pi/180. # Right ascension
            store['dec']        = radec_file['dec'][:]*np.pi/180. # Declination
            r_ = np.zeros(n_gal)
            gal_rng.generate(r_)
            store['pind']       = pind_list_[(r_*len(pind_list_)).astype(int)] # Index array into original galaxy photometry catalog
            r_ = np.zeros(int(n_gal/2)+n_gal%2)
            gal_rng.generate(r_)
            store['rot'][0::2]  = r_*2.*np.pi # Random rotation (every pair of objects is rotated 90 deg to cancel shape noise)
            store['rot'][1::2]  = store['rot'][0:n_gal-n_gal%2:2]+np.pi
            store['rot'][store['rot']>2.*np.pi]-=2.*np.pi
            r_ = np.zeros(n_gal)
            gal_rng.generate(r_)
            r_ = (r_*len(params['shear_list'])).astype(int)
            np.random.seed(seed=params['random_seed'])
            store['g1']         = np.array(params['shear_list'])[r_,0] # Shears to apply to galaxy
            store['g2']         = np.array(params['shear_list'])[r_,1]
            store['int_e1']     = np.random.normal(scale=0.27,size=n_gal) # Intrinsic shape of galaxy
            store['int_e2']     = np.random.normal(scale=0.27,size=n_gal)
            store['int_e1'][store['int_e1']>0.7] = 0.7
            store['int_e2'][store['int_e2']>0.7] = 0.7
            store['int_e1'][store['int_e1']<-0.7] = -0.7
            store['int_e2'][store['int_e2']<-0.7] = -0.7
            if params['gal_model'] == 'disk': # Disk only model, no bulge or knot flux
                store['bflux']  = np.zeros(n_gal)
                store['dflux']  = np.ones(n_gal)
            elif params['gal_model'] == 'bulge': # Bulge only model, no disk or knot flux
                store['bflux']  = np.ones(n_gal)
                store['dflux']  = np.zeros(n_gal)
            else: # General composite model. bflux = bulge flux fraction. dflux*(1-bflux) = disk flux fraction. Remaining flux is in form of star-knots, (1-bflux)*(1-dflux). Knot flux is capped at 50% of disk flux.
                r_ = np.zeros(n_gal)
                gal_rng.generate(r_)
                store['bflux']  = r_
                r_ = np.zeros(n_gal)
                gal_rng.generate(r_)
                store['dflux']  = r_/4.+0.75
            store['size']       = self.fwhm_to_hlr(phot['fwhm'][store['pind']]) # half-light radius
            store['z']          = phot['redshift'][store['pind']] # redshift
            for f in list(filter_dither_dict.keys()):
                store[f]        = phot[filter_flux_dict[f]][store['pind']] # magnitude in this filter
            for name in store.dtype.names:
                print(name,np.mean(store[name]),np.min(store[name]),np.max(store[name]))

            # Save truth file with galaxy properties
            return self.dump_truth_gal(filename,store)

            print('-------truth catalog built-------')

        else:
            raise ParamError('COSMOS profiles not currently implemented.')
            # cosmos gal not guaranteed to work. uncomment at own risk
            # # Cosmos real or parametric objects
            # if self.params['gal_type'] == 1:
            #     use_real = False
            #     gtype = 'parametric'
            # else:
            #     use_real = True
            #     gtype = 'real'

            # # Load cosmos catalog
            # cat = galsim.COSMOSCatalog(self.params['cat_name'], dir=self.params['cat_dir'], use_real=use_real)
            # self.logger.info('Read in %d galaxies from catalog'%cat.nobjects)

            # rand_ind = []
            # for i in range(self.params['gal_n_use']):
            #     # Select unique cosmos index list with length gal_n_use.
            #     rand_ind.append(int(self.gal_rng()*cat.nobjects))
            # # Make object list of unique cosmos galaxies
            # self.obj_list = cat.makeGalaxy(rand_ind, chromatic=True, gal_type=gtype)

    def init_star(self,params):
        """
        Compiles a list of stars properties to draw.
        Not working with new structure yet.

        Input
        params   : parameter dict
        """

        # Make sure star catalog filename is well-formed and link to it
        if isinstance(params['star_sample'],str):
            # Provided a catalog of star positions and properties.
            if 'tmpdir' in params:
                filename2 = get_filename(params['tmpdir'],
                                        '',
                                        params['output_truth'],
                                        name2='truth_star',
                                        overwrite=params['overwrite'])
                if not params['overwrite']:
                    if not os.path.exists(filename2):
                        shutil.copy(params['star_sample'],filename2, follow_symlinks=True)
                else:
                    shutil.copy(params['star_sample'],filename2, follow_symlinks=True)
                stars = fio.FITS(filename2)[-1]
            else:            
                stars = fio.FITS(params['star_sample'])[-1]
            self.n_star = stars.read_header()['NAXIS2']
        else:
            return None

        # # Cut really bright stars that take too long to draw for now
        # mask = np.ones(len(stars),dtype=bool)
        # for f in filter_dither_dict.keys():
        #     mask = mask & (stars_[f]<1e5)
        # stars = stars[mask]

        return stars

    def init_supernova(self,params):
        if 'supernovae' not in params:
            return None,None
        if isinstance(params['supernovae'],str):
            # Given a lightcurve Phot.fits file.
            with fits.open(params['supernovae'] + "_HEAD.FITS") as sn:
                supernovae = sn[1].data
                supernovae['ra'] = supernovae['ra'] * np.pi / 180
                supernovae['dec'] = supernovae['dec'] * np.pi / 180
                self.n_supernova = sn[1].header['NAXIS2']
            with fits.open(params['supernovae'] + "_PHOT.FITS") as light:
                lightcurves = light[1].data
        else:
            return None,None
        return supernovae,lightcurves

    def init_sed(self,params):
        """
        Loads the relevant SEDs into memory

        Input 
        params   : parameter dict
        """

        self.seds = {}
        if not params['dc2']:
            return None

        filename = get_filename(params['out_path'],
                                'truth',
                                params['output_truth'],
                                name2='truth_sed',
                                overwrite=False, ftype='h5')

        if 'tmpdir' in params:
            filename2 = get_filename(params['tmpdir'],
                                '',
                                params['output_truth'],
                                name2='truth_sed',
                                overwrite=False, ftype='h5')
            if not params['overwrite']:
                if not os.path.exists(filename2):
                    shutil.copy(filename,filename2, follow_symlinks=True)
            else:
                shutil.copy(filename,filename2, follow_symlinks=True)
        else:
            filename2 = filename

        sedfile = h5py.File(filename2,mode='r')

        print(self.gals['sed'])
        for s in np.unique(self.gals['sed']):
            print('gal',s)
            if s=='':
                continue
            self.seds[s] = sedfile[s.lstrip().rstrip()][:]

        for s in np.unique(self.stars['sed'][:]):
            print('star',s)
            if s=='':
                continue
            self.seds[s] = sedfile[s.lstrip().rstrip()][:]

        return self.seds


def setupCCM_ab(wavelen):
    """
    Calculate a(x) and b(x) for CCM dust model. (x=1/wavelen).
    If wavelen not specified, calculates a and b on the own object's wavelength grid.
    Returns a(x) and b(x) can be common to many seds, wavelen is the same.
    This method sets up extinction due to the model of
    Cardelli, Clayton and Mathis 1989 (ApJ 345, 245)

    Taken tempoarily for testing from https://github.com/lsst/sims_photUtils/blob/master/python/lsst/sims/photUtils/Sed.py
    """
    # This extinction law taken from Cardelli, Clayton and Mathis ApJ 1989.
    # The general form is A_l / A(V) = a(x) + b(x)/R_V  (where x=1/lambda in microns),
    # then different values for a(x) and b(x) depending on wavelength regime.
    # Also, the extinction is parametrized as R_v = A_v / E(B-V).
    # Magnitudes of extinction (A_l) translates to flux by a_l = -2.5log(f_red / f_nonred).
    a_x = np.zeros(len(wavelen), dtype='float')
    b_x = np.zeros(len(wavelen), dtype='float')
    # Convert wavelength to x (in inverse microns).
    x = np.empty(len(wavelen), dtype=float)
    nm_to_micron = 1/1000.0
    x = 1.0 / (wavelen * nm_to_micron)
    # Dust in infrared 0.3 /mu < x < 1.1 /mu (inverse microns).
    condition = (x >= 0.3) & (x <= 1.1)
    if len(a_x[condition]) > 0:
        y = x[condition]
        a_x[condition] = 0.574 * y**1.61
        b_x[condition] = -0.527 * y**1.61
    # Dust in optical/NIR 1.1 /mu < x < 3.3 /mu region.
    condition = (x >= 1.1) & (x <= 3.3)
    if len(a_x[condition]) > 0:
        y = x[condition] - 1.82
        a_x[condition] = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4
        a_x[condition] = a_x[condition] + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
        b_x[condition] = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4
        b_x[condition] = b_x[condition] - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    # Dust in ultraviolet and UV (if needed for high-z) 3.3 /mu< x< 8 /mu.
    condition = (x >= 3.3) & (x < 5.9)
    if len(a_x[condition]) > 0:
        y = x[condition]
        a_x[condition] = 1.752 - 0.316*y - 0.104/((y-4.67)**2 + 0.341)
        b_x[condition] = -3.090 + 1.825*y + 1.206/((y-4.62)**2 + 0.263)
    condition = (x > 5.9) & (x < 8)
    if len(a_x[condition]) > 0:
        y = x[condition]
        Fa_x = np.empty(len(a_x[condition]), dtype=float)
        Fb_x = np.empty(len(a_x[condition]), dtype=float)
        Fa_x = -0.04473*(y-5.9)**2 - 0.009779*(y-5.9)**3
        Fb_x = 0.2130*(y-5.9)**2 + 0.1207*(y-5.9)**3
        a_x[condition] = 1.752 - 0.316*y - 0.104/((y-4.67)**2 + 0.341) + Fa_x
        b_x[condition] = -3.090 + 1.825*y + 1.206/((y-4.62)**2 + 0.263) + Fb_x
    # Dust in far UV (if needed for high-z) 8 /mu < x < 10 /mu region.
    condition = (x >= 8) & (x <= 11.)
    if len(a_x[condition]) > 0:
        y = x[condition]-8.0
        a_x[condition] = -1.073 - 0.628*(y) + 0.137*(y)**2 - 0.070*(y)**3
        b_x[condition] = 13.670 + 4.257*(y) - 0.420*(y)**2 + 0.374*(y)**3
    return a_x, b_x

def addDust(a_x, b_x, A_v=None, ebv=None, R_v=3.1):
    """
    Add dust model extinction to the SED, modifying flambda and fnu.
    Get a_x and b_x either from setupCCMab or setupODonnell_ab
    Specify any two of A_V, E(B-V) or R_V (=3.1 default).

    Taken tempoarily for testing from https://github.com/lsst/sims_photUtils/blob/master/python/lsst/sims/photUtils/Sed.py
    """
    _ln10_04 = 0.4*np.log(10.0)

    # The extinction law taken from Cardelli, Clayton and Mathis ApJ 1989.
    # The general form is A_l / A(V) = a(x) + b(x)/R_V  (where x=1/lambda in microns).
    # Then, different values for a(x) and b(x) depending on wavelength regime.
    # Also, the extinction is parametrized as R_v = A_v / E(B-V).
    # The magnitudes of extinction (A_l) translates to flux by a_l = -2.5log(f_red / f_nonred).
    #
    # Input parameters for reddening can include any of 3 parameters; only 2 are independent.
    # Figure out what parameters were given, and see if self-consistent.
    if R_v == 3.1:
        if A_v is None:
            A_v = R_v * ebv
        elif (A_v is not None) and (ebv is not None):
            # Specified A_v and ebv, so R_v should be nondefault.
            R_v = A_v / ebv
    if (R_v != 3.1):
        if (A_v is not None) and (ebv is not None):
            calcRv = A_v / ebv
            if calcRv != R_v:
                raise ValueError("CCM parametrization expects R_v = A_v / E(B-V);",
                                 "Please check input values, because values are inconsistent.")
        elif A_v is None:
            A_v = R_v * ebv
    # R_v and A_v values are specified or calculated.

    A_lambda = (a_x + b_x / R_v) * A_v
    # dmag_red(dust) = -2.5 log10 (f_red / f_nored) : (f_red / f_nored) = 10**-0.4*dmag_red
    dust = np.exp(-A_lambda*_ln10_04)
    return dust
