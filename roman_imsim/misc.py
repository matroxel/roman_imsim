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

class ParamError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

def except_func(logger, proc, k, res, t):
    print(proc, k)
    print(t)
    raise res

def save_obj(obj, name ):
    """
    Helper function to save some data as a pickle to disk.
    """
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    """
    Helper function to read some data from a pickle on disk.
    """
    with open(name, 'rb') as f:
        return pickle.load(f)

def convert_dither_to_fits(ditherfile='observing_sequence_hlsonly'):
    """
    Helper function to used to convert Chris survey dither file to fits and extract HLS part.
    """

    dither = np.genfromtxt(ditherfile+'.dat',dtype=None,names = ['date','exptime','constraint','ra','dec','pa','program','filter','gal_lon','gal_lat','ec_lon','ec_lat','sun_ang_los','rol_ang_sun','earth_ang_los','roll_ang_earth','zodi_bright','EBV','slew_ang_from','slew_ang_to','yaw','comment'])
    dither = dither[['date','ra','dec','pa','filter']]
    fio.write(ditherfile+'.fits',dither,clobber=True)

    return

def convert_gaia():
    """
    Helper function to convert gaia data to star truth catalog.
    """

    n=100000000
    ra=[-5,80.]
    dec=[-64,3]
    ra1 = np.random.rand(n)*(ra[1]-ra[0])/180.*np.pi+ra[0]/180.*np.pi
    d0 = (np.cos((dec[0]+90)/180.*np.pi)+1)/2.
    d1 = (np.cos((dec[1]+90)/180.*np.pi)+1)/2.
    dec1 = np.arccos(2*np.random.rand(n)*(d1-d0)+2*d0-1)
    out = np.empty(n,dtype=[('ra',float)]+[('dec',float)]+[('H158',float)]+[('J129',float)]+[('Y106',float)]+[('F184',float)])
    out['ra']=ra1
    out['dec']=dec1-np.pi/2.

    g_band     = galsim.Bandpass('/users/PCON0003/cond0083/GalSim/galsim/share/bandpasses/gaia_g.dat', wave_type='nm').withZeropoint('AB')
    star_sed   = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')

    gaia = fio.FITS('../distwf-result.fits.gz')[-1].read()['phot_g_mean_mag'][:]
    h,b = np.histogram(gaia,bins=np.linspace(3,22.5,196))
    b = (b[1:]+b[:-1])/2
    x = np.random.choice(np.arange(len(b)),len(out),p=1.*h/np.sum(h),replace=True)
    for i,filter_ in enumerate(['J129','F184','Y106', 'H158']):
        print(filter_)
        bpass = roman.getBandpasses(AB_zeropoint=True)[filter_]
        b_=np.zeros(len(b))
        for ind in range(len(b)):
            star_sed_  = star_sed.withMagnitude(b[ind],g_band)
            b_[ind]    = star_sed_.calculateMagnitude(bpass)
        out[filter_]   = b_[x]

    fio.write('gaia_stars.fits',out,clobber=True)

    return

def convert_galaxia():
    """
    Helper function to convert galaxia data to star truth catalog.
    """

    j_band     = galsim.Bandpass('/users/PCON0003/cond0083/GalSim/galsim/share/bandpasses/UKIRT_UKIDSS.J.dat.txt', wave_type='nm').withZeropoint('AB')
    star_sed   = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')

    g = fio.FITS('/users/PCON0003/cond0083/galaxia_stars.fits')[-1].read()
    out = np.empty(len(g),dtype=[('ra',float)]+[('dec',float)]+[('H158',float)]+[('J129',float)]+[('Y106',float)]+[('F184',float)])
    out['ra']=g['ra']
    out['dec']=g['dec']
    for i,filter_ in enumerate(['J129','F184','Y106','H158']):
        print(filter_)
        bpass = roman.getBandpasses(AB_zeropoint=True)[filter_]
        star_sed_  = star_sed.withMagnitude(23,j_band)
        factor    = star_sed_.calculateMagnitude(bpass)-23
        out[filter_] = g['J']+factor

    s = np.random.choice(np.arange(len(out)),len(out),replace=False)
    out=out[s]
    fio.write('galaxia_stars_full.fits',out,clobber=True)

    return

def create_radec_fits(ra=[25.,27.5],dec=[-27.5,-25.],n=1500000):
    """
    Helper function that just creates random positions within some ra,dec range.
    """

    ra1 = np.random.rand(n)*(ra[1]-ra[0])/180.*np.pi+ra[0]/180.*np.pi
    d0 = (np.cos((dec[0]+90)/180.*np.pi)+1)/2.
    d1 = (np.cos((dec[1]+90)/180.*np.pi)+1)/2.
    dec1 = np.arccos(2*np.random.rand(n)*(d1-d0)+2*d0-1)
    out = np.empty(n,dtype=[('ra',float)]+[('dec',float)])
    out['ra']=ra1*180./np.pi
    out['dec']=dec1*180./np.pi-90
    fio.write('ra_'+str(ra[0])+'_'+str(ra[1])+'_dec_'+str(dec[0])+'_'+str(dec[1])+'_n_'+str(n)+'.fits.gz',out,clobber=True)

def hsm(im, psf=None, wt=None):
    """
    Not used currently, but this is a helper function to run hsm via galsim.
    """

    out = np.zeros(1,dtype=[('e1','f4')]+[('e2','f4')]+[('T','f4')]+[('dx','f4')]+[('dy','f4')]+[('flag','i2')])
    try:
        if psf is not None:
            shape_data = galsim.hsm.EstimateShear(im, psf, weight=wt, strict=False)
        else:
            shape_data = im.FindAdaptiveMom(weight=wt, strict=False)
    except:
        # print(' *** Bad measurement (caught exception).  Mask this one.')
        out['flag'] |= BAD_MEASUREMENT
        return out

    if shape_data.moments_status != 0:
        # print('status = ',shape_data.moments_status)
        # print(' *** Bad measurement.  Mask this one.')
        out['flag'] |= BAD_MEASUREMENT

    out['dx'] = shape_data.moments_centroid.x - im.true_center().x
    out['dy'] = shape_data.moments_centroid.y - im.true_center().y
    if out['dx']**2 + out['dy']**2 > MAX_CENTROID_SHIFT**2:
        # print(' *** Centroid shifted by ',out['dx'],out['dy'],'.  Mask this one.')
        out['flag'] |= CENTROID_SHIFT

    # Account for the image wcs
    if im.wcs.isPixelScale():
        out['e1'] = shape_data.observed_shape.g1
        out['e2'] = shape_data.observed_shape.g2
        out['T']  = 2 * shape_data.moments_sigma**2 * im.scale**2
    else:
        e1 = shape_data.observed_shape.e1
        e2 = shape_data.observed_shape.e2
        s = shape_data.moments_sigma

        jac = im.wcs.jacobian(im.trueCenter())
        M = np.matrix( [[ 1 + e1, e2 ], [ e2, 1 - e1 ]] ) * s*s
        J = jac.getMatrix()
        M = J * M * J.T
        scale = np.sqrt(M/2./s/s)

        e1 = (M[0,0] - M[1,1]) / (M[0,0] + M[1,1])
        e2 = (2.*M[0,1]) / (M[0,0] + M[1,1])
        out['T'] = M[0,0] + M[1,1]

        shear = galsim.Shear(e1=e1, e2=e2)
        out['e1'] = shear.g1
        out['e2'] = shear.g2

    return out

def get_filename( out_path, path, name, var=None, name2=None, ftype='fits', overwrite=False, make=True ):
    """
    Helper function to set up a file path, and create the path if it doesn't exist.
    """
    if var is not None:
        name += '_' + var
    if name2 is not None:
        name += '_' + name2
    name += '.' + ftype
    fpath = os.path.join(out_path,path)
    if make:
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        if not os.path.exists(fpath):
            os.mkdir(fpath)
    filename = os.path.join(fpath,name)
    if (overwrite)&(os.path.exists(filename)):
        os.remove(filename)
    return filename

def get_filenames( out_path, path, name, var=None, name2=None, ftype='fits', exclude = None ):
    """
    Helper function to set up a file path, and create the path if it doesn't exist.
    """
    if var is not None:
        name += '_' + var
    if name2 is not None:
        name += '_' + name2
    name += '*.' + ftype
    fpath = os.path.join(out_path,path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    filename = os.path.join(fpath,name)
    if exclude is None:
        return glob.glob(filename)
    else:
        return [ x for x in glob.glob(filename) if exclude not in x ]

def prep_new_header(self,old_header):
    new_header = old_header
    new_header.set('Detector','IR')
    new_header.set('PROPOSID','HLS_SIT')
    new_header.set('LINENUM','None')
    new_header.set('TARGNAME','HLS')
    new_header.set('EXPTIME', 140.25)
    new_header.set('ROOTNAME', 'dc2_test')
    new_header.set('INSTRUME', 'WFC3')
    new_header.set('NGOODPIX', 4088*4088-1)
    new_header.set('EXTVER',1)
    new_header.set('EXPNAME', 'Generated Image')
    new_header.set('MEANDARK', 0.015) 
    new_header.set('PA_V3', old_header['PA_FPA'])
    new_header.set('GAIN', 2.5)
    new_header.set('CCDAMP', 'ABCD')
    new_header.set('CCDGAIN', 1.0)
    new_header.set('CCDOFSAB', 190)
    new_header.set('CCDOFSCD', 190)
    new_header.set('ATODGNA', 2.34)
    new_header.set('ATODGNB', 2.37)
    new_header.set('ATODGNC', 2.33)
    new_header.set('ATODGND', 2.36)
    new_header.set('READNSEA', 20.2)
    new_header.set('READNSEB', 19.7)
    new_header.set('READNSEC', 20.1)
    new_header.set('READNSED', 20.6)
    new_header.set('RDNOISE', 8.5)
    new_header.set('IDCSCALE', .1)
    new_header.set('BUNIT', 'ELECTRONS/S')
    new_header.set('PFLTFILE', 'iref$uc721145i_pfl.fits')
    new_header.set('TIME',1.0)
    return new_header

def write_fits(filename,img,err,dq,sca,root_name):
    from astropy.io import fits

    hdr = fits.Header()
    img.wcs.writeToFitsHeader(hdr,img.bounds)
    hdr['GS_XMIN']  = hdr['GS_XMIN']
    hdr['GS_XMIN']  = hdr['GS_YMIN']
    hdr['GS_WCS']   = hdr['GS_WCS']
    hdr['Detector'] = 'IR'
    hdr['PROPOSID'] = 'HLS_SIT'
    hdr['LINENUM']  = 'None'
    hdr['TARGNAME'] = 'HLS'
    hdr['EXPTIME']  = 140.25
    hdr['ROOTNAME'] = root_name
    hdr['INSTRUME'] = 'WFC3'
    hdr['NGOODPIX'] = 4088*4088-1
    hdr['EXTVER']   = 1
    hdr['EXPNAME']  = 'GalSim Image'
    hdr['MEANDARK'] = 0.015
    hdr['PA_V3']    = hdr['PA_FPA']
    hdr['GAIN']     = 2.5
    hdr['CCDAMP']   = 'ABCD'
    hdr['CCDGAIN']  = 1.0
    hdr['CCDOFSAB'] = 190
    hdr['CCDOFSCD'] = 190
    hdr['ATODGNA']  = 2.34
    hdr['ATODGNB']  = 2.37
    hdr['ATODGNC']  = 2.33
    hdr['ATODGND']  = 2.36
    hdr['READNSEA'] = 20.2
    hdr['READNSEB'] = 19.7
    hdr['READNSEC'] = 20.1
    hdr['READNSED'] = 20.6
    hdr['RDNOISE']  = 8.5
    hdr['IDCSCALE'] = .1
    hdr['BUNIT']    = 'ELECTRONS/S'
    hdr['PFLTFILE'] = 'iref$uc721145i_pfl.fits'
    hdr['TIME']     = 1.0
    fit0 = fits.PrimaryHDU(header=hdr)
    fit1 = fits.ImageHDU(data=img.array,header=hdr, name='SCI', ver=1)
    if err is not None:
        fit2 = fits.ImageHDU(data=err.array,header=hdr, name='ERR', ver=1)
    if dq is not None:
        fit3 = fits.ImageHDU(data=dq,header=hdr, name='DQ', ver=1)
    if dq is not None:
        new_fits_file = fits.HDUList([fit0,fit1,fit2,fit3])
    elif err is not None:
        new_fits_file = fits.HDUList([fit0,fit1,fits2])
    else:
        new_fits_file = fits.HDUList([fit0,fit1])
    new_fits_file.writeto(filename,overwrite=True)

    return
