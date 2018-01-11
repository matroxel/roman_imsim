"""
An implementation of galaxy and star image simulations for WFIRST. 
Built from the WFIRST GalSim module. An example 
config file is provided as example.yaml.

Built from galsim demo13...
# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#
"""

import numpy as np
import healpy as hp
import sys, os
import math
import logging
import time
import yaml
import galsim as galsim
import galsim.wfirst as wfirst
import galsim.config.process as process
import galsim.des as des
import fitsio as fio
import cPickle as pickle
from astropy.time import Time
import mpi4py.MPI
import cProfile, pstats

path, filename = os.path.split(__file__)
sedpath = os.path.join(galsim.meta_data.share_dir, 'SEDs', 'CWW_Sbc_ext.sed')
sedpath_Star = os.path.join(galsim.meta_data.share_dir, 'SEDs', 'vega.txt')
g_band = os.path.join(galsim.meta_data.share_dir, 'bandpasses', 'LSST_g.dat')
if sys.version_info[0] == 3:
    string_types = str,
else:
    string_types = basestring,

t0=time.time()

MAX_RAD_FROM_BORESIGHT = 0.009
MAX_CENTROID_SHIFT = 999.
BAD_MEASUREMENT = 2
CENTROID_SHIFT = 1
BOX_SIZES = [32,48,64,96,128,192,256]
# flags for unavailable data
EMPTY_START_INDEX = 9999
EMPTY_JAC_diag    = 1
EMPTY_JAC_offdiag = 0
EMPTY_SHIFT = 0

big_fft_params = galsim.GSParams(maximum_fft_size=10240)

# Dict to convert GalSim WFIRST filter names to filter names for fluxes in:
# https://github.com/WFIRST-HLS-Cosmology/Docs/wiki/Home-Wiki#wfirstlsst-simulated-photometry-catalog-based-on-candels
filter_flux_dict = {
    'J129' : 'j_WFIRST',
    'F184' : 'F184W_WFIRST',
    'Y106' : 'y_WFIRST',
    'H158' : 'h_WFIRST'
}

# Converts galsim WFIRST filter names to indices in Chris' dither file.
filter_dither_dict = {
    'J129' : 3,
    'F184' : 1,
    'Y106' : 4,
    'H158' : 2
}

class ParamError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

def except_func(logger, proc, k, res, t):
    print proc, k
    print t
    raise res

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

def convert_dither_to_fits(ditherfile='observing_sequence_hlsonly'):

    dither = np.genfromtxt(ditherfile+'.dat',dtype=None,names = ['date','f1','f2','ra','dec','pa','program','filter','f8','f9','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21'])
    dither = dither[['date','ra','dec','pa','filter']][dither['program']==5]
    fio.write(ditherfile+'.fits',dither,clobber=True)

    return

def convert_gaia_to_fits(gaiacsv='../2017-09-14-19-58-07-4430',ralims=[0,360],declims=[-90,90]):

    # Need to replace with true gaia g bandpass
    g_band     = os.path.join(galsim.meta_data.share_dir, 'bandpasses', 'LSST_g.dat')
    g_band     = galsim.Bandpass(g_band, wave_type='nm').withZeropoint('AB')
    star_sed   = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')

    gaia = np.genfromtxt(gaiacsv+'.csv',dtype=None,delimiter=',',names = ['id','flux','ra','dec'],skip_header=1)
    gaia = gaia[(gaia['ra']>ralims[0])&(gaia['ra']<ralims[1])]
    gaia = gaia[(gaia['dec']>declims[0])&(gaia['dec']<declims[1])]
    out  = np.zeros(len(gaia),dtype=[('id','i4')]+[('J129','f4')]+[('F184','f4')]+[('Y106','f4')]+[('H158','f4')]+[('ra',float)]+[('dec',float)])
    out['id']  = gaia['id']
    out['ra']  = gaia['ra']
    out['dec'] = gaia['dec']
    for filter_ in ['J129','F184','Y106','H158']:
        print filter_
        bpass = wfirst.getBandpasses(AB_zeropoint=True)[filter_]
        for ind in range(len(gaia)):
            if ind%1000==0:
                print ind
            star_sed_         = star_sed.withFlux(gaia['flux'][ind],g_band)
            out[filter_][ind] = star_sed_.calculateFlux(bpass)

    fio.write('gaia_stars.fits',out,clobber=True)

    return

def create_radec_fits(ra=[25.,27.5],dec=[-27.5,-25.],n=1500000):

    ra1 = np.random.rand(n)*(ra[1]-ra[0])/180.*np.pi+ra[0]/180.*np.pi
    d0 = (np.cos((dec[0]+90)/180.*np.pi)+1)/2.
    d1 = (np.cos((dec[1]+90)/180.*np.pi)+1)/2.
    dec1 = np.arccos(2*np.random.rand(n)*(d1-d0)+2*d0-1)
    out = np.empty(n,dtype=[('ra',float)]+[('dec',float)])
    out['ra']=ra1*180./np.pi
    out['dec']=dec1*180./np.pi-90
    fio.write('ra_'+str(ra[0])+'_'+str(ra[1])+'_dec_'+str(dec[0])+'_'+str(dec[1])+'_n_'+str(n)+'.fits.gz',out,clobber=True)


def radec_to_chip(obsRA, obsDec, obsPA, ptRA, ptDec):
    """
    Converted from Chris' c code. Used here to limit ra, dec catalog to objects that fall in each pointing.
    """

    AFTA_SCA_Coords = np.array([
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

    sort  = np.argsort(ptDec)
    ptRA  = ptRA[sort]
    ptDec = ptDec[sort]
    # Crude cut of some objects more than some encircling radius away from the boresight - creates a fast dec slice. Probably not worth doing better than this.
    begin = np.searchsorted(ptDec, obsDec-MAX_RAD_FROM_BORESIGHT)
    end   = np.searchsorted(ptDec, obsDec+MAX_RAD_FROM_BORESIGHT)

    # Position of the object in boresight coordinates
    mX  = -np.sin(obsDec)*np.cos(ptDec[begin:end])*np.cos(obsRA-ptRA[begin:end]) + np.cos(obsDec)*np.sin(ptDec[begin:end])
    mY  = np.cos(ptDec[begin:end])*np.sin(obsRA-ptRA[begin:end])

    xi  = -(np.sin(obsPA)*mX + np.cos(obsPA)*mY) / 0.0021801102 # Image plane position in chips
    yi  =  (np.cos(obsPA)*mX - np.sin(obsPA)*mY) / 0.0021801102
    SCA = np.zeros(end-begin)
    for i in range(18):
        cptr = AFTA_SCA_Coords
        mask = (cptr[0+12*i]*xi+cptr[1+12*i]*yi<cptr[2+12*i]) \
                & (cptr[3+12*i]*xi+cptr[4+12*i]*yi<cptr[5+12*i]) \
                & (cptr[6+12*i]*xi+cptr[7+12*i]*yi<cptr[8+12*i]) \
                & (cptr[9+12*i]*xi+cptr[10+12*i]*yi<cptr[11+12*i])
        SCA[mask] = i+1

    return np.pad(SCA,(begin,len(ptDec)-end),'constant',constant_values=(0, 0))[np.argsort(sort)] # Pad SCA array with zeros and resort to original indexing

def hsm(im, psf=None, wt=None):

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

    out['dx'] = shape_data.moments_centroid.x - im.trueCenter().x
    out['dy'] = shape_data.moments_centroid.y - im.trueCenter().y
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

def EmptyMEDS(objs, exps, stampsize, psfstampsize, store, filename, images=0, clobber=True):
    """
    Based on galsim.des.des_meds.WriteMEDS().
    """

    from galsim._pyfits import pyfits

    MAX_NCUTOUTS = np.max(exps)+1
    cum_exps = np.cumsum(exps+1)

    # get number of objects
    n_obj = len(objs)

    # get the primary HDU
    primary = pyfits.PrimaryHDU()

    # second hdu is the object_data
    # cf. https://github.com/esheldon/meds/wiki/MEDS-Format
    cols = []
    tmp  = [[0]*MAX_NCUTOUTS]*n_obj
    cols.append( pyfits.Column(name='id',             format='K', array=np.arange(n_obj)            ) )
    cols.append( pyfits.Column(name='number',         format='K', array=objs                        ) )
    cols.append( pyfits.Column(name='ra',             format='D', array=store['ra'][objs]           ) )
    cols.append( pyfits.Column(name='dec',            format='D', array=store['dec'][objs]          ) )
    cols.append( pyfits.Column(name='ncutout',        format='K', array=exps[objs]+1                ) )
    cols.append( pyfits.Column(name='box_size',       format='K', array=np.ones(n_obj)*stampsize    ) )
    cols.append( pyfits.Column(name='psf_box_size',   format='K', array=np.ones(n_obj)*psfstampsize ) )
    cols.append( pyfits.Column(name='file_id',        format='%dK' % MAX_NCUTOUTS, array=[1]*n_obj  ) )
    cols.append( pyfits.Column(name='start_row',      format='%dK' % MAX_NCUTOUTS, array=tmp        ) )
    cols.append( pyfits.Column(name='orig_row',       format='%dD' % MAX_NCUTOUTS, array=tmp        ) )
    cols.append( pyfits.Column(name='orig_col',       format='%dD' % MAX_NCUTOUTS, array=tmp        ) )
    cols.append( pyfits.Column(name='orig_start_row', format='%dK' % MAX_NCUTOUTS, array=tmp        ) )
    cols.append( pyfits.Column(name='orig_start_col', format='%dK' % MAX_NCUTOUTS, array=tmp        ) )
    cols.append( pyfits.Column(name='cutout_row',     format='%dD' % MAX_NCUTOUTS, array=tmp        ) )
    cols.append( pyfits.Column(name='cutout_col',     format='%dD' % MAX_NCUTOUTS, array=tmp        ) )
    cols.append( pyfits.Column(name='dudrow',         format='%dD' % MAX_NCUTOUTS, array=tmp        ) )
    cols.append( pyfits.Column(name='dudcol',         format='%dD' % MAX_NCUTOUTS, array=tmp        ) )
    cols.append( pyfits.Column(name='dvdrow',         format='%dD' % MAX_NCUTOUTS, array=tmp        ) )
    cols.append( pyfits.Column(name='dvdcol',         format='%dD' % MAX_NCUTOUTS, array=tmp        ) )
    cols.append( pyfits.Column(name='psf_start_row',  format='%dK' % MAX_NCUTOUTS, array=tmp        ) )
    cols.append( pyfits.Column(name='dither',         format='%dK' % MAX_NCUTOUTS, array=tmp        ) )
    cols.append( pyfits.Column(name='sca',            format='%dK' % MAX_NCUTOUTS, array=tmp        ) )

    # Depending on the version of pyfits, one of these should work:
    try:
        object_data = pyfits.BinTableHDU.from_columns(cols)
        object_data.name = 'object_data'
    except AttributeError:  # pragma: no cover
        object_data = pyfits.new_table(pyfits.ColDefs(cols))
        object_data.update_ext_name('object_data')

    # third hdu is image_info
    cols = []
    gstring = 'generated_by_galsim'
    cols.append( pyfits.Column(name='image_path',  format='A256',   array=np.repeat(gstring,images) ) )
    cols.append( pyfits.Column(name='image_ext',   format='I',      array=np.zeros(images)          ) )
    cols.append( pyfits.Column(name='weight_path', format='A256',   array=np.repeat(gstring,images) ) )
    cols.append( pyfits.Column(name='weight_ext',  format='I',      array=np.zeros(images)          ) )
    cols.append( pyfits.Column(name='seg_path',    format='A256',   array=np.repeat(gstring,images) ) )
    cols.append( pyfits.Column(name='seg_ext',     format='I',      array=np.zeros(images)          ) )
    cols.append( pyfits.Column(name='bmask_path',  format='A256',   array=np.repeat(gstring,images) ) )
    cols.append( pyfits.Column(name='bmask_ext',   format='I',      array=np.zeros(images)          ) )
    cols.append( pyfits.Column(name='bkg_path',    format='A256',   array=np.repeat(gstring,images) ) )
    cols.append( pyfits.Column(name='bkg_ext',     format='I',      array=np.zeros(images)          ) )
    cols.append( pyfits.Column(name='image_id',    format='K',      array=np.ones(images)*-1        ) )
    cols.append( pyfits.Column(name='image_flags', format='K',      array=np.zeros(images)          ) )
    cols.append( pyfits.Column(name='magzp',       format='E',      array=np.ones(images)*30        ) )
    cols.append( pyfits.Column(name='scale',       format='E',      array=np.zeros(images)          ) )
    # TODO: Not sure if this is right!
    cols.append( pyfits.Column(name='position_offset', format='D',  array=np.zeros(images)          ) )
    try:
        image_info = pyfits.BinTableHDU.from_columns(cols)
        image_info.name = 'image_info'
    except AttributeError:  # pragma: no cover
        image_info = pyfits.new_table(pyfits.ColDefs(cols))
        image_info.update_ext_name('image_info')

    # fourth hdu is metadata
    # default values?
    cols = []
    cols.append( pyfits.Column(name='magzp_ref',     format='E',    array=[30.]                   ))
    cols.append( pyfits.Column(name='DESDATA',       format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='cat_file',      format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='coadd_image_id',format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='coadd_file',    format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='coadd_hdu',     format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='coadd_seg_hdu', format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='coadd_srclist', format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='coadd_wt_hdu',  format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='coaddcat_file', format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='coaddseg_file', format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='cutout_file',   format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='max_boxsize',   format='A3',   array=['-1']                  ))
    cols.append( pyfits.Column(name='medsconf',      format='A3',   array=['x']                   ))
    cols.append( pyfits.Column(name='min_boxsize',   format='A2',   array=['-1']                  ))
    cols.append( pyfits.Column(name='se_badpix_hdu', format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='se_hdu',        format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='se_wt_hdu',     format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='seg_hdu',       format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='psf_hdu',       format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='sky_hdu',       format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='fake_coadd_seg',format='K',    array=[9999]                  ))
    try:
        metadata = pyfits.BinTableHDU.from_columns(cols)
        metadata.name = 'metadata'
    except AttributeError:  # pragma: no cover
        metadata = pyfits.new_table(pyfits.ColDefs(cols))
        metadata.update_ext_name('metadata')

    # rest of HDUs are image vectors
    print 'exps',np.sum(exps[objs]+1)
    image_cutouts   = pyfits.ImageHDU( np.zeros(np.sum(exps[objs]+1)*stampsize*stampsize) , name='image_cutouts'  )
    weight_cutouts  = pyfits.ImageHDU( np.zeros(np.sum(exps[objs]+1)*stampsize*stampsize) , name='weight_cutouts' )
    seg_cutouts     = pyfits.ImageHDU( np.zeros(np.sum(exps[objs]+1)*stampsize*stampsize) , name='seg_cutouts'    )
    psf_cutouts     = pyfits.ImageHDU( np.zeros(np.sum(exps[objs]+1)*psfstampsize*psfstampsize) , name='psf'      )

    # write all
    hdu_list = pyfits.HDUList([
        primary,
        object_data,
        image_info,
        metadata,
        image_cutouts,
        weight_cutouts,
        seg_cutouts,
        psf_cutouts
    ])
    galsim.fits.writeFile(filename, hdu_list)

    return

class wfirst_sim(object):
    """
    WFIRST image simulation.

    Input:
    param_file : File path for input yaml config file or yaml dict. Example located at: ./example.yaml.
    """

    def __init__(self, param_file, use_mpi = None):

        # Load parameter file
        if isinstance(param_file, string_types):
            self.params     = yaml.load(open(param_file))
            self.param_file = param_file
            # Do some parsing
            for key in self.params.keys():
                if self.params[key]=='None':
                    self.params[key]=None
                if self.params[key]=='True':
                    self.params[key]=True
                if self.params[key]=='False':
                    self.params[key]=False
        else:
            self.params     = param_file

        if use_mpi is not None:
            self.params['use_mpi'] = use_mpi

        # GalSim logger
        logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
        # In non-script code, use getself.logger(__name__) at module scope instead.
        self.logger = logging.getLogger('wfirst_sim')

        # Initialize (pseudo-)random number generator.
        self.reset_rng()

        # Where to find and output data.
        path, filename = os.path.split(__file__)
        self.out_path = os.path.abspath(os.path.join(path, self.params['out_path']))

        # Make output directory if not already present.
        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)

        # Set total number of unique objects
        if isinstance(self.params['gal_dist'],string_types):
            self.n_gal = fio.FITS(self.params['gal_dist'])[-1].read_header()['NAXIS2']
        else:
            raise ParamError('Currently need gal_dist file.')

        # Read in the WFIRST filters, setting an AB zeropoint appropriate for this telescope given its
        # diameter and (since we didn't use any keyword arguments to modify this) using the typical
        # exposure time for WFIRST images.  By default, this routine truncates the parts of the
        # bandpasses that are near 0 at the edges, and thins them by the default amount.
        self.bpass      = wfirst.getBandpasses(AB_zeropoint=True)[self.params['filter']]
        # Setup galaxy SED
        # Need to generalize to vary sed based on input catalog
        self.galaxy_sed = galsim.SED(sedpath, wave_type='Ang', flux_type='flambda')
        # Setup star SED
        self.star_sed   = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')

        return

    def reset_rng(self):
        """
        Reset the (pseudo-)random number generators.
        """

        self.rng     = galsim.BaseDeviate(self.params['random_seed'])
        self.gal_rng = galsim.UniformDeviate(self.params['random_seed'])

        return

    def fwhm_to_hlr(self,fwhm):

        radius = fwhm*0.06/2. # 1 pix = 0.06 arcsec, factor 2 to convert to hlr

        return radius

    def meds_filename(self,chunk):

        return self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_'+str(chunk)+'.fits'

    def get_totpix(self):

        return np.unique(hp.ang2pix(self.params['nside'], np.pi/2.-self.store['dec'],self.store['ra'], nest=True))

    def get_npix(self,pix):

        return np.sum(pix==hp.ang2pix(self.params['nside'], np.pi/2.-self.store['dec'],self.store['ra'], nest=True))

    def get_pix_gals(self,pix):

        return np.where(pix==hp.ang2pix(self.params['nside'], np.pi/2.-self.store['dec'],self.store['ra'], nest=True))[0]

    def compile_tab(self,results=None,max_exp=25):

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_table.fits'
        if results is None:

            try:
                self.table = fio.FITS(filename)[-1].read()
            except:
                return False

        else:

            start = 0
            self.table = np.ones(len(self.store)*max_exp,dtype=[('sca',int)]+[('dither',int)]+[('gal',int)])
            for name in self.table.dtype.names:
                self.table[name] *=-1
            for result_ in results:
                if type(result_) is np.ndarray:

                    self.table[start:start+len(result_)] = result_
                    start += len(result_)

                else:

                    for result in result_:
                        self.table[start:start+len(result)] = result
                        start += len(result)

            self.table = self.table[:start]
            self.table = self.table[np.argsort(self.table,order=('gal','dither','sca'))]
            fio.write(filename,self.table,clobber=True)

        if self.params['remake_meds']:

            for pix in self.get_totpix():
                try:
                    fits=fio.FITS(self.meds_filename(pix))
                    fits.close()
                    if self.params['clobber']:
                        os.remove(self.meds_filename(pix))
                    else:
                        return True
                except:
                    pass

                print pix,len(self.get_pix_gals(pix))

                # low = chunk*self.params['meds_size']
                # high = (chunk+1)*self.params['meds_size']
                # if high>self.n_gal:
                #     high=self.n_gal
                gals = self.get_pix_gals(pix)
                pix_mask = np.in1d(self.table['gal'],gals,assume_unique=False)
                exps = np.bincount(self.table['gal'][pix_mask])
                EmptyMEDS(gals,exps,self.params['stamp_size'],64,self.store,self.meds_filename(pix))

                # extend pixel arrays
                # fits=fio.FITS(self.meds_filename(chunk),'rw')
                # for hdu in ['image_cutouts','weight_cutouts','seg_cutouts','psf']:
                #     if hdu == 'psf':
                #         fits[hdu].write(np.zeros(1),start=[np.sum(exps[low:high]+1)*64*64])
                #     else:
                #         fits[hdu].write(np.zeros(1),start=[np.sum(exps[low:high]+1)*self.params['stamp_size']*self.params['stamp_size']])
                # fits.close()

        return True

    def tabulate_exposures_loop(self,node,nodes,max_exp=25):

        tasks = []
        for i in range(self.params['nproc']):
            tasks.append({
                'node'       : node,
                'nodes'      : nodes,
                'proc'       : i,
                'params'     : self.params,
                'store'      : self.store,
                'stars'      : self.stars})

        tasks = [ [(job, k)] for k, job in enumerate(tasks) ]

        results = process.MultiProcess(self.params['nproc'], {}, tabulate_exposures, tasks, 'tabulate_exposures', logger=self.logger, done_func=None, except_func=except_func, except_abort=True)

        return results

    def init_galaxy(self):
        """
        Does the work to return a random, unique object property list. 
        """

        if isinstance(self.params['gal_dist'],string_types):
            # Provided an ra,dec catalog of object positions.
            radec_file     = fio.FITS(self.params['gal_dist'])[-1]
        else:
            raise ParamError('Bad gal_dist filename.')

        if self.params['gal_type'] == 0:
            # Analytic profile - sersic disk

            # Check if output truth file path exists or if explicitly remaking galaxy properties 
            filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_truth_gal.fits'
            if self.params['rerun_models']:

                # Read in file with photometry/size/redshift distribution similar to WFIRST galaxies
                phot       = fio.FITS(self.params['gal_sample'])[-1].read(columns=['fwhm','redshift',filter_flux_dict[self.params['filter']]])
                pind_list_ = np.ones(len(phot)).astype(bool) # storage list for original index of photometry catalog
                pind_list_ = pind_list_&(phot[filter_flux_dict[self.params['filter']]]<99)&(phot[filter_flux_dict[self.params['filter']]]>0) # remove bad mags
                pind_list_ = pind_list_&(phot['redshift']>0)&(phot['redshift']<5) # remove bad redshifts
                pind_list_ = pind_list_&(phot['fwhm']*0.06/wfirst.pixel_scale<16) # remove large objects to maintain 32x32 stamps
                pind_list_ = np.where(pind_list_)[0]

                # Create minimal storage array for galaxy properties to pass to parallel tasks
                store = np.ones(self.n_gal, dtype=[('rot','i2')]+[('e','i2')]+[('size','f4')]+[('z','f4')]+[('mag','f4')]+[('ra',float)]+[('dec',float)])
                store['ra'] = radec_file.read(columns='ra')*np.pi/180.
                store['dec'] = radec_file.read(columns='dec')*np.pi/180.
                pind = np.zeros(len(store)).astype(int)
                g1   = np.zeros(len(store))
                g2   = np.zeros(len(store))
                for i in range(self.n_gal):
                    pind[i] = pind_list_[int(self.gal_rng()*len(pind_list_))]
                    store['rot'][i]  = int(self.gal_rng()*360.)
                    store['e'][i]    = int(self.gal_rng()*len(self.params['shear_list']))
                    g1[i] = self.params['shear_list'][store['e'][i]][0]
                    g2[i] = self.params['shear_list'][store['e'][i]][1]
                    store['size'][i] = self.fwhm_to_hlr(phot['fwhm'][pind[i]])
                    store['z'][i]    = phot['redshift'][pind[i]]
                    store['mag'][i]  = phot[filter_flux_dict[self.params['filter']]][pind[i]]

                # Save truth file with galaxy properties
                self.dump_truth_gal(store,pind,g1,g2)

            else:

                # Load truth file with galaxy properties
                store = self.load_truth_gal()

        else:
            raise ParamError('COSMOS profiles not currently implemented.')            
            pass # cosmos gal not guaranteed to work. uncomment at own risk 
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

        return store

    def init_star(self):
        """
        Compiles a list of stars properties to draw. 
        """

        if isinstance(self.params['star_sample'],string_types):
            # Provided a catalog of star positions and properties.
            fits = fio.FITS(self.params['star_sample'])[-1]
        else:
            raise ParamError('Bad star_sample filename.')

        stars_ = fits.read(columns=['ra','dec',self.params['filter']])

        # Create minimal storage array for galaxy properties to pass to parallel tasks
        stars         = np.ones(len(stars_), dtype=[('flux','f4')]+[('ra',float)]+[('dec',float)])
        stars['ra']   = stars_['ra']*np.pi/180.
        stars['dec']  = stars_['dec']*np.pi/180.
        stars['flux'] = stars_[self.params['filter']]

        return stars

    def init_noise_model(self):
        """
        Generate a poisson noise model.
        """

        self.noise = galsim.PoissonNoise(self.rng)

        return 

    def add_effects(self,im):
        """
        Add detector effects for WFIRST.

        Input:

        im      : Postage stamp or image.

        Preserve order:
        1) add_background
        2) add_poisson_noise
        3) recip_failure 
        4) quantize
        5) dark_current
        6) nonlinearity
        7) interpix_cap
        8) e_to_ADU
        9) quantize


        Where does persistence get added? Immediately before/after background?
        """

        if self.params['use_background']:
            im, sky_image = self.add_background(im) # Add background to image and save background

        if self.params['use_poisson_noise']:
            im = self.add_poisson_noise(im) # Add poisson noise to image

        if self.params['use_recip_failure']:
            im = self.recip_failure(im) # Introduce reciprocity failure to image

        im.quantize() # At this point in the image generation process, an integer number of photons gets detected

        if self.params['use_dark_current']:
            im = self.dark_current(im) # Add dark current to image

        if self.params['use_nonlinearity']:
            im = self.nonlinearity(im) # Apply nonlinearity

        if self.params['use_interpix_cap']:
            im = self.interpix_cap(im) # Introduce interpixel capacitance to image.

        im = self.e_to_ADU(im) # Convert electrons to ADU

        im.quantize() # Finally, the analog-to-digital converter reads in an integer value.

        # Note that the image type after this step is still a float. If we want to actually
        # get integer values, we can do new_img = galsim.Image(im, dtype=int)
        # Since many people are used to viewing background-subtracted images, we return a
        # version with the background subtracted (also rounding that to an int).
        if self.params['use_background']:
            im,sky_image = self.finalize_background_subtract(im,sky_image)

        # im = galsim.Image(im, dtype=int)

        # get weight map
        sky_image.invertSelf()

        return im, sky_image

    def add_background(self,im):
        """
        Add backgrounds to image (sky, thermal).

        First we get the amount of zodaical light for a position corresponding to the position of 
        the object. The results are provided in units of e-/arcsec^2, using the default WFIRST
        exposure time since we did not explicitly specify one. Then we multiply this by a factor
        >1 to account for the amount of stray light that is expected. If we do not provide a date
        for the observation, then it will assume that it's the vernal equinox (sun at (0,0) in
        ecliptic coordinates) in 2025.
        """

        sky_level = wfirst.getSkyLevel(self.bpass, world_pos=self.radec, date=self.date)
        sky_level *= (1.0 + wfirst.stray_light_fraction)
        # Make a image of the sky that takes into account the spatially variable pixel scale. Note
        # that makeSkyImage() takes a bit of time. If you do not care about the variable pixel
        # scale, you could simply compute an approximate sky level in e-/pix by multiplying
        # sky_level by wfirst.pixel_scale**2, and add that to final_image.

        sky_stamp = galsim.Image(bounds=im.bounds, wcs=self.local_wcs)
        self.local_wcs.makeSkyImage(sky_stamp, sky_level)

        # This image is in units of e-/pix. Finally we add the expected thermal backgrounds in this
        # band. These are provided in e-/pix/s, so we have to multiply by the exposure time.
        sky_stamp += wfirst.thermal_backgrounds[self.params['filter']]*wfirst.exptime

        # Adding sky level to the image.
        im += sky_stamp
        
        return im,sky_stamp

    def add_poisson_noise(self,im):
        """
        Add pre-initiated poisson noise to image.
        """

        # Check if noise initiated
        if not hasattr(self,'noise'):
            self.init_noise_model()

        im.addNoise(self.noise)

        return im

    def recip_failure(self,im):
        """
        Introduce reciprocity failure to image.

        Reciprocity, in the context of photography, is the inverse relationship between the
        incident flux (I) of a source object and the exposure time (t) required to produce a given
        response(p) in the detector, i.e., p = I*t. However, in NIR detectors, this relation does
        not hold always. The pixel response to a high flux is larger than its response to a low
        flux. This flux-dependent non-linearity is known as 'reciprocity failure', and the
        approximate amount of reciprocity failure for the WFIRST detectors is known, so we can
        include this detector effect in our images.
        """

        im.addReciprocityFailure(exp_time=wfirst.exptime, alpha=wfirst.reciprocity_alpha,
                              base_flux=1.0)
        # wfirst.addReciprocityFailure(im)

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
        """

        dark_current = wfirst.dark_current*wfirst.exptime
        dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(self.rng, dark_current))
        im.addNoise(dark_noise)

        # NOTE: Sky level and dark current might appear like a constant background that can be
        # simply subtracted. However, these contribute to the shot noise and matter for the
        # non-linear effects that follow. Hence, these must be included at this stage of the
        # image generation process. We subtract these backgrounds in the end.

        return im

    def nonlinearity(self,im):
        """
        Applying a quadratic non-linearity.

        Note that users who wish to apply some other nonlinearity function (perhaps for other NIR
        detectors, or for CCDs) can use the more general nonlinearity routine, which uses the
        following syntax:
        final_image.applyNonlinearity(NLfunc=NLfunc)
        with NLfunc being a callable function that specifies how the output image pixel values
        should relate to the input ones.
        """

        # Apply the WFIRST nonlinearity routine, which knows all about the nonlinearity expected in
        # the WFIRST detectors.
        im.applyNonlinearity(NLfunc=wfirst.NLfunc)
        # wfirst.applyNonlinearity(im)

        return im

    def interpix_cap(self,im):
        """
        Including Interpixel capacitance

        The voltage read at a given pixel location is influenced by the charges present in the
        neighboring pixel locations due to capacitive coupling of sense nodes. This interpixel
        capacitance effect is modeled as a linear effect that is described as a convolution of a
        3x3 kernel with the image. The WFIRST IPC routine knows about the kernel already, so the
        user does not have to supply it.
        """
        im.applyIPC(wfirst.ipc_kernel, edge_treatment='extend', fill_value=None)
        # wfirst.applyIPC(im)

        return im

    def add_read_noise(self,im):
        """
        Adding read noise

        Read noise is the noise due to the on-chip amplifier that converts the charge into an
        analog voltage.  We already applied the Poisson noise due to the sky level, so read noise
        should just be added as Gaussian noise:
        """

        read_noise = galsim.GaussianNoise(self.rng, sigma=wfirst.read_noise)
        im.addNoise(read_noise)

        return im

    def e_to_ADU(self,im):
        """
        We divide by the gain to convert from e- to ADU. Currently, the gain value in the WFIRST
        module is just set to 1, since we don't know what the exact gain will be, although it is
        expected to be approximately 1. Eventually, this may change when the camera is assembled,
        and there may be a different value for each SCA. For now, there is just a single number,
        which is equal to 1.
        """

        return im/wfirst.gain

    def finalize_sky_im(self,im):
        """
        Finalize sky background for subtraction from final image. Add dark current, 
        convert to analog voltage, and quantize.
        """

        if (self.params['sub_true_background'])&(self.params['use_dark_current']):
            im = (im + round(wfirst.dark_current*wfirst.exptime))
        im = self.e_to_ADU(im)
        im.quantize()

        return im

    def finalize_background_subtract(self,im,sky):
        """
        Finalize background subtraction of image.
        """

        sky.quantize() # Quantize sky
        sky = self.finalize_sky_im(sky) # Finalize sky with dark current, convert to ADU, and quantize.
        im -= sky

        return im,sky

    def galaxy(self, ind, sca, radec, bound = None, return_xy = False, return_sed = False):
        """
        Draw a postage stamp for one of the galaxy objects using the local wcs for its position in the SCA plane. Apply add_effects. 
        """

        out = []
        # Check if galaxy falls on SCA and continue if not
        xy = self.WCS[sca].toImage(radec)
        if bound is not None:
            if not bound.includes(galsim.PositionI(int(xy.x),int(xy.y))):
                out.append(None)
                if return_sed:
                    out.append(None)
                if return_xy:
                    out.append(xy)
                return out

        # Generate galaxy model
        gal          = galsim.Sersic(self.params['disk_n'], half_light_radius=1.*self.store['size'][ind]) # sersic disk galaxy
        gal          = gal.rotate(self.store['rot'][ind]*galsim.degrees) # random rotation
        gal          = gal.shear(g1=self.params['shear_list'][self.store['e'][ind]][0],g2=self.params['shear_list'][self.store['e'][ind]][1]) # apply a shear
        galaxy_sed   = self.galaxy_sed.atRedshift(self.store['z'][ind]) # redshift SED
        galaxy_sed   = galaxy_sed.withMagnitude(self.store['mag'][ind],self.bpass) # apply correct flux from magnitude
        gal          = gal * galaxy_sed * galsim.wfirst.collecting_area * galsim.wfirst.exptime

        # ignoring chromatic stuff for now
        flux = gal.calculateFlux(self.bpass) # store flux
        gal  = gal.evaluateAtWavelength(self.bpass.effective_wavelength) # make achromatic
        gal  = gal.withFlux(flux) # reapply correct flux
        
        if self.params['draw_sca']:
            gal  = galsim.Convolve(gal, self.PSF[sca], gsparams=big_fft_params) # Convolve with PSF and append to final image list
        else:
            gal  = galsim.Convolve(gal, self.PSF[sca]) # Convolve with PSF and append to final image list

        # replaced by above lines
        # # Draw galaxy igal into stamp.
        # self.gal_list[igal].drawImage(self.pointing.bpass[self.params['filter']], image=gal_stamp)
        # # Add detector effects to stamp.

        out.append(gal)
        if return_sed:
            out.append(galaxy_sed)
            if return_xy:
                out.append(xy)

        return out

    def star(self, sed, sca, flux = 1., radec = None, bound = None, return_xy = False):

        out = []
        # Check if star falls on SCA and continue if not
        if radec is not None:
            xy = self.WCS[sca].toImage(radec)
            if bound is not None:
                if not bound.includes(galsim.PositionI(int(xy.x),int(xy.y))):
                    out.append(None)
                if return_xy:
                    out.append(xy)
                return out

        # Generate star model
        star = galsim.DeltaFunction() * sed
        # Draw the star
        # new effective version for speed
        star = star.evaluateAtWavelength(self.bpass.effective_wavelength)
        star = star.withFlux(flux)
        if self.params['draw_sca']:
            star = galsim.Convolve(star, self.PSF[sca], gsparams=big_fft_params)
        else:
            star = galsim.Convolve(star, self.PSF[sca], galsim.Pixel(wfirst.pixel_scale))

        # old chromatic version
        # self.psf_list[igal].drawImage(self.pointing.bpass[self.params['filter']],image=psf_stamp, wcs=local_wcs)

        #galaxy_sed = galsim.SED(
        #    os.path.join(sedpath, 'CWW_Sbc_ext.sed'), wave_type='Ang', flux_type='flambda').withFlux(
        #    1.,self.pointing.bpass[self.params['filter']])
        #self.pointing.PSF[self.SCA[igal]] *= galaxy_sed
        #pointing_psf = galsim.Convolve(galaxy_sed, self.pointing.PSF[self.SCA[igal]])
        #self.pointing.PSF[self.SCA[igal]].drawImage(self.pointing.bpass[self.params['filter']],image=psf_stamp, wcs=local_wcs)
        #pointing_psf = galaxy_sed * self.pointing.PSF[self.SCA[igal]]
        #pointing_psf.drawImage(self.pointing.bpass[self.params['filter']],image=psf_stamp, wcs=local_wcs)
        #self.pointing.PSF[self.SCA[igal]].drawImage(self.pointing.bpass[self.params['filter']],image=psf_stamp, wcs=local_wcs)

        out.append(star)
        if return_xy:
            out.append(xy)

        return out

    def draw_galaxy(self, ind, sca, bound):

        self.radec = galsim.CelestialCoord(self.store['ra'][ind]*galsim.radians,self.store['dec'][ind]*galsim.radians)
        gal,sed,xy = self.galaxy(ind,sca,self.radec,bound=bound,return_xy=True,return_sed=True)

        # Get local wcs solution at galaxy position in SCA.
        self.local_wcs = self.WCS[sca].local(xy)
        # Create stamp at this position.
        gal_stamp = galsim.Image(self.params['stamp_size'], self.params['stamp_size'], wcs=self.local_wcs)
        gal.drawImage(image=gal_stamp) # draw galaxy stamp
        # Apply background, noise, and WFIRST detector effects
        gal_stamp, weight_stamp = self.add_effects(gal_stamp) # Get final galaxy stamp and weight map

        out = [gal_stamp, weight_stamp]
        if self.params['draw_true_psf']:
            psf       = self.star(sed, sca)[0]
            wcs = galsim.JacobianWCS(dudx=self.local_wcs.dudx/self.params['oversample'],
                                    dudy=self.local_wcs.dudy/self.params['oversample'],
                                    dvdx=self.local_wcs.dvdx/self.params['oversample'],
                                    dvdy=self.local_wcs.dvdy/self.params['oversample'])
            psf_stamp = galsim.Image(self.params['psf_stampsize']*self.params['oversample'], self.params['psf_stampsize']*self.params['oversample'], wcs=wcs)
            psf.drawImage(image=psf_stamp,wcs=wcs,method='no_pixel')

            out.append(psf_stamp)

        return out

    def draw_pure_stamps(self,sca,proc,dither,d_,d,cnt,dumps):

        if self.params['timing']:
            print 'after gal_use_ind',time.time()-t0

        out = self.draw_galaxy(ind,None)
        if self.params['timing']:
            if i%1000==0:
                print 'drawing galaxy ',i,time.time()-t0

        if ind in self.gal_exps.keys():
            self.gal_exps[ind].append(out[0])
            self.wcs_exps[ind].append(self.local_wcs)
            self.wgt_exps[ind].append(out[1])
            if self.params['draw_true_psf']:
                self.psf_exps[ind].append(out[2]) 
            self.dither_list[ind].append(d_[d])
        else:
            self.gal_exps[ind]     = [out[0]]
            self.wcs_exps[ind]     = [self.local_wcs]
            self.wgt_exps[ind]     = [out[1]]
            if self.params['draw_true_psf']:
                self.psf_exps[ind] = [out[2]] 
            self.dither_list[ind]  = [d_[d]]

        print '------------- dither done ',d_[d]

        if cnt>self.params['pickle_size']:
            dumps,cnt = self.dump_stamps_pickle(sca,proc,dumps,cnt)

        return cnt,dumps

    def draw_sca(self,sca,proc,dither,d_,d):

        # Find objects near pointing.
        gal_use_ind = self.near_pointing(dither['ra'][d], dither['dec'][d], dither['pa'][d], self.store['ra'], self.store['dec'])

        if self.params['draw_stars']:
            # Find stars near pointing.
            star_use_ind = self.near_pointing(dither['ra'][d], dither['dec'][d], dither['pa'][d], self.stars['ra'], self.stars['dec'])
        else:
            star_use_ind = []

        if len(gal_use_ind)+len(star_use_ind)==0: # If nothing in focal plane, skip dither
            return None, None
        if self.params['timing']:
            print 'after _use_ind',time.time()-t0

        # Setup image for SCA
        b0  = galsim.BoundsI(xmin=-int(self.params['stamp_size'])/2,
                            ymin=-int(self.params['stamp_size'])/2,
                            xmax=wfirst.n_pix+int(self.params['stamp_size'])/2,
                            ymax=wfirst.n_pix+int(self.params['stamp_size'])/2)
        im = galsim.ImageF(bounds=b0, wcs=self.WCS)

        cnt = 0
        for i,ind in enumerate(gal_use_ind):
            radec  = galsim.CelestialCoord(self.store['ra'][ind]*galsim.radians,self.store['dec'][ind]*galsim.radians)
            gal,xy = self.galaxy(ind,
                                radec,
                                bound=b0,
                                return_xy = True)
            if gal is None:
                continue
            if self.params['timing']:
                if i%1==0:
                    print 'drawing galaxy ',i,time.time()-t0
            b = galsim.BoundsI(xmin=int(xy.x)-int(self.params['stamp_size'])/2,
                                ymin=int(xy.y)-int(self.params['stamp_size'])/2,
                                xmax=int(xy.x)+int(self.params['stamp_size'])/2,
                                ymax=int(xy.y)+int(self.params['stamp_size'])/2)
            b = b & im.bounds
            gal.drawImage(image=im[b], add_to_image=True, offset=xy-im[b].trueCenter())
            cnt+=1
            if ind in self.dither_list[0].keys():
                self.dither_list[0][ind].append(d_[d])
                self.sca_list[0][ind].append(sca)
                self.xy_list[0][ind].append(xy)
            else:
                self.dither_list[0][ind]  = [d_[d]]
                self.sca_list[0][ind]     = [sca]
                self.xy_list[0][ind]      = [xy]

        if self.params['draw_stars']:
            star_sed = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')
            for i,ind in enumerate(star_use_ind):
                radec    = galsim.CelestialCoord(self.stars['ra'][ind]*galsim.radians,self.stars['dec'][ind]*galsim.radians)
                star,xy  = self.star(star_sed, 
                                    flux = self.stars['flux'][ind], 
                                    radec = radec, 
                                    bound = b0,
                                    return_xy = True)
                if star is None:
                    continue
                if self.params['timing']:
                    if i%1==0:
                        print 'drawing star ',i,time.time()-t0
                b = galsim.BoundsI(xmin=int(xy.x)-int(self.params['stamp_size'])/2,
                                    ymin=int(xy.y)-int(self.params['stamp_size'])/2,
                                    xmax=int(xy.x)+int(self.params['stamp_size'])/2,
                                    ymax=int(xy.y)+int(self.params['stamp_size'])/2)
                b = b & im.bounds
                star.drawImage(image=im[b], add_to_image=True, offset=xy-im[b].trueCenter())
                cnt+=1
                if ind in self.dither_list[1].keys():
                    self.dither_list[1][ind].append(d_[d])
                    self.sca_list[1][ind].append(sca)
                    self.xy_list[1][ind].append(xy)
                else:
                    self.dither_list[1][ind]  = [d_[d]]
                    self.sca_list[1][ind]     = [sca]
                    self.xy_list[1][ind]      = [xy]

        print 'done dither ',sca,d_[d],cnt
        if cnt==0:
            # sys.exit()
            return None, None

        im = im[galsim.BoundsI(xmin=im.bounds.xmin+int(self.params['stamp_size'])/2,
                                ymin=im.bounds.ymin+int(self.params['stamp_size'])/2,
                                xmax=im.bounds.xmax-int(self.params['stamp_size'])/2,
                                ymax=im.bounds.ymax-int(self.params['stamp_size'])/2)]

        im, wgt = self.add_effects(im) # Get final image and weight map

        return im,wgt

    def accumulate_sca(self):
        """
        Accumulate the written pickle files that contain the images for each SCA, with SCA and dither ids.
        Write images to fits file.
        """
        import glob

        d_ = self.setup_dither(proc=1,only_index = True)
        for d in d_:
            filenames = glob.glob(self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_image_*_'+str(d)+'.pickle')
            if len(filenames) == 0:
                continue
            print d,time.time()-t0
            im_list = []
            for sca in range(18):
                try:
                    filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_image_'+str(sca)+'_'+str(d)+'.pickle'
                    im = load_obj(filename)[0]
                except:
                    im = galsim.ImageF(wfirst.n_pix,wfirst.n_pix)
                im_list.append(im)

            self.dump_sca_fits(im_list,d)

        return

    def test_accumulate_stamps(self):
        """
        Prints out any missing sim output chunks.
        """

        max_dumps = 0
        for sca in range(18):
            for proc in range(20):
                for dumps in range(10):
                    filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_stamps_'+str(sca)+'_'+str(proc)+'_'+str(dumps)+'.pickle'
                    if os.path.exists(filename) and dumps>max_dumps:
                        max_dumps = dumps

        safe = True
        for sca in range(18):
            for proc in range(20):
                for dumps in range(10):
                    filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_stamps_'+str(sca)+'_'+str(proc)+'_'+str(dumps)+'.pickle'
                    if not os.path.exists(filename):
                        if dumps <= max_dumps:
                            print 'not found', sca,proc,dumps
                        if dumps == 0:
                            safe = False

        return safe


    def accumulate_stamps(self, ignore_missing_files = False):
        """
        Accumulate the written pickle files that contain the postage stamps for all objects, with SCA and dither ids.
        Write stamps to MEDS file, and SCA and dither ids to truth files. 
        """

        if (not ignore_missing_files) and (not self.test_accumulate_stamps()):
            raise ParamError('Missing pickle files - see printout above.')
            return

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_table.fits'
        table = fio.FITS(filename)[-1].read()
        table_check = np.zeros(len(table)).astype(bool)
        utable = np.unique(table[['sca','dither']])
        chunks = np.linspace(0,self.n_gal,self.n_gal//self.params['meds_size']+1).astype(int)

        # Loop over each sca and dither pickles to accumulate into meds and truth files
        for sca in range(18):
            table_mask_sca = table['sca']==sca
            if np.sum(table_mask_sca)==0:
                print 'no objects with sca',sca
                print 'NEED TO ADD ONE TO LOOP HERE'
                continue
            utable_mask = utable['sca']==sca
            for proc in range(20):
                for dumps in range(10):
                    try:
                        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_stamps_'+str(sca)+'_'+str(proc)+'_'+str(dumps)+'.pickle'
                        gal_exps_,wcs_exps_,wgt_exps_,psf_exps_,dither_list_ = load_obj(filename)
                    except:
                        continue
                    for ichunk,chunk in enumerate(chunks[:-1]):
                        if ichunk==0:
                            continue
                        meds = fio.FITS(self.meds_filename(ichunk),'rw')
                        print self.meds_filename(ichunk)
                        object_data = meds['object_data'].read()
                        image_info = meds['image_info'].read()
                        print '------',time.time()-t0, sca, proc,dumps,ichunk
                        start_exps = 0
                        for ind in range(chunks[ichunk],chunks[ichunk+1]):
                            # if ind%100==0:
                            #     print time.time()-t0,sca,proc,dumps,ichunk,ind#,dither_list_[ind],table[table['gal']==ind],np.unique(table['gal'])
                            if (ind not in gal_exps_):
                                continue
                            if (gal_exps_[ind]==[]):
                                continue
                            # table_mask = np.where(table_mask_sca & (table['gal']==ind) & (np.in1d(table['dither'],dither_list_[ind],assume_unique=False)))[0]
                            # if len(table_mask)==0:
                            #     continue
                            ind_ = ind%self.params['meds_size']
                            j_start = np.argmax(object_data['start_row'][ind_])
                            # print sca,proc,dumps,ichunk,ind#,dither_list_[ind],table[table['gal']==ind],np.unique(table['gal'])
                            for j in range(len(gal_exps_[ind])):
                                # if table_check[table_mask[j]]:
                                #     continue
                                # else:
                                #     table_check[table_mask[j]] = True
                                # print j
                                if object_data['start_row'][ind_][j_start+j-1] != 0:
                                    start_row = object_data['start_row'][ind_][j_start+j-1]/self.params['stamp_size']/self.params['stamp_size']
                                else:
                                    if ind_!=0:
                                        start_exps = np.sum(object_data['ncutout'][object_data['number']<ind_])
                                    start_row = start_exps
                                gal_exps_[ind][j].setOrigin(0,0)
                                wcs = gal_exps_[ind][j].wcs.affine(image_pos=gal_exps_[ind][j].trueCenter())
                                object_data['dudcol'][ind_][j_start+j] = wcs.dudx
                                object_data['dudrow'][ind_][j_start+j] = wcs.dudy
                                object_data['dvdcol'][ind_][j_start+j] = wcs.dvdx
                                object_data['dvdrow'][ind_][j_start+j] = wcs.dvdy
                                object_data['cutout_row'][ind_][j_start+j] = wcs.origin.y
                                object_data['cutout_col'][ind_][j_start+j] = wcs.origin.x
                                object_data['ncutout'][ind_] = j_start+j+1
                                object_data['start_row'][ind_][j_start+j] = start_row * self.params['stamp_size'] * self.params['stamp_size']
                                object_data['psf_start_row'][ind_][j_start+j] = start_row * 64 * 64
                                object_data['file_id'][ind_][j_start+j] = np.where(utable_mask&(utable['dither']==dither_list_[ind][j]))[0]
                                image_info['image_id'][object_data['file_id'][ind_][j_start+j]] = dither_list_[ind][j]
                                image_info['image_ext'][object_data['file_id'][ind_][j_start+j]] = sca

                                meds['image_cutouts'].write(gal_exps_[ind][j].array.flatten(), start=object_data['start_row'][ind_][j_start+j])
                                meds['weight_cutouts'].write(wgt_exps_[ind][j].array.flatten(), start=object_data['start_row'][ind_][j_start+j])
                                meds['psf'].write(psf_exps_[ind][j].array.flatten(), start=object_data['psf_start_row'][ind_][j_start+j])

                                if j_start==0:
                                    j_start+=1
                                    object_data['dudcol'][ind_][j_start+j] = wcs.dudx
                                    object_data['dudrow'][ind_][j_start+j] = wcs.dudy
                                    object_data['dvdcol'][ind_][j_start+j] = wcs.dvdx
                                    object_data['dvdrow'][ind_][j_start+j] = wcs.dvdy
                                    object_data['cutout_row'][ind_][j_start+j] = wcs.origin.y
                                    object_data['cutout_col'][ind_][j_start+j] = wcs.origin.x
                                    object_data['ncutout'][ind_] = j_start+j+1
                                    object_data['start_row'][ind_][j_start+j] = (start_row+1) * self.params['stamp_size'] * self.params['stamp_size']
                                    object_data['psf_start_row'][ind_][j_start+j] = (start_row+1) * 64 * 64
                                    object_data['file_id'][ind_][j_start+j] = np.where(utable_mask&(utable['dither']==dither_list_[ind][j]))[0]
                                    image_info['image_id'][object_data['file_id'][ind_][j_start+j]] = dither_list_[ind][j]
                                    image_info['image_ext'][object_data['file_id'][ind_][j_start+j]] = sca

                                    meds['image_cutouts'].write(gal_exps_[ind][j].array.flatten(), start=object_data['start_row'][ind_][j_start+j])
                                    meds['weight_cutouts'].write(wgt_exps_[ind][j].array.flatten(), start=object_data['start_row'][ind_][j_start+j])
                                    meds['psf'].write(psf_exps_[ind][j].array.flatten(), start=object_data['psf_start_row'][ind_][j_start+j])

                        meds['object_data'].write(object_data)
                        meds['image_info'].write(image_info)
                        meds.close()

        return

    def add_to_meds_obj(self,obj,gal,wgt,psf,coadd=False):

        if coadd:
            obj.n_cutouts+=1
            obj.images.insert(0,obj.images[0])
            obj.weight.insert(0,obj.weight[0])
            obj.seg.insert(0,obj.seg[0])
            obj.wcs.insert(0,obj.wcs[0])
            obj.psf.insert(0,obj.psf[0])
            return obj

        n = len(gal)
        if n == 0:
            return obj

        for i in range(n):
            obj.n_cutouts+=1
            gal[i].setOrigin(0,0)
            obj.images.append( gal[i] )
            obj.weight.append( wgt[i] )
            obj.seg.append( obj.seg[-1] )
            obj.wcs.append( obj.wcs[-1] )
            obj.psf.append( psf[i] )

        return obj

    def dump_stamps_pickle(self,sca,proc,dumps,cnt):

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_stamps_'+str(sca)+'_'+str(proc)+'_'+str(dumps)+'.pickle'
        save_obj([self.gal_exps,self.wcs_exps,self.wgt_exps,self.psf_exps,self.dither_list], filename )

        cnt   = 0
        dumps+= 1
        self.gal_exps    = {}
        self.wcs_exps    = {}
        self.wgt_exps    = {}
        self.psf_exps    = {}
        self.dither_list = {}

        return dumps,cnt

    def dump_meds(self,objs,chunk):
        """
        Accepts a list of meds MultiExposureObject's and writes to meds file.
        """

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_'+str(chunk)+'.fits'
        des.WriteMEDS(objs, filename, clobber=True)

        return

    def dump_truth_gal(self,store,pind,g1,g2):
        """
        Accepts a list of meds MultiExposureObject's and writes to meds file.
        """

        if len(store)!=self.n_gal:
            raise ParamError('Lengths of truth array and expected number of galaxies do not match.')

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_truth_gal.fits'
        out = np.ones(self.n_gal, dtype=[('gal_index','i4')]+[('ra',float)]+[('dec',float)]+[('g1','f4')]+[('g2','f4')]+[('e_index','i2')]+[('rot_angle','i2')]+[('gal_size','f4')]+[('redshift','f4')]+[('magnitude',float)]+[('phot_index','i4')])

        out['gal_index']    = np.arange(len(store))
        out['ra']           = store['ra']
        out['dec']          = store['dec']
        out['rot_angle']    = store['rot']
        out['gal_size']     = store['size']
        out['redshift']     = store['z']
        out['magnitude']    = store['mag']
        out['e_index']      = store['e']
        out['g1']           = g1
        out['g2']           = g2
        out['phot_index']   = pind

        fio.write(filename,out,clobber=True)

        return

    def load_truth_gal(self):
        """
        Accepts a list of meds MultiExposureObject's and writes to meds file.
        """

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_truth_gal.fits'
        store = np.ones(self.n_gal, dtype=[('rot','i2')]+[('e','i2')]+[('size','f4')]+[('z','f4')]+[('mag','f4')]+[('ra',float)]+[('dec',float)])
        out = fio.FITS(filename)[-1].read()

        if len(out)!=self.n_gal:
            raise ParamError('Lengths of truth array and expected number of galaxies do not match.')

        store['rot']  = out['rot_angle']
        store['e']    = out['e_index']
        store['size'] = out['gal_size']
        store['z']    = out['redshift']
        store['mag']  = out['magnitude']
        store['ra']   = out['ra']
        store['dec']  = out['dec']

        return store

    def dump_truth_ind(self,dither_list,sca_list,chunk):
        """
        Accepts a list of meds MultiExposureObject's and writes to meds file.
        """

        depth = 0
        for ind in dither_list.keys():
            if len(dither_list[ind])>depth:
                depth = len(dither_list[ind])

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_truth_ind_'+str(chunk)+'.fits'
        out = np.ones(self.n_gal, dtype=[('gal_index',int)]+[('dither_index',int,(depth))]+[('sca',int,(depth))])
        for name in out.dtype.names:
            out[name] *= -999
        for ind in dither_list.keys():
            stop = len(dither_list[ind])
            out['dither_index'][ind][:stop] = dither_list[ind]
            out['sca'][ind][:stop]          = sca_list[ind]

        fio.write(filename,out,clobber=True)

        return

    def dump_sca_pickle(self,sca,proc):

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_sca_'+str(sca)+'_'+str(proc)+'.pickle'
        save_obj([self.dither_list,self.sca_list,self.xy_list], filename )

        return

    def dump_sca_fits_pickle(self,im,sca,d):

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_image_'+str(sca)+'_'+str(d)+'.pickle'
        save_obj(im, filename)

        return

    def dump_sca_fits(self,im,d):

        filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_image_'+str(d)+'.fits'
        galsim.fits.writeMulti(im, file_name=filename)

        return

    def setup_dither(self, proc=None, pix=None, only_index = False, exact_index = None, dither_list = None):

        fits    = fio.FITS(self.params['dither_file'])[-1]
        date    = fits.read(columns='date')
        dfilter = fits.read(columns='filter')
        dither  = fits.read(columns=['ra','dec','pa'])

        if dither_list is not None:
            for name in dither.dtype.names:
                dither[name] *= np.pi/180.
            return dither[dither_list], Time(date[dither_list],format='mjd').datetime

        if exact_index is not None:
            if exact_index == -1:
                exact_index = np.random.choice(np.arange(len(dither)),1)[0]
                dither = dither[exact_index]
            else:
                dither = dither[exact_index]
            for name in dither.dtype.names:
                dither[name] *= np.pi/180.
            return dither,Time(date[exact_index],format='mjd').datetime,exact_index

        if pix is not None:
            ra,dec = self.get_pix_radec(pix)
            d_      = np.where((np.abs(dither['ra']-ra)<2.5)&(np.abs(dither['dec']-dec)<2.5)&(dfilter == filter_dither_dict[self.params['filter']]))[0]
        if proc is not None:
            d_      = np.where((dither['ra']>24)&(dither['ra']<28.5)&(dither['dec']>-28.5)&(dither['dec']<-24)&(dfilter == filter_dither_dict[self.params['filter']]))[0]
            d_ = d_[proc::self.params['nproc']]
        if only_index:
            return d_
        dfilter = None
        dither  = dither[d_]
        date    = Time(date[d_],format='mjd').datetime

        for name in dither.dtype.names:
            dither[name] *= np.pi/180.

        return dither,date,d_

    def near_pointing(self, obsRA, obsDec, obsPA, ptRA, ptDec):
        """
        Returns mask of objects too far from pointing to consider more expensive checking.
        """

        x = np.cos(ptDec) * np.cos(ptRA)
        y = np.cos(ptDec) * np.sin(ptRA)
        z = np.sin(ptDec)

        d2 = (x - np.cos(obsDec)*np.cos(obsRA))**2 + (y - np.cos(obsDec)*np.sin(obsRA))**2 + (z - np.sin(obsDec))**2
        dist = 2.*np.arcsin(np.sqrt(d2)/2.)

        return np.where(dist<=MAX_RAD_FROM_BORESIGHT)[0].astype('i4')

    def dither_sim(self,pix):
        """
        Loop over each dither - prepares task list for each process and loops over dither_loop().
        """

        # Loops over dithering file
        tasks = []
        for i in range(self.params['nproc']):
            tasks.append({
                'proc'       : i,
                'pix'        : pix,
                'params'     : self.params,
                'store'      : self.store,
                'stars'      : self.stars,
                'table'      : self.table})

        tasks = [ [(job, k)] for k, job in enumerate(tasks) ]

        results = process.MultiProcess(self.params['nproc'], {}, dither_loop, tasks, 'dithering', logger=self.logger, done_func=None, except_func=except_func, except_abort=True)

        return 

    def open_meds(self,pix):

        print self.meds_filename(pix)
        self.meds = fio.FITS(self.meds_filename(pix),'rw')
        self.object_data = self.meds['object_data'].read()
        self.image_info = self.meds['image_info'].read()


    def add_to_meds(self,gal,cumexps,sca,dither):

        ind = np.where(self.object_data['number']==gal)[0]

        for j in range(len(self.gal_exps)):
            self.object_data['ncutout'][ind] = j
            print ind,j,self.object_data['start_row'][ind][j],self.object_data['start_row'][ind],cumexps[ind]+j*self.object_data['box_size'][ind]**2
            self.object_data['start_row'][ind][j] = cumexps[ind]+j*self.object_data['box_size'][ind]**2
            self.object_data['psf_start_row'][ind][j] = cumexps[ind]+j*self.object_data['psf_box_size'][ind]**2
            self.gal_exps[j].setOrigin(0,0)
            wcs = self.gal_exps[j].wcs.affine(image_pos=self.gal_exps[j].trueCenter())
            self.object_data['dudcol'][ind][j] = wcs.dudx
            self.object_data['dudrow'][ind][j] = wcs.dudy
            self.object_data['dvdcol'][ind][j] = wcs.dvdx
            self.object_data['dvdrow'][ind][j] = wcs.dvdy
            self.object_data['cutout_row'][ind][j] = wcs.origin.y
            self.object_data['cutout_col'][ind][j] = wcs.origin.x
            self.object_data['dither'][ind][j] = dither[j]
            self.object_data['sca'][ind][j] = sca[j]

            print self.object_data['start_row'][ind][j],self.object_data['start_row'][ind]
            self.meds['image_cutouts'].write(self.gal_exps[j].array.flatten(), start=self.object_data['start_row'][ind][j])
            self.meds['weight_cutouts'].write(self.wgt_exps[j].array.flatten(), start=self.object_data['start_row'][ind][j])
            self.meds['psf'].write(self.psf_exps[j].array.flatten(), start=self.object_data['psf_start_row'][ind][j])

    def close_meds(self):

        self.meds['object_data'].write(self.object_data)
        self.meds['image_info'].write(self.image_info)
        self.meds.close()

def tabulate_exposures(node=None,nodes=None,proc=None,params=None,store=None,stars=None,max_exp=25,**kwargs):

    sim       = wfirst_sim(params)
    sim.store = store
    sim.stars = stars

    guess = len(sim.store)*max_exp
    output = np.ones(guess,dtype=[('sca',int)]+[('dither',int)]+[('gal',int)])
    for name in output.dtype.names:
        output[name] *= -1
    bound  = galsim.BoundsI(xmin=int(sim.params['stamp_size'])/2,
                        ymin=int(sim.params['stamp_size'])/2,
                        xmax=wfirst.n_pix-int(sim.params['stamp_size'])/2,
                        ymax=wfirst.n_pix-int(sim.params['stamp_size'])/2)

    dither,date,d_ = sim.setup_dither(proc=proc)
    dither = dither[node::nodes]
    date   = date[node::nodes]
    d_     = d_[node::nodes]
    cnt = 0
    for d in range(len(dither)):

        sim.date = date[d]
        sim.WCS = wfirst.getWCS(world_pos=galsim.CelestialCoord(ra=dither['ra'][d]*galsim.radians, 
                                                                dec=dither['dec'][d]*galsim.radians), 
                                PA=dither['pa'][d]*galsim.radians, 
                                date=date[d], 
                                PA_is_FPA=True)

        gal_use_ind = sim.near_pointing(dither['ra'][d], dither['dec'][d], dither['pa'][d], sim.store['ra'], sim.store['dec'])
        if len(gal_use_ind)==0: # If nothing in focal plane, skip dither
            continue

        print d_[d],'out of ',d_[-1]
        for i,ind in enumerate(gal_use_ind):
            # if sim.params['timing']:
            #     if i%100==0:
            #         print 'tab gal loop',i,ind,time.time()-t0
            radec  = galsim.CelestialCoord(sim.store['ra'][ind]*galsim.radians,sim.store['dec'][ind]*galsim.radians)
            sca = wfirst.findSCA(sim.WCS, radec)

            if sca is not None:
                output['sca'][cnt]    = sca
                output['dither'][cnt] = d_[d]
                output['gal'][cnt]    = ind
                cnt+=1

    return output[:cnt]

def dither_loop(calcs):
    """

    """

    pix,params,store,stars,table=calcs

    sim = wfirst_sim(params)
    sim.store = store
    sim.stars = stars
    sim.table = table

    if sim.params['draw_sca']:
        sim.dither_list = [{},{}]
        sim.sca_list    = [{},{}]
        sim.xy_list     = [{},{}]

    sim.open_meds(pix)

    gals = sim.get_pix_gals(pix)
    gals = np.sort(gals)
    tablemask = np.in1d(sim.table['gal'],gals,assume_unique=False)
    gal_      = sim.table['gal'][tablemask]
    sca_      = sim.table['sca'][tablemask]
    dither_   = sim.table['dither'][tablemask]

    dither,date_ = sim.setup_dither(dither_list = dither_)

    cnt   = 0
    dumps = 0
    # Here we carry out the initial steps that are necessary to get a fully chromatic PSF.  We use
    # the getPSF() routine in the WFIRST module, which knows all about the telescope parameters
    # (diameter, bandpasses, obscuration, etc.).
    # only doing this once to save time when its chromatic - need to check if duplicating other steps outweights this, though, once chromatic again
    sim.PSF = wfirst.getPSF(SCAs=np.unique(sca_), 
                            approximate_struts=sim.params['approximate_struts'], 
                            n_waves=sim.params['n_waves'], 
                            logger=sim.logger, 
                            wavelength=sim.bpass)
    # sim.logger.info('Done PSF precomputation in %.1f seconds!'%(time.time()-t0))


    exps = np.bincount(gals)
    cumexps = np.cumsum(exps+1)
    for gal in gals[:100]:
        sim.gal_exps    = []
        sim.wcs_exps    = []
        sim.wgt_exps    = []
        sim.psf_exps    = []

        galmask = np.where(gal_==gal)
        date = date_[galmask]
        sca = sca_[galmask]
        for idither,d in enumerate(dither[galmask]):
            sim.date = date[idither]

            # Get the WCS for an observation at this position. We are not supplying a date, so the routine
            # will assume it's the vernal equinox. The output of this routine is a dict of WCS objects, one 
            # for each SCA. We then take the WCS for the SCA that we are using.
            sim.WCS = wfirst.getWCS(world_pos=galsim.CelestialCoord(ra=d['ra']*galsim.radians, 
                                                                    dec=d['dec']*galsim.radians), 
                                    PA=d['pa']*galsim.radians, 
                                    date=sim.date,
                                    SCAs=sca,
                                    PA_is_FPA=True)

            if sim.params['draw_sca']:
                print 'Troxel broke draw_sca - sorry'
                sim.radec     = sim.WCS.toWorld(galsim.PositionD(wfirst.n_pix/2, wfirst.n_pix/2))
                sim.local_wcs = sim.WCS
                im,wgt = sim.draw_sca(sca,proc,dither,d_,d)
                if im is not None:
                    sim.dump_sca_fits_pickle([im,wgt],sca,d_[d])
            else:
                # cnt,dumps = sim.draw_pure_stamps(sca,proc,dither,d_,d,cnt,dumps)
                out = sim.draw_galaxy(gal,sca[idither],None)

            sim.gal_exps.append(out[0])
            sim.wcs_exps.append(sim.local_wcs)
            sim.wgt_exps.append(out[1])
            if sim.params['draw_true_psf']:
                sim.psf_exps.append(out[2]) 
            if idither==0:
                sim.gal_exps.append(out[0])
                sim.wcs_exps.append(sim.local_wcs)
                sim.wgt_exps.append(out[1])
                if sim.params['draw_true_psf']:
                    sim.psf_exps.append(out[2]) 

        sim.add_to_meds(gal,cumexps,sca,dither_[galmask])


    sim.close_meds()

    # if sim.params['draw_sca']:
    #     sim.dump_sca_pickle(sca,proc)
    # else:
    #     dumps,cnt = sim.dump_stamps_pickle(sca,proc,dumps,cnt)

    print 'dither loop done for pix ',pix

    if sim.params['break_cnt'] == -1:
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('time')
        ps.print_stats(100)
        sys.exit()

    return 

def pix_loop(calcs):

    params,pix,store,stars,table=calcs
    sim       = wfirst_sim(params)
    sim.store = store
    sim.stars = stars
    sim.table = table

    # Dither function that loops over pointings, SCAs, objects for each filter loop.
    # Returns a meds MultiExposureObject of galaxy stamps, psf stamps, and wcs.
    if sim.params['timing']:
        print 'before dither sim',time.time()-t0
    sim.dither_sim(pix)

    return

def tab_loop(calcs):

    params,node,nodes,store,stars=calcs
    sim       = wfirst_sim(params)
    sim.store = store
    sim.stars = stars

    if sim.params['timing']:
        print 'before dither sim',time.time()-t0
    results = sim.tabulate_exposures_loop(node,nodes)

    return results

def acc_loop(chunk = None, params = None, **kwargs):

    sim = wfirst_sim(params)
    sim.accumulate_stamps(chunk,ignore_missing_files=True)

    return

def get_psf_star(oversample,flux,PSF,WCS,bp,sca,star_sed):
    local_wcs = WCS.local(galsim.PositionD(wfirst.n_pix/2,wfirst.n_pix/2))
    local_wcs = galsim.JacobianWCS(dudx=local_wcs.dudx/oversample,dudy=local_wcs.dudy/oversample,dvdx=local_wcs.dvdx/oversample,dvdy=local_wcs.dvdy/oversample)
    stamp = galsim.Image(128*oversample, 128*oversample, wcs=local_wcs)

    star = galsim.DeltaFunction() * star_sed
    star = star.withFlux(flux,bp)
    star = galsim.Convolve(star, PSF, gsparams=big_fft_params)
    star.drawImage(bp,image=stamp) # draw galaxy stamp
    return stamp

def test_psf_loop(sca=None,n_wave=None,star_sed=None,WCS=None,PSF=None,bp=None,stars=None,**kwargs):

    for oversample in [1,2,4,8,16,32]:
        stamps = {}
        for filter_ in filter_dither_dict.keys():
            stamps[filter_] = {}
            print n_wave,sca,oversample,filter_
            stamps[filter_]['min'] = get_psf_star(oversample,np.min(stars[filter_]),PSF,WCS,bp[filter_],sca,star_sed)
            stamps[filter_]['max'] = get_psf_star(oversample,np.max(stars[filter_]),PSF,WCS,bp[filter_],sca,star_sed)
            stamps[filter_]['mid'] = get_psf_star(oversample,np.mean(stars[filter_]),PSF,WCS,bp[filter_],sca,star_sed)

        save_obj(stamps,'psf_test_'+str(n_wave)+'_'+str(oversample)+'_'+str(sca)+'.pickle')

    return

def test_psf_sampling(yaml):

    sim = wfirst_sim(yaml)
    dither,date,d_ = sim.setup_dither(proc=0,exact_index=-1)
    print dither,date,d_
    stars = fio.FITS(sim.params['star_sample'])[-1].read()
    WCS = wfirst.getWCS(world_pos=galsim.CelestialCoord(ra=dither['ra']*galsim.radians, dec=dither['dec']*galsim.radians), PA=dither['pa']*galsim.radians, date=date, PA_is_FPA=True)
    PSF_ = wfirst.getPSF(logger=sim.logger)
    bp   = galsim.wfirst.getBandpasses()
    print 'start loop',time.time()-t0

    for n_wave in [2,4,8,16,32,-1]:
        PSF = {}
        if n_wave > 0:
            blue_limit, red_limit = wfirst.wfirst_psfs._find_limits(['J129', 'F184', 'W149', 'Y106', 'Z087', 'H158'], bp)
            for key in PSF_.keys():
                PSF[key] = PSF_[key].interpolate(waves=np.linspace(blue_limit, red_limit, n_wave),oversample_fac=1.5)
        else:
            PSF = PSF_
        print 'start inner loop',n_wave,time.time()-t0

        tasks = []
        for i in range(18):
            tasks.append({
                'sca'      : i,
                'n_wave'   : n_wave,
                'star_sed' : sim.star_sed,
                'WCS'      : WCS[i+1],
                'PSF'      : PSF[i+1],
                'bp'       : bp,
                'stars'    : stars})

        tasks = [ [(job, k)] for k, job in enumerate(tasks) ]

        results = process.MultiProcess(9, {}, test_psf_loop, tasks, 'psf_test', logger=sim.logger, done_func=None, except_func=except_func, except_abort=True)

    return

def test_psf_sampling_2(yaml,sca):

    import galsim
    import pickle
    import numpy as np
    import matplotlib
    matplotlib.use ('agg')
    import matplotlib.pyplot as plt
    import os
    import inspect
    import matplotlib.cm as cm
    from matplotlib.colors import LogNorm
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import pylab
    import time
    filter_dither_dict = {
        'J129' : 3,
        'F184' : 1,
        'Y106' : 4,
        'H158' : 2
    }

    def load_obj(name ):
        with open(name, 'rb') as f:
            return pickle.load(f)


    def tmp(sca):
        l=0
        flux='mid'
        trueout = np.zeros((4,5))
        truth = load_obj('psf_test_'+str(8)+'_'+str(32)+'_'+str(sca)+'.pickle')
        for k,filter_ in enumerate(filter_dither_dict.keys()):
            tmp = hsm(truth[filter_][flux])
            trueout[k,:] = tmp[:-1]
            tmp = truth[filter_][flux].array
            tmp = tmp[48*32:80*32,:]
            tmp = tmp[:,48*32:80*32]
            plt.figure(figsize=(8, 8), dpi=160)
            plt.imshow(tmp,interpolation='none')
            plt.colorbar()
            plt.savefig('psf_truth_'+str(sca)+'_'+filter_+'_'+flux+'.eps', bbox_inches='tight')
            plt.close()
        t0=time.time()
        out = np.zeros((3,6,4,8))
        for i,n_wave in enumerate([2,4,8]):
            for j,oversample in enumerate([1,2,4,8,16,32]):
                print n_wave,oversample,time.time()-t0
                if not os.path.exists('psf_test_'+str(n_wave)+'_'+str(oversample)+'_'+str(sca)+'.pickle'):
                    continue
                stamps = load_obj('psf_test_'+str(n_wave)+'_'+str(oversample)+'_'+str(sca)+'.pickle')
                for k,filter_ in enumerate(filter_dither_dict.keys()):
                    tmp = hsm(stamps[filter_][flux])
                    out[i,j,k,:-3] = tmp[:-1]
                    out[i,j,k,4] = tmp.moments_centroid.y
                    # tmp = truth[sca][filter_][flux].reshape(-1, (32*128)/(32/oversample), 32/oversample).sum(axis=-1).T
                    # tmp = tmp.T[sca].reshape(-1, (32*128)/(32/oversample), 32/oversample).sum(axis=-1)
                    # tmp = stamps[filter_][flux] - tmp.T
                    tmp = np.repeat(stamps[filter_][flux].array,32/oversample,axis=0)
                    tmp = np.repeat(tmp,32/oversample,axis=1)
                    assert np.shape(tmp) == np.shape(truth[filter_][flux].array)
                    tmp = (tmp/(32/oversample)**2 - truth[filter_][flux].array)/truth[filter_][flux].array
                    tmp = tmp[48*32:80*32,:]
                    tmp = tmp[:,48*32:80*32]
                    out[i,j,k,5] = np.min(tmp)
                    out[i,j,k,6] = np.mean(tmp)
                    out[i,j,k,7] = np.max(tmp)
                    # plt.figure(figsize=(8, 8), dpi=512)
                    plt.figure(figsize=(8, 8), dpi=160)
                    plt.imshow(tmp,interpolation='none',vmin=-2,vmax=2)
                    plt.colorbar()
                    plt.savefig('psf_diff_'+str(n_wave)+'_'+str(oversample)+'_'+str(sca)+'_'+filter_+'_'+flux+'.eps', bbox_inches='tight')
                    plt.close()
        for k,filter_ in enumerate(filter_dither_dict.keys()):
            for i,marker in enumerate(['x','s','v']):
                if i==0:
                    plt.plot(np.arange(6),out[i,:,k,0],color='r',marker=marker,label='e1')
                    plt.plot(np.arange(6),out[i,:,k,1],color='b',marker=marker,label='e2')
                else:
                    plt.plot(np.arange(6),out[i,:,k,0],color='r',marker=marker)
                    plt.plot(np.arange(6),out[i,:,k,1],color='b',marker=marker)
            plt.axhline(trueout[k,0],color='k')
            plt.axhline(trueout[k,1],color='k')
            plt.xlabel('2^x oversampling (nwave: x=2,s=4,v=8)')
            plt.ylabel('measured ellipticity')
            plt.legend()
            plt.savefig('psf_diff_e_'+'_'+str(sca)+'_'+filter_+'_'+flux+'.png', bbox_inches='tight')
            plt.close()
            for i,marker in enumerate(['x','s','v']):
                plt.plot(np.arange(6),out[i,:,k,2]*((2**5)/(2**np.arange(6))),color='b',marker=marker)
            plt.axhline(trueout[k,2],color='k')
            plt.xlabel('2^x oversampling (nwave: x=2,s=4,v=8)')
            plt.ylabel('measured sigma')
            plt.savefig('psf_diff_sigma_'+'_'+str(sca)+'_'+filter_+'_'+flux+'.png', bbox_inches='tight')
            plt.close()
            for i,marker in enumerate(['x','s','v']):
                if i==0:
                    plt.plot(np.arange(6),out[i,:,k,3]*((2**5)/(2**np.arange(6))),color='r',marker=marker,label='x')
                    plt.plot(np.arange(6),out[i,:,k,4]*((2**5)/(2**np.arange(6))),color='b',marker=marker,label='y')
                else:
                    plt.plot(np.arange(6),out[i,:,k,3]*((2**5)/(2**np.arange(6))),color='r',marker=marker)
                    plt.plot(np.arange(6),out[i,:,k,4]*((2**5)/(2**np.arange(6))),color='b',marker=marker)
            plt.axhline(trueout[k,3],color='k')
            plt.axhline(trueout[k,4],color='k')
            plt.xlabel('2^x oversampling (nwave: x=2,s=4,v=8)')
            plt.ylabel('measured centroid')
            plt.legend()
            plt.savefig('psf_diff_cent_'+'_'+str(sca)+'_'+filter_+'_'+flux+'.png', bbox_inches='tight')
            plt.close()
        for k,filter_ in enumerate(filter_dither_dict.keys()):
            for i,marker in enumerate(['x','s','v']):
                if i==0:
                    plt.plot(np.arange(6),(out[i,:,k,0]-trueout[k,0])/trueout[k,0],color='r',marker=marker,label='e1')
                    plt.plot(np.arange(6),(out[i,:,k,1]-trueout[k,1])/trueout[k,1],color='b',marker=marker,label='e2')
                else:
                    plt.plot(np.arange(6),(out[i,:,k,0]-trueout[k,0])/trueout[k,0],color='r',marker=marker)
                    plt.plot(np.arange(6),(out[i,:,k,1]-trueout[k,1])/trueout[k,1],color='b',marker=marker)
            plt.axhline(0.,color='k')
            plt.xlabel('2^x oversampling (nwave: x=2,s=4,v=8)')
            plt.ylabel('measured ellipticity (de/e)')
            plt.legend()
            plt.savefig('psf_diff_frac_e_'+'_'+str(sca)+'_'+filter_+'_'+flux+'.png', bbox_inches='tight')
            plt.close()
            for i,marker in enumerate(['x','s','v']):
                plt.plot(np.arange(6),(out[i,:,k,2]*((2**5)/(2**np.arange(6)))-trueout[k,2])/trueout[k,2],color='b',marker=marker)
            plt.axhline(0.,color='k')
            plt.xlabel('2^x oversampling (nwave: x=2,s=4,v=8)')
            plt.ylabel('measured sigma (dsigma/sigma)')
            plt.savefig('psf_diff_frac_sigma_'+'_'+str(sca)+'_'+filter_+'_'+flux+'.png', bbox_inches='tight')
            plt.close()
            for i,marker in enumerate(['x','s','v']):
                if i==0:
                    plt.plot(np.arange(6),(out[i,:,k,3]*((2**5)/(2**np.arange(6)))-trueout[k,3])/trueout[k,3],color='r',marker=marker,label='x')
                    plt.plot(np.arange(6),(out[i,:,k,4]*((2**5)/(2**np.arange(6)))-trueout[k,4])/trueout[k,4],color='b',marker=marker,label='y')
                else:
                    plt.plot(np.arange(6),(out[i,:,k,3]*((2**5)/(2**np.arange(6)))-trueout[k,3])/trueout[k,3],color='r',marker=marker)
                    plt.plot(np.arange(6),(out[i,:,k,4]*((2**5)/(2**np.arange(6)))-trueout[k,4])/trueout[k,4],color='b',marker=marker)
            plt.axhline(0.,color='k')
            plt.xlabel('2^x oversampling (nwave: x=2,s=4,v=8)')
            plt.ylabel('measured centroid (dc/c)')
            plt.legend()
            plt.savefig('psf_diff_frac_cent_'+'_'+str(sca)+'_'+filter_+'_'+flux+'.png', bbox_inches='tight')
            plt.close()
        return out
    return

pr = cProfile.Profile()

if __name__ == "__main__":
    """
    """

    # test_psf_sampling(sys.argv[1])
    # sys.exit()
    pr.enable()

    # This instantiates the simulation based on settings in input param file (argv[1])
    sim = wfirst_sim(sys.argv[1])

    if sim.params['accumulate']:
        if sim.params['draw_sca']:
            sim.accumulate_sca()
        else:
            # tasks = []
            # for chunk in range(sim.n_gal//sim.params['meds_size']):
            #     tasks.append({
            #         'chunk'       : chunk,
            #         'params'     : sim.params})

            # tasks = [ [(job, k)] for k, job in enumerate(tasks) ]

            # results = process.MultiProcess(len(tasks), {}, acc_loop, tasks, 'accumulate', logger=sim.logger, done_func=None, except_func=except_func, except_abort=True)
            sim.accumulate_stamps(ignore_missing_files=True)

        # pr.disable()
        # ps = pstats.Stats(pr).sort_stats('time')
        # ps.print_stats(100)
        sys.exit()

    if sim.params['mpi']:
        from mpi_pool import MPIPool
        comm = mpi4py.MPI.COMM_WORLD
        pool = MPIPool(comm)
        if not pool.is_master():
            sim = None
            pool.wait()
            sys.exit(0)

    if sim.params['timing']:
        print 'before init galaxy',time.time()-t0
    # Initiate unique galaxy image list and noise models
    sim.store = sim.init_galaxy()
    if sim.params['draw_stars']:
        sim.stars = sim.init_star()
    else:
        sim.stars = None
    if sim.params['timing']:
        print 'after init galaxy',time.time()-t0

    if sim.params['rerun_tabulation'] or (not sim.compile_tab()):
        print '... ... ... ... ... ...'
        calcs = []
        if sim.params['mpi']:
            nodes = comm.Get_size()
            for node in range(nodes):
                calcs.append((sim.params,node,nodes,sim.store,sim.stars))
            results = pool.map(tab_loop, calcs)
            pool.close()
        else:
            results = map(tab_loop, [sim.params,0,1,sim.store,sim.stars])

        sim.compile_tab(results = results)

    if sim.params['remake_meds']:
        sim.compile_tab()

    # define loop over SCAs
    if sim.params['simulate_run']:
        calcs = []
        pix = sim.get_totpix()
        for i in pix:
            calcs.append((i,sim.params,sim.store,sim.stars,sim.table))
        if sim.params['mpi']:
            pool.map(dither_loop, calcs)
            pool.close()
        else:
            map(dither_loop, calcs)

    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('time')
    # ps.print_stats(100)



#export PYTHONPATH=$PYTHONPATH:/users/PCON0003/cond0083/im3shape-git/
#python -m py3shape.analyze_meds /fs/scratch/cond0083/wfirst_sim_out/test_H158_0.fits disc_ini.txt all test.out 0 100000

