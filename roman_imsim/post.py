
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

import matplotlib
matplotlib.use ('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pylab

from .sim import roman_sim 
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

filter_dither_dict_ = {
    3:'J129',
    1:'F184',
    4:'Y106',
    2:'H158'
}

class postprocessing(roman_sim):
    """
    Roman image simulation postprocssing functions.

    Input:
    param_file : File path for input yaml config file or yaml dict. Example located at: ./example.yaml.
    """

    def __init__(self, param_file):
        super().__init__(param_file)

        self.final_scale = 0.0575
        self.final_nxy   = 7825+1000 # SCA size + 500 pixel buffer
        self.dd_         = self.final_scale*self.final_nxy/60/60/2
        self.dd          = self.final_scale*(self.final_nxy-1000)/60/60/2
        self.dsca        = 0.11*4088/60/60/2

        return

    def verify_output_files(self,cap=None):

        d = np.loadtxt(self.params['dither_from_file'])
        if cap is None:
            cap = len(d)
        d = d[:cap].astype(int)
        pointings  = fio.FITS(self.params['dither_file'])[-1][d[:,0]]
        filename = get_filename(self.params['out_path'],
                                'truth',
                                self.params['output_truth'],
                                name2='truth_gal',
                                overwrite=False)
        gal = fio.FITS(filename)[-1][['ra','dec']][:]
        gal['ra']*=180./np.pi
        gal['dec']*=180./np.pi

        # truth dir
        truth = []
        f = glob.glob(self.params['out_path']+'/truth/'+self.params['output_meds']+'*')
        for j,d_ in enumerate(d):
            s = '_'+str(d_[0])+'_'+str(d_[1])+'_'
            test = [i for i in f if s in i]
            s = '_'+str(d_[0])+'_'+str(d_[1])+'.'
            test.append( [i for i in f if s in i] )
            if len(test) != 2:
                truth.append(d_)
        truth = np.array(truth)
        print('........',len(truth))

        self.setup_pointing()
        #truth plot
        plt.hist2d(gal['ra'],gal['dec'],bins=500)
        for d_ in truth:
            self.update_pointing(dither=d_[0],sca=d_[1],psf=False)
            print('missing truth',j,test,d_[0],d_[1],self.pointing.radec.ra/galsim.degrees,self.pointing.radec.dec/galsim.degrees)
            plt.plot(self.pointing.radec.ra/galsim.degrees,self.pointing.radec.dec/galsim.degrees,marker='.',ls='',color='r')
        plt.savefig('missing_truth.png')
        plt.close()

        # images dir
        images = []
        f = glob.glob(self.params['out_path']+'/images/'+self.params['output_meds']+'*')
        for j,d_ in enumerate(d):
            s = '_'+str(d_[0])+'_'+str(d_[1])+'_'
            test = [i for i in f if s in i]
            s = '_'+str(d_[0])+'_'+str(d_[1])+'.'
            test.append( [i for i in f if s in i] )
            if len(test) != 1:
                images.append(d_)
                print('missing images',j,test,d_[0],d_[1])
        images = np.array(images)

        #images plot
        plt.hist2d(gal['ra'],gal['dec'],bins=500)
        for d_ in images:
            self.update_pointing(dither=d_[0],sca=d_[1],psf=False)
            plt.plot(self.pointing.radec.ra/galsim.degrees,self.pointing.radec.dec/galsim.degrees,marker='.',ls='',color='r')
        plt.savefig('missing_images.png')
        plt.close()

        # stamps dir
        stamps = []
        f = glob.glob(self.params['out_path']+'/stamps/'+self.params['output_meds']+'*')
        for j,d_ in enumerate(d):
            s = '_'+str(d_[0])+'_'+str(d_[1])+'_'
            test = [i for i in f if s in i]
            s = '_'+str(d_[0])+'_'+str(d_[1])+'.'
            test.append( [i for i in f if s in i] )
            if len(test) != 2:
                stamps.append(d_)
                print('missing stamps',j,test,d_[0],d_[1])
        stamps = np.array(stamps)

        #stamps plot
        plt.hist2d(gal['ra'],gal['dec'],bins=500)
        for d_ in stamps:
            self.update_pointing(dither=d_[0],sca=d_[1],psf=False)
            plt.plot(self.pointing.radec.ra/galsim.degrees,self.pointing.radec.dec/galsim.degrees,marker='.',ls='',color='r')
        plt.savefig('missing_stamps.png')
        plt.close()

        return

    def setup_pointing(self,filter_=None):
        """
        Set up initial objects.

        Input:
        filter_ : A filter name. 'None' to determine by dither.
        """

        if (filter_!='None')&(filter_ is not None):
            # Filter be present in filter_dither_dict{} (exists in survey strategy file).
            if filter_ not in list(filter_dither_dict.keys()):
                raise ParamError('Supplied invalid filter: '+filter_)

        # This sets up a mostly-unspecified pointing object in this filter. We will later specify a dither and SCA to complete building the pointing information.
        if (filter_=='None') or (filter_ is None):
            self.pointing = pointing(self.params,self.logger,filter_=None,sca=None,dither=None,rank=self.rank)
        else:
            self.pointing = pointing(self.params,self.logger,filter_=filter_,sca=None,dither=None,rank=self.rank)

    def update_pointing(self,dither=None,sca=None,psf=True):

        if dither is not None:
            # This updates the dither
            self.pointing.update_dither(dither,force_filter=True)
        if sca is not None:
            # This sets up a specific pointing for this SCA (things like WCS, PSF)
            self.pointing.update_sca(sca,psf=psf)

    def merge_fits_old(self):

        dither = fio.FITS(self.params['dither_file'])[-1]
        for d in np.loadtxt(self.params['dither_from_file']):
            print(int(d))
            f = filter_dither_dict_[dither['filter'][d]]
            self.update_pointing(dither=d)
            filename_ = get_filename(self.params['tmpdir'],
                                    '',
                                    self.params['output_meds'],
                                    var=f+'_'+str(int(d)),
                                    ftype='fits',
                                    overwrite=True)
            if os.path.exists(filename_+'.gz'):
                continue
            out = fio.FITS(filename_,'rw')
            i_=0
            for sca in range(1,19):
                filename = get_filename(self.params['out_path'],
                                        'images',
                                        self.params['output_meds'],
                                        var=f+'_'+str(int(d)),
                                        name2=str(sca),
                                        ftype='fits.gz',
                                        overwrite=False)
                if not os.path.exists(filename):
                    continue
                filename = filename[:-3]
                tmp = fio.FITS(filename,'r')[-1]
                hdr = tmp.read_header()
                hdr = self.prep_new_header(hdr,sca)
                data = tmp.read()
                hdr['extname'] = 'SCI'
                out.write(data,header=hdr)
                hdr = tmp.read_header()
                hdr = self.prep_new_header(hdr,sca)
                hdr['extname'] = 'ERR'
                sky = self.get_sky_inv(sca)
                out.write(sky.array,header=hdr)
                hdr = tmp.read_header()
                hdr = self.prep_new_header(hdr,sca)
                hdr['extname'] = 'DQ'
                out.write(np.zeros_like(data,dtype='int16'),header=hdr)
                i_+=1
            out.close()
            # os.system('gzip '+filename_)

    def merge_fits(self):

        dither = fio.FITS(self.params['dither_file'])[-1]
        for d in np.loadtxt(self.params['dither_from_file']):
            print(int(d))
            f = filter_dither_dict_[dither['filter'][d]]
            self.update_pointing(dither=d)
            filename_ = get_filename(self.params['out_path'],
                                    'images/visits',
                                    self.params['output_meds'],
                                    var=f+'_'+str(int(d)),
                                    ftype='fits.gz',
                                    overwrite=True)
            if os.path.exists(filename_+'.gz'):
                continue
            out = fio.FITS(filename_,'rw')
            for sca in range(1,19):
                filename = get_filename(self.params['out_path'],
                                        'images',
                                        self.params['output_meds'],
                                        var=f+'_'+str(int(d)),
                                        name2=str(sca),
                                        ftype='fits.gz',
                                        overwrite=False)
                if not os.path.exists(filename):
                    continue
                filename = filename[:-3]
                tmp = fio.FITS(filename,'r')[('SCI',sca)]
                hdr = tmp.read_header()
                data = tmp.read()
                out.write(data,header=hdr)
                tmp = fio.FITS(filename,'r')[('ERR',sca)]
                hdr = tmp.read_header()
                data = tmp.read()
                out.write(data,header=hdr)
                tmp = fio.FITS(filename,'r')[('DQ',sca)]
                hdr = tmp.read_header()
                data = tmp.read()
                out.write(data,header=hdr)
            out.close()
            # os.system('gzip '+filename_)

    def get_sky_inv(self,sca):
        self.update_pointing(sca=sca)
        sky_level = roman.getSkyLevel(self.pointing.bpass, world_pos=self.pointing.radec, date=self.pointing.date)
        sky_level *= (1.0 + roman.stray_light_fraction)
        b   = galsim.BoundsI(  xmin=1,
                                ymin=1,
                                xmax=roman.n_pix,
                                ymax=roman.n_pix)
        xy = self.pointing.WCS.toImage(self.pointing.radec)
        local_wcs = self.pointing.WCS.local(xy)
        sky_stamp = galsim.Image(bounds=b, wcs=local_wcs)
        local_wcs.makeSkyImage(sky_stamp, sky_level)
        sky_stamp += roman.thermal_backgrounds[self.pointing.filter]*roman.exptime
        noise = galsim.PoissonNoise(galsim.BaseDeviate(1))
        sky_stamp.addNoise(noise)
        sky_stamp.quantize()
        dark_current_ = roman.dark_current*roman.exptime
        sky_stamp = (sky_stamp + round(dark_current_))
        sky_stamp = sky_stamp/roman.gain
        sky_stamp.quantize()
        sky_stamp.invertSelf()
        return sky_stamp

    def prep_new_header(self,hdr):
        hdr['GS_XMIN']  = hdr['GS_XMIN']#[0]
        hdr['GS_XMIN']  = hdr['GS_YMIN']#[0]
        hdr['GS_WCS']   = hdr['GS_WCS']#[0]
        hdr['extver']   = 1
        hdr['Detector'] = 'IR'
        hdr['PROPOSID'] = 'HLS_SIT'
        hdr['LINENUM']  = 'None'
        hdr['TARGNAME'] = 'HLS'
        hdr['EXPTIME']  = 140.25
        hdr['ROOTNAME'] = self.params['output_meds']
        hdr['INSTRUME'] = 'WFC3'
        hdr['NGOODPIX'] = 4088*4088-1
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
        return hdr

    def accumulate_index(self):

        filename_ = get_filename(self.params['out_path'],
                                'truth',
                                self.params['output_meds'],
                                var='index',
                                ftype='fits.gz',
                                overwrite=True)
        filename_star_ = get_filename(self.params['out_path'],
                                'truth',
                                self.params['output_meds'],
                                var='index_star',
                                ftype='fits.gz',
                                overwrite=True)
        limits_filename = get_filename(self.params['out_path'],
                                'truth',
                                self.params['output_meds'],
                                var='radec',
                                ftype='txt',
                                overwrite=True)

        self.setup_pointing()
        start=True
        gal_i = 0
        star_i = 0
        dither = np.loadtxt(self.params['dither_from_file'])
        limits = np.ones((len(dither),2))*-999
        for i,(d,sca) in enumerate(dither.astype(int)):
            print(d,sca)
            self.update_pointing(dither=d,sca=sca,psf=False)
            f = filter_dither_dict_[fio.FITS(self.params['dither_file'])[-1][int(d)]['filter']]
            filename = get_filename(self.params['out_path'],
                                    'truth',
                                    self.params['output_meds'],
                                    var='index',
                                    name2=f+'_'+str(d)+'_'+str(sca),
                                    ftype='fits',
                                    overwrite=False)
            filename_star = get_filename(self.params['out_path'],
                                    'truth',
                                    self.params['output_meds'],
                                    var='index',
                                    name2=f+'_'+str(d)+'_'+str(sca)+'_star',
                                    ftype='fits',
                                    overwrite=False)
            if start:
                gal = fio.FITS(filename)[-1].read()
                star = fio.FITS(filename_star)[-1].read()
                start=False
                gal = np.zeros(100000000,dtype=gal.dtype)
                star = np.zeros(100000000,dtype=star.dtype)
                gal['ind']=-1
                star['ind']=-1
            try:
                tmp = fio.FITS(filename)[-1].read()
            except:
                print('missing',filename)
            for name in gal.dtype.names:
                gal[name][gal_i:gal_i+len(tmp)] = tmp[name]
            gal_i+=len(tmp)
            limits[i,0] = self.pointing.radec.ra/galsim.degrees
            limits[i,1] = self.pointing.radec.dec/galsim.degrees
            try:
                tmp = fio.FITS(filename_star)[-1].read()
            except:
                print('missing',filename_star)
            for name in star.dtype.names:
                star[name][star_i:star_i+len(tmp)] = tmp[name]
            star_i+=len(tmp)
        gal=gal[gal['ind']!=-1]
        star=star[star['ind']!=-1]
        gal = np.sort(gal,order=['ind','dither','sca'])
        star = np.sort(star,order=['ind','dither','sca'])
        fio.write(filename_,gal)
        f = fio.FITS(filename_,'rw',clobber=True)
        f.write(gal)
        f.close()
        f = fio.FITS(filename_star_,'rw',clobber=True)
        f.write(star)
        f.close()
        np.savetxt(limits_filename,limits)

    def get_psf_fits(self):

        for filter_ in ['Y106','J129','H158','F184']:
            low = fio.FITS('PSF_model_'+filter_+'_low.fits','rw')
            high = fio.FITS('PSF_model_'+filter_+'_high.fits','rw')
            self.setup_pointing(filter_)
            for sca in range(1,19):
                self.pointing.update_dither(15459)
                self.pointing.update_sca(sca)
                for star in [False,True]:
                    psf_stamp = galsim.Image(257,257, wcs=self.pointing.WCS)
                    st_model = galsim.DeltaFunction(flux=1.)
                    gsparams = galsim.GSParams( maximum_fft_size=16384 )
                    st_model = st_model.evaluateAtWavelength(self.pointing.bpass.effective_wavelength)
                    st_model = st_model.withFlux(1.)
                    if star:
                        psf = self.pointing.load_psf(None,star=True)
                        psf = psf.withGSParams(galsim.GSParams(folding_threshold=1e-3))
                        st_model = galsim.Convolve(st_model, psf)
                    else:
                        psf = self.pointing.load_psf(None,star=False)
                        st_model = galsim.Convolve(st_model, psf)
                    st_model.drawImage(image=psf_stamp,wcs=self.pointing.WCS)
                    hdr={}
                    psf_stamp.wcs.writeToFitsHeader(hdr,psf_stamp.bounds)
                    hdr['extname']  = 'sca_'+str(sca)
                    if star:
                        high.write(psf_stamp.array,header=hdr)
                    else:
                        low.write(psf_stamp.array,header=hdr)
            low.close()
            high.close()

    def near_coadd(self,ra,dec):
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        d2 = (x - self.cdec*self.cra)**2 + (y - self.cdec*self.sra)**2 + (z - self.sdec)**2
        return np.where(np.sqrt(d2)/2.<=np.sin(0.009/2.))[0]

    def generate_coaddlist(self):

        limits_filename = get_filename(self.params['out_path'],
                                'truth',
                                self.params['output_meds'],
                                var='radec',
                                ftype='txt',
                                overwrite=False)

        self.limits = np.loadtxt(limits_filename)
        # self.limits = self.limits[self.limits[:,0]!=-999]

        dither = fio.FITS(self.params['dither_file'])[-1].read()
        dither_list = np.loadtxt(self.params['dither_from_file']).astype(int)

        dec = np.arange(180/2./self.dd)*2*self.dd-90+self.dd
        coaddlist = np.empty((180*5)*(360*5),dtype=[('tilename','S11'), ('coadd_i','i8'), ('coadd_j','i8'), ('coadd_ra',float), ('coadd_dec',float), ('d_dec',float), ('d_ra',float), ('input_list','i8',(4,100))])
        coaddlist['coadd_i'] = -1
        coaddlist['input_list'] = -1
        i_ = 0
        for j in range(len(dec)):
            if dec[j]-self.dd_>np.max(self.limits[:,1][self.limits[:,1]!=-999])+self.dsca:
                continue
            if dec[j]+self.dd_<np.min(self.limits[:,1][self.limits[:,1]!=-999])-self.dsca:
                continue
            print('----',j,dec[j])
            cosdec = np.cos(np.radians(dec[j]))
            dra = self.dd/cosdec
            ra  = []
            for i in range(1800):
                ra_ = i*dra*2.+dra
                if ra_>360.:
                    break
                ra.append(ra_)
            ra = np.array(ra)
            for i in range(len(ra)):
                if ra[i]-self.dd_>np.max(self.limits[:,0][self.limits[:,0]!=-999])+self.dsca:
                    continue
                if ra[i]+self.dd_<np.min(self.limits[:,0][self.limits[:,0]!=-999])-self.dsca:
                    continue

                print(j,i,ra[i])
                coaddlist['coadd_i'][i_] = i
                coaddlist['coadd_j'][i_] = j
                coaddlist['tilename'][i_] = "{:.2f}".format(ra[i])+'_'+"{:.2f}".format(dec[j])
                coaddlist['d_ra'][i_] = dra
                coaddlist['coadd_ra'][i_] = ra[i]
                coaddlist['d_dec'][i_] = self.dd
                coaddlist['coadd_dec'][i_] = dec[j]

                mask = np.where((self.limits[:,0]>ra[i]-self.dd_)&(self.limits[:,0]<ra[i]+self.dd_)&(self.limits[:,1]>dec[j]-self.dd_)&(self.limits[:,1]<dec[j]+self.dd_))[0]

                f = dither['filter'][dither_list[mask,0]]

                for fi in range(4):
                    for di in range(np.sum(f==fi+1)):
                        coaddlist['input_list'][i_][fi,di] = mask[f==fi+1][di]
                if np.sum(coaddlist['input_list'][i_][:,1]==-1)==4:
                    coaddlist['coadd_i'][i_] = -1
                i_+=1

        coaddlist_filename = get_filename(self.params['out_path'],
                                'truth',
                                self.params['output_meds'],
                                var='coaddlist',
                                ftype='fits.gz',
                                overwrite=True)
        coaddlist = coaddlist[coaddlist['coadd_i'] != -1]
        print(coaddlist)
        fits = fio.FITS(coaddlist_filename,'rw',clobber=True)
        fits.write(coaddlist)
        fits.close()

        coadd_from_file = np.empty((int((180/self.dd)*(360/self.dd)),2))
        coadd_from_file[:,:] = -1
        i_=0
        for i in range(len(coaddlist)):
            for j in range(4):
                if coaddlist['input_list'][i][j,1] != -1:
                    coadd_from_file[i_] = [i,j]
                    i_+=1

        coadd_from_file=coadd_from_file[coadd_from_file[:,0]>-1]
        np.savetxt(self.params['coadd_from_file'],coadd_from_file,fmt='%i')

        return 

    def get_coadd(self,i,f):
        from drizzlepac.astrodrizzle import AstroDrizzle
        from astropy.io import fits

        dither = fio.FITS(self.params['dither_file'])[-1].read()
        dither_list = np.loadtxt(self.params['dither_from_file']).astype(int)
        coaddlist_filename = get_filename(self.params['out_path'],
                                'truth',
                                self.params['output_meds'],
                                var='coaddlist',
                                ftype='fits.gz',
                                overwrite=False)
        coaddlist = fio.FITS(coaddlist_filename)[-1][i]

        tilename  = coaddlist['tilename']
        filter_ = filter_dither_dict_[f+1]
        print(filter_)

        filename = get_filename(self.params['out_path'],
                                'images/coadd',
                                self.params['output_meds'],
                                var=filter_+'_'+tilename,
                                ftype='fits.gz',
                                overwrite=True)
        filename_ = get_filename(self.params['tmpdir'],
                                '',
                                self.params['output_meds'],
                                var=filter_+'_'+tilename,
                                ftype='fits',
                                overwrite=True)

        input_list = []
        for j in coaddlist['input_list'][f]:
            if j==-1:
                break
            d = dither_list[j,0]
            sca = dither_list[j,1]
            tmp_filename = get_filename(self.params['out_path'],
                'images/final',
                self.params['output_meds'],
                var=filter_+'_'+str(int(d))+'_'+str(int(sca)),
                ftype='fits.gz',
                overwrite=False)
            if os.path.exists(tmp_filename):
                tmp_filename_ = get_filename(self.params['tmpdir'],
                    'tmp_coadd'+str(i)+'_'+str(f),
                    self.params['output_meds'],
                    var=filter_+'_'+str(int(d))+'_'+str(int(sca)),
                    ftype='fits',
                    overwrite=False)

                #if not os.path.exists(filename_[:-5] + '_flt.fits'):
                if not os.path.exists(tmp_filename_):
                    shutil.copy(tmp_filename,tmp_filename_+'.gz')
                    os.system('gunzip '+tmp_filename_+'.gz')

                input_list.append(tmp_filename_)
            else:
                raise RuntimeError("missing input file:",tmp_filename)

        if len(input_list)<1:
            return

        AstroDrizzle(list(input_list),
                     output=filename_,
                     num_cores=1,
                     runfile='',
                     context=True,
                     build=True,
                     preserve=False,
                     clean=True,
                     driz_cr=False,
                     skysub=True,
                     final_pixfrac=0.8,
                     final_outnx=self.final_nxy,
                     final_outny=self.final_nxy,
                     final_ra=coaddlist['coadd_ra'],
                     final_dec=coaddlist['coadd_dec'],
                     final_rot=0.,
                     final_scale=self.final_scale,
                     in_memory=False,
                     final_wht_type='ERR',
                     combine_type='median')

        os.system('gzip '+filename_)
        shutil.copy(filename_+'.gz',filename)
        os.remove(filename_+'.gz')
        shutil.rmtree(os.path.join(self.params['tmpdir'],'tmp_coadd'+str(i)+'_'+str(f)))

