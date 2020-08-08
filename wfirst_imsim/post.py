
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

from .sim import wfirst_sim 
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

# Converts galsim WFIRST filter names to indices in Chris' dither file.
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

class postprocessing(wfirst_sim):
    """
    WFIRST image simulation postprocssing functions.

    Input:
    param_file : File path for input yaml config file or yaml dict. Example located at: ./example.yaml.
    """

    def __init__(self, param_file):
        super().__init__(param_file)

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

    def update_pointing(self,dither=None,sca=None):

        if dither is not None:
            # This updates the dither
            self.pointing.update_dither(dither,force_filter=True)
        if sca is not None:
            # This sets up a specific pointing for this SCA (things like WCS, PSF)
            self.pointing.update_sca(sca)

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
        sky_level = wfirst.getSkyLevel(self.pointing.bpass, world_pos=self.pointing.radec, date=self.pointing.date)
        sky_level *= (1.0 + wfirst.stray_light_fraction)
        b   = galsim.BoundsI(  xmin=1,
                                ymin=1,
                                xmax=wfirst.n_pix,
                                ymax=wfirst.n_pix)
        xy = self.pointing.WCS.toImage(self.pointing.radec)
        local_wcs = self.pointing.WCS.local(xy)
        sky_stamp = galsim.Image(bounds=b, wcs=local_wcs)
        local_wcs.makeSkyImage(sky_stamp, sky_level)
        sky_stamp += wfirst.thermal_backgrounds[self.pointing.filter]*wfirst.exptime
        noise = galsim.PoissonNoise(galsim.BaseDeviate(1))
        sky_stamp.addNoise(noise)
        sky_stamp.quantize()
        dark_current_ = wfirst.dark_current*wfirst.exptime
        sky_stamp = (sky_stamp + round(dark_current_))
        sky_stamp = sky_stamp/wfirst.gain
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
                                ftype='fits',
                                overwrite=True)
        filename_star_ = get_filename(self.params['out_path'],
                                'truth',
                                self.params['output_meds'],
                                var='index_star',
                                ftype='fits',
                                overwrite=True)
        start=True
        gal_i = 0
        star_i = 0
        for d in dither:
            print(int(d))
            f = filter_dither_dict_[fio.FITS(self.params['dither_file'])[-1][int(d)]['filter']]
            for sca in range(1,19):
                filename = get_filename(self.params['out_path'],
                                        'truth',
                                        self.params['output_meds'],
                                        var='index',
                                        name2=f+'_'+str(int(d))+'_'+str(sca),
                                        ftype='fits',
                                        overwrite=False)
                filename_star = get_filename(self.params['out_path'],
                                        'truth',
                                        self.params['output_meds'],
                                        var='index',
                                        name2=f+'_'+str(int(d))+'_'+str(sca)+'_star',
                                        ftype='fits',
                                        overwrite=False)
                if start:
                    gal = fio.FITS(filename)[-1].read()
                    star = fio.FITS(filename_star)[-1].read()
                    start=False
                    gal = np.empty(100000000,dtype=gal.dtype)
                    star = np.empty(100000000,dtype=star[['ind','sca','dither','x','y','ra','dec','mag']].dtype)
                tmp = fio.FITS(filename)[-1].read()
                for name in gal.dtype.names:
                    gal[name][gal_i:gal_i+len(tmp)] = tmp[name]
                gal_i+=len(tmp)
                tmp = fio.FITS(filename_star)[-1].read()
                for name in star.dtype.names:
                    star[name][star_i:star_i+len(tmp)] = tmp[name]
                star_i+=len(tmp)
        gal=gal[gal['ind']!=0]
        star=star[star['ind']!=0]
        gal = np.sort(gal,order=['ind','dither','sca'])
        star = np.sort(star,order=['ind','dither','sca'])
        gal = gal[(gal['x']<4088+256)&(gal['x']>-256)&(gal['y']<4088+256)&(gal['y']>-256)]
        star = star[(star['x']<4088+256)&(star['x']>-256)&(star['y']<4088+256)&(star['y']>-256)]
        fio.write(filename_,gal)
        f = fio.FITS(filename_[:-3],'rw')
        f.write(gal)
        f.close()
        f = fio.FITS(filename_star_[:],'rw')
        f.write(star)
        f.close()

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

    def load_index(self):

        index_filename = get_filename(self.params['out_path'],
                                'truth',
                                self.params['output_meds'],
                                var='index',
                                ftype='fits',
                                overwrite=False)

        indexfile = fio.FITS(index_filename)[-1].read()
        indexfile = indexfile[np.argsort(indexfile['dither'])]
        self.dither_list = np.unique(indexfile['dither'])
        dithers = np.append(0,np.where(np.diff(indexfile['dither'])!=0)[0])
        dithers = np.append(dithers,len(indexfile))

        limits = np.zeros((len(dithers)-1,18,4))
        for d in range(len(dithers)-1):
            for sca in range(18):
                tmp = indexfile[dithers[d]:dithers[d+1]]
                mask = np.where(tmp['sca']==sca+1)
                limits[d,sca,0] = np.min(tmp[mask]['ra']) * 180. / np.pi
                limits[d,sca,1] = np.max(tmp[mask]['ra']) * 180. / np.pi
                limits[d,sca,2] = np.min(tmp[mask]['dec']) * 180. / np.pi
                limits[d,sca,3] = np.max(tmp[mask]['dec']) * 180. / np.pi

        self.limits = limits

        return 

    def get_coadd(self):
        from drizzlepac.astrodrizzle import AstroDrizzle
        from astropy.io import fits

        dither = fio.FITS(self.params['dither_file'])[-1].read()

        ra  = np.zeros(360*2)+np.arange(360*2)*0.5+.25
        ra  = ra[(ra<np.max(self.limits[:,:,1])+.5)&(ra>np.min(self.limits[:,:,0])-.5)]
        dec = np.zeros(180*2)+np.arange(180*2)*0.5-90+.25
        dec = dec[(dec<np.max(self.limits[:,:,3])-.5)&(dec>np.min(self.limits[:,:,2])-.5)]
        dd  = 17000*.11/60/60/2
        for i in range(len(ra)):
            for j in range(len(dec)):
                tile_name = "{:.2f}".format(ra[i])+'_'+"{:.2f}".format(dec[i])
                ra_min  = (ra[i]-dd)# * np.pi / 180.
                ra_max  = (ra[i]+dd)# * np.pi / 180.
                dec_min = (dec[j]-dd)# * np.pi / 180.
                dec_max = (dec[j]+dd)# * np.pi / 180.

                mask = np.where((self.limits[:,:,1]+0.1>ra_min)&(self.limits[:,:,0]-0.1<ra_max)&(self.limits[:,:,3]+0.1>dec_min)&(self.limits[:,:,2]-0.1>dec_min))

                input_list = []
                filter_list = []
                for ii in range(len(mask[0])):
                    d = self.dither_list[mask[0][ii]]
                    sca = mask[1][ii]+1
                    f = filter_dither_dict_[dither['filter'][d]]
                    filename_2 = get_filename(self.params['out_path'],
                        'images',
                        self.params['output_meds'],
                        var=f+'_'+str(int(d))+'_'+str(int(sca)),
                        ftype='fits.gz',
                        overwrite=False)
                    if os.path.exists(filename_2):
                        filename_ = get_filename(self.params['tmpdir'],
                            '',
                            self.params['output_meds'],
                            var=f+'_'+str(int(d))+'_'+str(int(sca)),
                            ftype='fits',
                            overwrite=False)
                        if not os.path.exists(filename_[:-5] + '_flt.fits'):
                            shutil.copy(filename_2,filename_+'.gz')
                            os.system('gunzip '+filename_+'.gz')

                            data_temp = fits.open(filename_)
                            old_header  = data_temp[0].header
                            data = data_temp[0].data
                            data_temp.close()
                            new_header = self.prep_new_header(old_header)
                            fit0 = fits.PrimaryHDU(header=new_header)
                            fit1 = fits.ImageHDU(data=data,header=new_header, name='SCI')
                            fit2 = fits.ImageHDU(data=np.ones_like(data),header=new_header, name='ERR')
                            fit3 = fits.ImageHDU(data=np.zeros_like(data,dtype='int16'),header=new_header, name='DQ')
                            new_fits_file = fits.HDUList([fit0,fit1,fit2,fit3])
                            new_fits_file.writeto(filename_[:-5] + '_flt.fits',overwrite=True)
                            os.remove(filename_)
                        input_list.append(filename_[:-5] + '_flt.fits')
                        filter_list.append(f)
                input_list = np.array(input_list)
                filter_list = np.array(filter_list)

                print(ra[i],dec[j],input_list,filter_list)
                if len(input_list)==0:
                    continue

                for filter_ in ['Y106','J129','H158','F184']:
                    filename = get_filename(self.params['out_path'],
                                            'images/coadd',
                                            self.params['output_meds'],
                                            var=filter_+'_'+tile_name,
                                            ftype='fits',
                                            overwrite=False)

                    if os.path.exists(filename):
                        continue

                    mask_ = np.where(filter_list==filter_)[0]
                    if len(input_list[mask_])==0:
                        continue
                    AstroDrizzle(list(input_list[mask_]),
                                 output=filename,
                                 num_cores=1,
                                 runfile='',
                                 context=True,
                                 build=True,
                                 preserve=False,
                                 clean=True,
                                 driz_cr=False,
                                 skysub=False,
                                 final_pixfrac=0.8,
                                 final_outnx=17000,
                                 final_outny=17000,
                                 final_ra=ra[i],
                                 final_dec=dec[j],
                                 final_rot=0.,
                                 final_scale=0.055,
                                 in_memory=True,
                                 combine_type='median')

                    # os.system('gzip '+filename)

                # filename = get_filename(self.params['out_path'],
                #                         'images/coadd',
                #                         self.params['output_meds'],
                #                         var='det'+'_'+tile_name,
                #                         ftype='fits',
                #                         overwrite=True)
                # AstroDrizzle(list(input_list),
                #              output=filename,
                #              num_cores=1,
                #              runfile='',
                #              context=True,
                #              build=True,
                #              preserve=False,
                #              clean=True,
                #              driz_cr=False,
                #              skysub=False,
                #              final_pixfrac=0.8,
                #              final_outnx=17000,
                #              final_outny=17000,
                #              final_ra=ra[i],
                #              final_dec=dec[j],
                #              final_rot=0.,
                #              final_scale=0.055,
                #              in_memory=True,
                #              combine_type='median')

                # os.system('gzip '+filename)
                for f in input_list:
                    os.remove(f)
