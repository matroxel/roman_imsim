
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
from astropy.io import fits

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

good=np.array([137972, 137951, 138120, 138115, 137969, 137950, 137949, 138114,
       138113, 137968, 137947, 137948, 137943, 138112, 138027, 137957,
       137946, 137945, 137942, 137941, 138026, 138025, 137956, 137935,
       137944, 137939, 137940, 137855, 138024, 138019, 137934, 137933,
       137938, 137937, 137854, 137853, 138018, 138017, 137931, 137932,
       137927, 137936, 137851, 137852, 137847, 138016, 137995, 137930,
       137929, 137926, 137925, 137850, 137849, 137846, 137845, 137994,
       137928, 137923, 137924, 137839, 137848, 137843, 137844, 137823,
       137992, 137885, 137922, 137921, 137838, 137837, 137842, 137841,
       137822, 137821, 137879, 137920, 137835, 137836, 137831, 137840,
       137819, 137820, 137877, 137834, 137833, 137830, 137829, 137818,
       137817, 137832, 137827, 137828, 137807, 137816, 137826, 137825,
       137806])

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
        self.dsca        = 0.11*4088/60/60/2*np.sqrt(2.)
        self.stamp_size  = 32
        self.oversample_factor = 8
        self.psf_cache   = {}
        self.ra_min      = 51
        self.ra_max      = 56
        self.dec_min     = -42
        self.dec_max     = -38

        return


    def cull_input_ditherlist(self):

        dd = np.sqrt(2)*roman.n_pix*roman.pixel_scale/60./60.
        d = np.loadtxt(self.params['dither_from_file']).astype(int)
        pointings  = fio.FITS(self.params['dither_file'])[-1].read()[d[:,0]]
        max_rad_from_boresight = 0.009/np.pi*180.
        bore_mask = (pointings['ra']>self.ra_min-max_rad_from_boresight) & (pointings['ra']<self.ra_max+max_rad_from_boresight) & (pointings['dec']>self.dec_min-max_rad_from_boresight) & (pointings['dec']<self.dec_max+max_rad_from_boresight)
        d = d[bore_mask]
        print(len(d))
        plt.plot([self.ra_min,self.ra_max],[self.dec_max,self.dec_max],color='k')
        plt.plot([self.ra_min,self.ra_max],[self.dec_min,self.dec_min],color='k')
        plt.plot([self.ra_min,self.ra_min],[self.dec_min,self.dec_max],color='k')
        plt.plot([self.ra_max,self.ra_max],[self.dec_min,self.dec_max],color='k')

        self.setup_pointing()
        mask = np.ones(len(d)).astype(bool)
        for j,d_ in enumerate(d):
            print(j)
            self.update_pointing(dither=d_[0],sca=d_[1],psf=False)
            ra  = self.pointing.radec.ra/galsim.degrees
            dec = self.pointing.radec.dec/galsim.degrees
            if (ra<self.ra_min-2*dd) or (ra>self.ra_max+2*dd) or (dec<self.dec_min-2*dd) or (dec>self.dec_max+2*dd):
                mask[j] = False
            else:
                plt.plot(self.pointing.radec.ra/galsim.degrees,self.pointing.radec.dec/galsim.degrees,marker='.',ls='',color='r')
        plt.savefig('dither_list.png')
        plt.close()
        np.savetxt('ditherlist_culled.txt',d[mask])

    def verify_output_files(self,cap=None,plot_region=None):

        dd = np.sqrt(2)*roman.n_pix*roman.pixel_scale/60./60.

        d = np.loadtxt(self.params['dither_from_file']).astype(int)
        if cap is not None:
            d = d[:cap]
        pointings  = fio.FITS(self.params['dither_file'])[-1][d[:,0]]
        plt.plot([self.ra_min,self.ra_max],[self.dec_max,self.dec_max],color='k')
        plt.plot([self.ra_min,self.ra_max],[self.dec_min,self.dec_min],color='k')
        plt.plot([self.ra_min,self.ra_min],[self.dec_min,self.dec_max],color='k')
        plt.plot([self.ra_max,self.ra_max],[self.dec_min,self.dec_max],color='k')

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
        np.savetxt('missing_truth.txt',truth)
        print('........',len(truth))

        self.setup_pointing()
        #truth plot
        radec = []
        # if plot_region is not None:
        #     plt.scatter(gal['ra'][arg],gal['dec'][arg],c=pix[arg],marker='.')
        #     for i in np.unique(pix[arg]):
        #         ra,dec=hp.pix2ang(nside,i,lonlat=True,nest=True)
        #         plt.text(ra,dec,str(i),fontsize='x-small')
        for d_ in truth:
            self.update_pointing(dither=d_[0],sca=d_[1],psf=False)
            print('missing truth',j,test,d_[0],d_[1],self.pointing.radec.ra/galsim.degrees,self.pointing.radec.dec/galsim.degrees)
            plt.plot(self.pointing.radec.ra/galsim.degrees,self.pointing.radec.dec/galsim.degrees,marker='.',ls='',color='r')
            radec.append([self.pointing.radec.ra/galsim.degrees,self.pointing.radec.dec/galsim.degrees])
        plt.savefig('missing_truth.png')
        plt.close()
        radec = np.array(radec)
        np.savetxt('missing_radec.txt',radec)
        return

        # images dir
        images = []
        f = glob.glob(self.params['out_path']+'/images/'+self.params['output_meds']+'*')
        for j,d_ in enumerate(d):
            # s = '_'+str(d_[0])+'_'+str(d_[1])+'_'
            # test = [i for i in f if s in i]
            s = '_'+str(d_[0])+'_'+str(d_[1])+'.'
            test.append( [i for i in f if s in i] )
            if len(test) != 1:
                images.append(d_)
                print('missing images',j,test,d_[0],d_[1])
        images = np.array(images)
        np.savetxt('missing_images.txt',images)

        #images plot
        plt.hist2d(gal['ra'],gal['dec'],bins=500)
        for i in np.unique(pix[arg]):
            ra,dec=hp.pix2ang(nside,i,lonlat=True,nest=True)
            plt.text(ra,dec,str(i),fontsize='x-small')
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
            # s = '_'+str(d_[0])+'_'+str(d_[1])+'.'
            # test.append( [i for i in f if s in i] )
            if len(test) != 2:
                stamps.append(d_)
                # print('missing stamps',j,test,d_[0],d_[1])
        stamps = np.array(stamps)
        np.savetxt('missing_stamps.txt',stamps)

        #stamps plot
        plt.hist2d(gal['ra'],gal['dec'],bins=500)
        for i in np.unique(pix[arg]):
            ra,dec=hp.pix2ang(nside,i,lonlat=True,nest=True)
            plt.text(ra,dec,str(i),fontsize='x-small')
        for d_ in stamps:
            self.update_pointing(dither=d_[0],sca=d_[1],psf=False)
            plt.plot(self.pointing.radec.ra/galsim.degrees,self.pointing.radec.dec/galsim.degrees,marker='.',ls='',color='r')
        plt.savefig('missing_stamps.png')
        plt.close()

        return

    def plot_good(self,cap=-1):

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
        nside=128
        pix = hp.ang2pix(nside,gal['ra'],gal['dec'],lonlat=True,nest=True)
        mask = np.where(np.in1d(pix, good,assume_unique=False))[0]
        arg = np.random.choice(mask,1000000,replace=False)

        truth = np.loadtxt('missing_truth.txt')
        print('........',len(truth))

        self.setup_pointing()
        #truth plot
        radec = []
        plt.hist2d(gal['ra'],gal['dec'],bins=500)
        plt.scatter(gal['ra'][arg],gal['dec'][arg],c=pix[arg],marker='.')
        for i in np.unique(pix[arg]):
            ra,dec=hp.pix2ang(nside,i,lonlat=True,nest=True)
            plt.text(ra,dec,str(i),fontsize='x-small')
        for d_ in np.unique(d[:,0]):
            if len(truth)!=0:
                if d_ in truth[:,0]:
                    continue
            self.update_pointing(dither=d_,sca=1,psf=False)
            plt.plot(self.pointing.radec.ra/galsim.degrees,self.pointing.radec.dec/galsim.degrees,marker='.',ls='',color='r')
            radec.append([self.pointing.radec.ra/galsim.degrees,self.pointing.radec.dec/galsim.degrees])
        plt.savefig('found_truth.png')
        plt.close()

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

    def generate_limits(self):

        limits_filename = get_filename(self.params['out_path'],
                                'truth',
                                self.params['output_meds'],
                                var='radec',
                                ftype='txt',
                                overwrite=True)

        self.setup_pointing()
        dither = np.loadtxt(self.params['dither_from_file'])
        limits = np.ones((len(dither),2))*-999
        for i,(d,sca) in enumerate(dither.astype(int)):
            if i%100==0:
                print(i,d,sca)
            self.update_pointing(dither=d,sca=sca,psf=False)
            limits[i,0] = self.pointing.radec.ra/galsim.degrees
            limits[i,1] = self.pointing.radec.dec/galsim.degrees
        np.savetxt(limits_filename,limits)

    def get_psf_fits(self,i):
        b_psf = galsim.BoundsI( xmin=1,
                                ymin=1,
                                xmax=self.stamp_size*self.oversample_factor,
                                ymax=self.stamp_size*self.oversample_factor)
        self.setup_pointing()
        dither = np.loadtxt(self.params['dither_from_file'])
        print(len(np.unique(dither[:,0].astype(int))))
        d = np.unique(dither[:,0].astype(int))[i]
        psf_filename = get_filename(self.params['out_path'],
                                'psf',
                                self.params['output_meds']+'_psf',
                                var=str(d),
                                ftype='fits',
                                overwrite=True)
        for sca in range(1,19):
            self.update_pointing(dither=d,sca=sca)
            wcs = self.pointing.WCS.local( galsim.PositionI(int(roman.n_pix/2),int(roman.n_pix/2)) )
            wcs = galsim.JacobianWCS(dudx=wcs.dudx/self.oversample_factor,
                                     dudy=wcs.dudy/self.oversample_factor,
                                     dvdx=wcs.dvdx/self.oversample_factor,
                                     dvdy=wcs.dvdy/self.oversample_factor)
            st_model = galsim.DeltaFunction()
            st_model = st_model*galsim.SED(lambda x:1, 'nm', 'flambda').withFlux(1.,self.pointing.bpass)
            # flux = st_model.calculateFlux(self.pointing.bpass)
            # st_model = st_model.evaluateAtWavelength(self.pointing.bpass.effective_wavelength)
            # st_model = st_model.withFlux(flux)
            psf = self.pointing.load_psf(None)
            st_model = galsim.Convolve(st_model , psf)
            psf_stamp = galsim.Image(b_psf, wcs=wcs)
            st_model.drawImage(self.pointing.bpass,image=psf_stamp,wcs=wcs,method='no_pixel')
            hdr = fits.Header()
            psf_stamp.wcs.writeToFitsHeader(hdr,psf_stamp.bounds)
            hdr['GS_XMIN']  = hdr['GS_XMIN']
            hdr['GS_XMIN']  = hdr['GS_YMIN']
            hdr['GS_WCS']   = hdr['GS_WCS']
            if sca==1:
                fits_ = [ fits.PrimaryHDU(header=hdr) ]
            fits_.append( fits.ImageHDU(data=psf_stamp.array,header=hdr, name=str(sca)) )
        new_fits_file = fits.HDUList(fits_)
        new_fits_file.writeto(psf_filename,overwrite=True)
        os.system('gzip '+psf_filename)

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
        coaddlist = np.empty((180*5)*(360*5),dtype=[('tilename','S11'), ('coadd_i','i8'), ('coadd_j','i8'), ('coadd_ra',float), ('coadd_dec',float), ('d_dec',float), ('d_ra',float), ('input_list','i8',(4,101))])
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
                if ra[i]-self.dd_/cosdec>np.max(self.limits[:,0][self.limits[:,0]!=-999])+self.dsca:
                    continue
                if ra[i]+self.dd_/cosdec<np.min(self.limits[:,0][self.limits[:,0]!=-999])-self.dsca:
                    continue

                coaddlist['coadd_i'][i_] = i
                coaddlist['coadd_j'][i_] = j
                coaddlist['tilename'][i_] = "{:.2f}".format(ra[i])+'_'+"{:.2f}".format(dec[j])
                coaddlist['d_ra'][i_] = dra
                coaddlist['coadd_ra'][i_] = ra[i]
                coaddlist['d_dec'][i_] = self.dd
                coaddlist['coadd_dec'][i_] = dec[j]

                mask = np.where((self.limits[:,0]+self.dsca>ra[i]-self.dd_/cosdec)&(self.limits[:,0]-self.dsca<ra[i]+self.dd_/cosdec)&(self.limits[:,1]+self.dsca>dec[j]-self.dd_)&(self.limits[:,1]-self.dsca<dec[j]+self.dd_))[0]
                f = dither['filter'][dither_list[mask,0]]
                print('mask',len(mask),len(f))

                for fi in range(4):
                    for di in range(np.sum(f==fi+1)):
                        if di>100:
                            print('Cutting input file list to be less than 100 images deep.')
                            break
                        coaddlist['input_list'][i_][fi,di] = mask[f==fi+1][di]
                if np.sum(coaddlist['input_list'][i_][:,1]==-1)==4:
                    coaddlist['coadd_i'][i_] = -1
                i_+=1

        coaddlist_filename = get_filename(self.params['out_path'],
                                'truth/coadd',
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

    def check_coaddfile(self,i,f):
        dither = fio.FITS(self.params['dither_file'])[-1].read()
        dither_list = np.loadtxt(self.params['dither_from_file']).astype(int)
        coaddlist_filename = get_filename(self.params['out_path'],
                                'truth/coadd',
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
                                overwrite=False)
        
        return os.path.exists(filename)


    def get_coadd(self,i,f,noise=True):
        from drizzlepac.astrodrizzle import AstroDrizzle
        from astropy.io import fits

        dither = fio.FITS(self.params['dither_file'])[-1].read()
        dither_list = np.loadtxt(self.params['dither_from_file']).astype(int)
        coaddlist_filename = get_filename(self.params['out_path'],
                                'truth/coadd',
                                self.params['output_meds'],
                                var='coaddlist',
                                ftype='fits.gz',
                                overwrite=False)
        coaddlist = fio.FITS(coaddlist_filename)[-1][i]
        if 'sca_file_path' in self.params:
            impath = 'sca_model/'
        else:
            impath = 'simple_model/'

        tilename  = coaddlist['tilename']
        filter_ = filter_dither_dict_[f+1]
        print(filter_)


        filename = get_filename(self.params['out_path'],
                                'images/'+impath+'coadd',
                                self.params['output_meds'],
                                var=filter_+'_'+tilename,
                                ftype='fits.gz',
                                overwrite=False)
        if os.path.exists(filename):
            return
        filename_ = get_filename(self.params['tmpdir'],
                                '',
                                self.params['output_meds'],
                                var=filter_+'_'+tilename,
                                ftype='fits',
                                overwrite=True)

        filename_noise = get_filename(self.params['tmpdir'],
                                '',
                                self.params['output_meds'],
                                var=filter_+'_'+tilename+'_noise',
                                ftype='fits',
                                overwrite=True)
        input_list = []
        input_noise_list = []
        d_list = []
        sca_list = []
        for j in coaddlist['input_list'][f]:
            if j==-1:
                break
            d = dither_list[j,0]
            d_list.append(d)
            sca = dither_list[j,1]
            sca_list.append(sca)
            tmp_filename = get_filename(self.params['out_path'],
                'images/'+impath,
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
                print("missing input file:",tmp_filename)
                continue

            if noise:
                sky = galsim.fits.read(tmp_filename_,hdu=2)
                sky_mean = fio.FITS(tmp_filename_)[2].read_header()['sky_mean']
                #sky_mean = np.mean(sky.array[:,:])
                sky.array[:,:] -= sky_mean
                tmp_filename_noise = get_filename(self.params['tmpdir'],
                    'tmp_coadd'+str(i)+'_'+str(f),
                    self.params['output_meds'],
                    var=filter_+'_'+str(int(d))+'_'+str(int(sca))+'_noise',
                    ftype='fits',
                    overwrite=False)
                shutil.copy(tmp_filename_, tmp_filename_noise)
                fio.FITS(tmp_filename_noise,'rw')[1].write(sky.array)
                input_noise_list.append(tmp_filename_noise)

        sky = None
        d_list = np.array(d_list)
        sca_list = np.array(sca_list)

        if len(input_list)<1:
            return

        if len(input_list)>63:
            print('Cutting input file list to be less than 64 images deep.')
            input_list = input_list[:63]
            d_list = d_list[:63]
            sca_list = sca_list[:63]
        print(input_list)
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
                     final_pixfrac=0.7,
                     final_outnx=self.final_nxy,
                     final_outny=self.final_nxy,
                     final_ra=coaddlist['coadd_ra'],
                     final_dec=coaddlist['coadd_dec'],
                     final_rot=0.,
                     final_scale=self.final_scale,
                     in_memory=False,
                     #final_wht_type='ERR',
                     combine_type='median')

        if noise:
            if len(input_noise_list)>63:
                input_noise_list = input_noise_list[:63]
            AstroDrizzle(list(input_noise_list),
                         output=filename_noise,
                         num_cores=1,
                         runfile='',
                         context=True,
                         build=True,
                         preserve=False,
                         clean=True,
                         driz_cr=False,
                         skysub=False,
                         final_pixfrac=0.7,
                         final_outnx=self.final_nxy,
                         final_outny=self.final_nxy,
                         final_ra=coaddlist['coadd_ra'],
                         final_dec=coaddlist['coadd_dec'],
                         final_rot=0.,
                         final_scale=self.final_scale,
                         in_memory=False,
                         #final_wht_type='ERR',
                         combine_type='median')
            data, hdr = fits.getdata(filename_, 'SCI', header=True)
            data = fio.FITS(filename_noise)[1].read()
            hdr['EXTNAME']='ERR'
            fits.append(filename_,data,hdr)

        self.get_coadd_psf(filename_,filter_+'_'+tilename,d_list,sca_list)

        os.system('gzip '+filename_)
        shutil.copy(filename_+'.gz',filename)
        os.remove(filename_+'.gz')
        os.remove(filename_noise)
        shutil.rmtree(os.path.join(self.params['tmpdir'],'tmp_coadd'+str(i)+'_'+str(f)))

    def get_coadd_psf(self,filename_,filetag,d_list,sca_list):

        psf_filename = get_filename(self.params['out_path'],
                                'psf/coadd',
                                self.params['output_meds'],
                                var=filetag+'_psf',
                                ftype='fits',
                                overwrite=True)

        psf_filename_ = get_filename(self.params['tmpdir'],
                                '',
                                self.params['output_meds'],
                                var=filetag+'_psf',
                                ftype='fits',
                                overwrite=True)

        ctx = fio.FITS(filename_)['CTX'].read()
        if len(np.shape(ctx))>2:
            nplane = np.shape(ctx)[0]
        else:
            nplane = 1
        if nplane<2:
            cc = ctx.astype('uint32')
        elif nplane<3:
            cc = np.left_shift(ctx[1,:,:].astype('uint64'),32)+ctx[0,:,:].astype('uint32')
        else:
            # if nplane>2:
            #     for i in range(nplane-2):
            #         cc += np.left_shift(ctx[i+2,:,:].astype('uint64'),32*(i+2))
            print('Not designed to work with more than 64 images.')
        ctxu = np.unique(cc)

        psf_images = {}
        for d in d_list:
            tmp_filename = get_filename(self.params['out_path'],
                                        'psf',
                                        self.params['output_meds']+'_psf',
                                        var=str(int(d)),
                                        ftype='fits.gz',
                                        overwrite=False)
            psf_images[int(d)] = [galsim.InterpolatedImage(tmp_filename,hdu=sca,x_interpolant='lanczos50') for sca in range(1,19)]

        b_psf = galsim.BoundsI( xmin=1,
                                ymin=1,
                                xmax=self.stamp_size*self.oversample_factor,
                                ymax=self.stamp_size*self.oversample_factor)
        wcs = galsim.AstropyWCS(file_name=filename_,hdu=1).local(galsim.PositionI(int(self.final_nxy/2),int(self.final_nxy/2)))
        wcs = galsim.JacobianWCS(dudx=wcs.dudx/(self.oversample_factor/(roman.pixel_scale/self.final_scale)),
                                 dudy=wcs.dudy/(self.oversample_factor/(roman.pixel_scale/self.final_scale)),
                                 dvdx=wcs.dvdx/(self.oversample_factor/(roman.pixel_scale/self.final_scale)),
                                 dvdy=wcs.dvdy/(self.oversample_factor/(roman.pixel_scale/self.final_scale)))

        ctest = ctxu[0]
        for c in ctxu:
            if c==0:
                if len(ctxu)==1:
                    return
                ctest = ctxu[1]
                continue
            b = np.binary_repr(c)[::-1]
            bi = np.array([b[i] for i in range(len(b))],dtype=int)
            bi = np.pad(bi, (0, len(d_list)-len(bi)), 'constant').astype(int)
            psf_coadd = galsim.Add([psf_images[int(d)][int(sca)-1] for d,sca in zip(d_list[bi],sca_list[bi])])
            psf_stamp = galsim.Image(b_psf, wcs=wcs)
            psf_coadd.drawImage(image=psf_stamp)
            hdr = fits.Header()
            psf_stamp.wcs.writeToFitsHeader(hdr,psf_stamp.bounds)
            hdr['GS_XMIN']  = hdr['GS_XMIN']
            hdr['GS_XMIN']  = hdr['GS_YMIN']
            hdr['GS_WCS']   = hdr['GS_WCS']
            if c==ctest:
                fits_ = [ fits.PrimaryHDU(header=hdr) ]
            fits_.append( fits.ImageHDU(data=psf_stamp.array,header=hdr, name=str(c)) )
        new_fits_file = fits.HDUList(fits_)
        new_fits_file.writeto(psf_filename_,overwrite=True)
        os.system('gzip '+psf_filename_)
        shutil.copy(psf_filename_+'.gz',psf_filename)
        os.remove(psf_filename_+'.gz')


    def get_coadd_psf_stamp(self,coadd_file,coadd_psf_file,x,y,stamp_size,oversample_factor=1):

        if not hasattr(self,'psf_wcs'):
            xy = galsim.PositionD(int(self.final_nxy/2),int(self.final_nxy/2))
            wcs = galsim.AstropyWCS(file_name=coadd_file,hdu=1).local(xy)
            self.psf_wcs = galsim.JacobianWCS(dudx=wcs.dudx/oversample_factor,
                                     dudy=wcs.dudy/oversample_factor,
                                     dvdx=wcs.dvdx/oversample_factor,
                                     dvdy=wcs.dvdy/oversample_factor)

        hdr = fio.FITS(coadd_file)['CTX'].read_header()
        if hdr['NAXIS']==3:
            ctx = np.left_shift(fio.FITS(coadd_file)['CTX'][1,int(x),int(y)].astype('uint64'),32)+fio.FITS(coadd_file)['CTX'][0,int(x),int(y)].astype('uint32')
        elif hdr['NAXIS']==2:
            ctx = fio.FITS(coadd_file)['CTX'][int(x),int(y)].astype('uint32')
        else:
            # if nplane>2:
            #     for i in range(nplane-2):
            #         cc += np.left_shift(ctx[i+2,:,:].astype('uint64'),32*(i+2))
            print('Not designed to work with more than 64 images.')

        if ctx==0:
            return None
        if ctx not in self.psf_cache:
            psf_coadd = galsim.InterpolatedImage(coadd_psf_file,hdu=fio.FITS(coadd_psf_file)[str(ctx)].get_extnum(),x_interpolant='lanczos5')
            b_psf = galsim.BoundsI( xmin=1,
                            ymin=1,
                            xmax=stamp_size*oversample_factor,
                            ymax=stamp_size*oversample_factor)
            psf_stamp = galsim.Image(b_psf, wcs=self.psf_wcs)
            # psf_coadd.drawImage(image=psf_stamp,offset=xy-psf_stamp.true_center)
            psf_coadd.drawImage(image=psf_stamp)
            self.psf_cache[ctx] = psf_stamp

        return self.psf_cache[ctx]

    def get_detection(self,i,detect=True):
        from photutils import detect_threshold
        # from astropy.convolution import Gaussian2DKernel
        # from astropy.stats import gaussian_fwhm_to_sigma
        # from photutils.segmentation import detect_sources
        # from photutils.segmentation import deblend_sources
        # from photutils.segmentation import source_properties
        import sep

        def find_y_in_x(x,y):
            xs = np.argsort(x)
            y_ = np.searchsorted(x[xs], y)
            return xs[y_]

        def get_phot(data,obj,seg):

            # list of object id numbers that correspond to the segments
            seg_id = np.arange(1, len(obj)+1, dtype=np.int32)

            kronrad, krflag = sep.kron_radius(data, obj['x'], obj['y'], obj['a'], obj['b'], obj['theta'], 6.0, seg_id=seg_id, segmap=seg)
            kronrad[np.isnan(kronrad)] = 0.
            print(np.min(obj['a']),np.min(obj['b']),np.min(obj['theta']),np.min(kronrad))
            print(np.max(obj['a']),np.max(obj['b']),np.max(obj['theta']),np.max(kronrad))
            flux, fluxerr, flag = sep.sum_ellipse(data, obj['x'], obj['y'], obj['a'], obj['b'], obj['theta'], 2.5*kronrad,
                                                  subpix=1, seg_id=seg_id, segmap=seg)
            flag |= krflag  # combine flags into 'flag'

            r_min = 1.75  # minimum diameter = 3.5
            use_circle = kronrad * np.sqrt(obj['a'] * obj['b']) < r_min
            cflux, cfluxerr, cflag = sep.sum_circle(data, obj['x'][use_circle], obj['y'][use_circle],
                                                    r_min, subpix=1, seg_id=seg_id[use_circle], segmap=seg)
            flux[use_circle] = cflux
            fluxerr[use_circle] = cfluxerr
            flag[use_circle] = cflag

            r, flag_ = sep.flux_radius(data, obj['x'], obj['y'], 6.*obj['a'], 0.5, seg_id=seg_id, segmap=seg,
                      normflux=flux, subpix=5)

            sig = 2. / 2.35 * r  # r from sep.flux_radius() above, with fluxfrac = 0.5
            xwin, ywin, winflag = sep.winpos(data, obj['x'], obj['y'], sig)
            winflag |= flag_

            return kronrad, flux, fluxerr, flag, xwin, ywin, winflag

        dither = fio.FITS(self.params['dither_file'])[-1].read()
        dither_list = np.loadtxt(self.params['dither_from_file']).astype(int)
        coaddlist_filename = get_filename(self.params['out_path'],
                                'truth/coadd',
                                self.params['output_meds'],
                                var='coaddlist',
                                ftype='fits.gz',
                                overwrite=False)
        coaddlist_ = fio.FITS(coaddlist_filename)[-1]

        coaddlist = coaddlist_[i]
        tilename  = coaddlist['tilename']
        filename_ = get_filename(self.params['out_path'],
                                'truth/coadd',
                                self.params['output_meds'],
                                var='index'+'_'+tilename,
                                ftype='fits.gz',
                                overwrite=True)
        fgal  = fio.FITS(filename_,'rw',clobber=True)
        start_row = 0
        length_gal = 1000000
        gal = None
        coadd_imgs = []
        for f in range(4):
            filter_ = filter_dither_dict_[f+1]
            if detect:
                coaddfilename = get_filename(self.params['out_path'],
                            'images/coadd',
                            self.params['output_meds'],
                            var=filter_+'_'+tilename,
                            ftype='fits.gz',
                            overwrite=False)
                coadd_imgs.append( fio.FITS(coaddfilename)['SCI'].read() )
            for j in coaddlist['input_list'][f]:
                if j==-1:
                    break
                d = dither_list[j,0]
                sca = dither_list[j,1]
                # if i%100==0:
                #     print(i,j,d,sca,start_row)
                filename = get_filename(self.params['out_path'],
                                        'truth',
                                        self.params['output_meds'],
                                        var='index',
                                        name2=filter_+'_'+str(d)+'_'+str(sca),
                                        ftype='fits',
                                        overwrite=False)
                filename_star = get_filename(self.params['out_path'],
                                        'truth',
                                        self.params['output_meds'],
                                        var='index',
                                        name2=filter_+'_'+str(d)+'_'+str(sca)+'_star',
                                        ftype='fits',
                                        overwrite=False)
                print(filename)

                try:
                    tmp = fio.FITS(filename)[-1].read()
                except:
                    print('failed to open',filename)
                    continue
                tmp['ra'] *= 180./np.pi
                tmp['dec'] *= 180./np.pi
                mask = (tmp['ra']>coaddlist['coadd_ra']-coaddlist['d_ra'])&(tmp['ra']<coaddlist['coadd_ra']+coaddlist['d_ra'])&(tmp['dec']>coaddlist['coadd_dec']-coaddlist['d_dec'])&(tmp['dec']<coaddlist['coadd_dec']+coaddlist['d_dec'])
                tmp = tmp[mask]
                if len(tmp)==0:
                    continue
                if start_row==0:
                    gal = np.zeros(length_gal,dtype=np.dtype([('ind', 'i8'), ('sca', 'i8'), ('dither', 'i8'), ('x', 'f8'), ('y', 'f8'), ('ra', 'f8'), ('dec', 'f8'), ('mag', 'f8', (4,)), ('stamp', 'i8'), ('start_row', 'i8'), ('gal_star', 'i2')]))
                    gal['ind'] = -1
                    gal['gal_star'] = -1
                    gal['x']=-1
                    gal['y']=-1
                mask = ~np.in1d(tmp['ind'],gal['ind'][:start_row][gal['gal_star'][:start_row]==0],assume_unique=True)
                for col in ['ind','sca','dither','ra','dec','mag','stamp']:
                    if col=='mag':
                        gal[col][start_row:start_row+np.sum(mask),f] = tmp[col][mask]
                    else:
                        gal[col][start_row:start_row+np.sum(mask)] = tmp[col][mask]
                gal['gal_star'][start_row:start_row+np.sum(mask)] = 0
                start_row+=np.sum(mask)
                gal[:start_row] = gal[:start_row][np.argsort(gal[:start_row]['ind'])]
                if np.sum(~mask)>0:
                    gmask = np.where(np.in1d(gal['ind'][:start_row],tmp['ind'][~mask],assume_unique=False))[0]
                    gmask = gmask[gal['gal_star'][gmask]==0]
                    gal['mag'][gmask,f] = tmp['mag'][~mask]

                try:
                    tmp = fio.FITS(filename_star)[-1].read()
                except:
                    print('failed to open',filename)
                    continue
                tmp['ra'] *= 180./np.pi
                tmp['dec'] *= 180./np.pi
                mask = (tmp['ra']>coaddlist['coadd_ra']-coaddlist['d_ra'])&(tmp['ra']<coaddlist['coadd_ra']+coaddlist['d_ra'])&(tmp['dec']>coaddlist['coadd_dec']-coaddlist['d_dec'])&(tmp['dec']<coaddlist['coadd_dec']+coaddlist['d_dec'])
                tmp = tmp[mask]
                if len(tmp)==0:
                    continue
                mask = ~np.in1d(tmp['ind'],gal['ind'][:start_row][gal['gal_star'][:start_row]==1],assume_unique=True)
                for col in ['ind','sca','dither','ra','dec','mag','stamp']:
                    if col=='mag':
                        gal[col][start_row:start_row+np.sum(mask),f] = tmp[col][mask]
                    else:
                        gal[col][start_row:start_row+np.sum(mask)] = tmp[col][mask]
                gal['gal_star'][start_row:start_row+np.sum(mask)] = 1
                start_row+=np.sum(mask)
                gal[:start_row] = gal[:start_row][np.argsort(gal[:start_row]['ind'])]
                if np.sum(~mask)>0:
                    gmask = np.where(np.in1d(gal['ind'][:start_row],tmp['ind'][~mask],assume_unique=False))[0]
                    gmask = gmask[gal['gal_star'][gmask]==1]
                    gal['mag'][gmask,f] = tmp['mag'][~mask]

        if gal is None:
            fgal.close()
            os.remove(filename_)
            return

        gal = gal[gal['ind']!=-1]
        if len(gal)==0:
            return

        filename = get_filename(self.params['out_path'],
                                'images/coadd',
                                self.params['output_meds'],
                                var='H158'+'_'+tilename,
                                ftype='fits.gz',
                                overwrite=False)

        if detect:
            wcs = galsim.AstropyWCS(file_name=filename,hdu=1)
            for i in range(len(gal)):
                xy = wcs.toImage(galsim.CelestialCoord(gal[i]['ra']*galsim.degrees, gal[i]['dec']*galsim.degrees))
                # print(xy.x,xy.y,gal[i]['ra'],gal[i]['dec'])
                gal['x'][i]    = xy.x-1
                gal['y'][i]    = xy.y-1
        gal['stamp']   *= 2
        gal = np.sort(gal,order=['ind'])
        fgal.write(gal)
        gal = None
        fgal.close()

        if not detect:
            return

        data = np.nanmedian(np.stack(coadd_imgs),axis=0)
        threshold = detect_threshold(data, nsigma=2.)


        # sigma = 5.0 * gaussian_fwhm_to_sigma
        # kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
        # kernel.normalize()
        # segm = detect_sources(data, threshold, npixels=5, filter_kernel=kernel)
        # segm_deblend = deblend_sources(data, segm, npixels=5, filter_kernel=kernel,
        #                                nlevels=32, contrast=0.05)
        # cat = source_properties(data, segm_deblend,kron_params=('correct', 2.5, 0.0, 'exact', 5))
        # tbl = cat.to_table(columns=['id','xcentroid','ycentroid','kron_flux','kron_fluxerr'])
        # tbl.rename_columns( ('xcentroid','ycentroid'), ('x','y'))


        obj,seg = sep.extract(data,threshold[0,0],segmentation_map=True)
        out = np.zeros(len(obj),np.dtype([('x', 'f8'), ('y', 'f8'),('x_win', 'f8'), ('y_win', 'f8'), ('ra', 'f8'), ('dec', 'f8'),('ra_win', 'f8'), ('dec_win', 'f8'), ('a', 'f8'), ('b', 'f8'), ('theta', 'f8'), ('fluxauto_Y106', 'f8'), ('fluxauto_J129', 'f8'), ('fluxauto_H158', 'f8'), ('fluxauto_F184', 'f8'), ('magauto_Y106', 'f8'), ('magauto_J129', 'f8'), ('magauto_H158', 'f8'), ('magauto_F184', 'f8'), ('fluxauto_Y106_err', 'f8'), ('fluxauto_J129_err', 'f8'), ('fluxauto_H158_err', 'f8'), ('fluxauto_F184_err', 'f8'), ('kronrad_Y106', 'f8'), ('kronrad_J129', 'f8'), ('kronrad_H158', 'f8'), ('kronrad_F184', 'f8'), ('flag', 'i8'), ('flag_win', 'i8'), ('flag_phot_Y106', 'i8'), ('flag_phot_J129', 'i8'), ('flag_phot_H158', 'i8'), ('flag_phot_F184', 'i8')]))

        for col in ['x','y','a','b','theta','flag']:
            out[col] = obj[col]
        kronrad, flux, fluxerr, flag, xwin, ywin, winflag = get_phot(data, obj, seg)
        out['x_win'] = xwin
        out['y_win'] = ywin
        out['flag_win'] = winflag
        for i in range(len(out)):
            radec = wcs.toWorld(galsim.PositionD(out['x'][i]+1,out['y'][i]+1))
            out['ra'][i]    = radec.ra/galsim.degrees
            out['dec'][i]   = radec.dec/galsim.degrees
            radec = wcs.toWorld(galsim.PositionD(out['x_win'][i]+1,out['y_win'][i]+1))
            out['ra_win'][i]    = radec.ra/galsim.degrees
            out['dec_win'][i]   = radec.dec/galsim.degrees

        for i in range(4):
            filter_ = filter_dither_dict_[i+1]
            kronrad, flux, fluxerr, flag, xwin, ywin, winflag = get_phot(coadd_imgs[i], obj, seg)
            print(filter_,flux,fluxerr)
            out['fluxauto_'+filter_]        = flux
            out['fluxauto_'+filter_+'_err'] = fluxerr
            out['kronrad_'+filter_]         = kronrad
            out['flag_phot_'+filter_]       = flag
            out['magauto_'+filter_]         = -2.5*np.log10(flux)-16.8008709162+48.6
            out['magauto_'+filter_][np.isnan(out['magauto_'+filter_])] = 99.


        filename = get_filename(self.params['out_path'],
                                'detection',
                                self.params['output_meds'],
                                var='det'+'_'+tilename,
                                ftype='fits.gz',
                                overwrite=True)
        fio.write(filename,out,clobber=True)
        filename = get_filename(self.params['out_path'],
                                'detection',
                                self.params['output_meds'],
                                var='seg'+'_'+tilename,
                                ftype='fits.gz',
                                overwrite=True)
        fio.write(filename,seg,clobber=True)


    def accumulate_index(self):

        coadd_list = np.loadtxt(self.params['coadd_from_file']).astype(int)
        coaddlist_filename = get_filename(self.params['out_path'],
                                'truth/coadd',
                                self.params['output_meds'],
                                var='coaddlist',
                                ftype='fits.gz',
                                overwrite=False)
        coaddlist = fio.FITS(coaddlist_filename)[-1].read()

        filename = get_filename(self.params['out_path'],
                                'truth',
                                self.params['output_truth'],
                                name2='truth_gal',
                                overwrite=False)
        length = fio.FITS(filename)[-1].read_header()['NAXIS2']
        length += fio.FITS(self.params['star_sample'])[-1].read_header()['NAXIS2']
        start_row = 0
        gal = np.zeros(length,dtype=np.dtype([('ind', 'i8'), ('sca', 'i8'), ('dither', 'i8'), ('x', 'f8'), ('y', 'f8'), ('ra', 'f8'), ('dec', 'f8'), ('mag', 'f8', (4,)), ('stamp', 'i8'), ('start_row', 'i8'), ('gal_star', 'i2'), ('tilename', 'S12')]))
        for i in range(len(np.unique(coadd_list[:,0]))):
            tilename  = coaddlist[i]['tilename']
            filename = get_filename(self.params['out_path'],
                                    'truth/coadd',
                                    self.params['output_meds'],
                                    var='index'+'_'+tilename,
                                    ftype='fits.gz',
                                    overwrite=False)
            try:
                tmp = fio.FITS(filename)[-1].read()
            except:
                print('failed to open '+filename)
                continue
            for col in tmp.dtype.names:
                gal[col][start_row:start_row+len(tmp)] = tmp[col]
            gal['tilename'][start_row:start_row+len(tmp)] = tilename
            start_row+=len(tmp)

        gal=gal[gal['ind']!=0]
        gal=gal[np.argsort(gal['ind'])]
        filename = get_filename(self.params['out_path'],
                                'truth/coadd',
                                self.params['output_meds'],
                                var='full_index',
                                ftype='fits',
                                overwrite=True)
        print(filename,gal)
        fio.write(filename,gal,clobber=True)
        os.system('gzip '+filename)
