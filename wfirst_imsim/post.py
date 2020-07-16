import numpy as np
import fitsio as fio
import os
import yaml

from wfirst_imsim.misc import get_filename

filter_dither_dict_ = {
    3:'J129',
    1:'F184',
    4:'Y106',
    2:'H158'
}

params = yaml.load(open('dc2.yaml'))
dither = np.loadtxt(params['dither_from_file'])

# for d in dither:
#     print(int(d))
#     f = filter_dither_dict_[fio.FITS(params['dither_file'])[-1][int(d)]['filter']]
#     filename_ = get_filename(params['out_path'],
#                             'images/visits',
#                             params['output_meds'],
#                             var=f+'_'+str(int(d)),
#                             ftype='fits',
#                             overwrite=True)
#     if os.path.exists(filename_+'.gz'):
#         continue
#     out = fio.FITS(filename_,'rw')
#     for sca in range(1,19):
#         filename = get_filename(params['out_path'],
#                                 'images',
#                                 params['output_meds'],
#                                 var=f+'_'+str(int(d)),
#                                 name2=str(sca),
#                                 ftype='fits.gz',
#                                 overwrite=False)
#         if not os.path.exists(filename):
#             continue
#         filename = filename[:-3]
#         tmp = fio.FITS(filename,'r')[-1]
#         hdr = tmp.read_header()
#         hdr['extname'] = 'sca_'+str(sca)
#         data = tmp.read()
#         out.write(data,header=hdr)
#     out.close()
#     os.system('gzip '+filename_)

##########

# filename_ = get_filename(params['out_path'],
#                         'truth',
#                         params['output_meds'],
#                         var='index',
#                         ftype='fits',
#                         overwrite=True)
# filename_star_ = get_filename(params['out_path'],
#                         'truth',
#                         params['output_meds'],
#                         var='index_star',
#                         ftype='fits',
#                         overwrite=True)
# start=True
# gal_i = 0
# star_i = 0
# for d in dither:
#     print(int(d))
#     f = filter_dither_dict_[fio.FITS(params['dither_file'])[-1][int(d)]['filter']]
#     for sca in range(1,19):
#         filename = get_filename(params['out_path'],
#                                 'truth',
#                                 params['output_meds'],
#                                 var='index',
#                                 name2=f+'_'+str(int(d))+'_'+str(sca),
#                                 ftype='fits',
#                                 overwrite=False)
#         filename_star = get_filename(params['out_path'],
#                                 'truth',
#                                 params['output_meds'],
#                                 var='index',
#                                 name2=f+'_'+str(int(d))+'_'+str(sca)+'_star',
#                                 ftype='fits',
#                                 overwrite=False)
#         if start:
#             gal = fio.FITS(filename)[-1].read()
#             star = fio.FITS(filename_star)[-1].read()
#             start=False
#             gal = np.empty(100000000,dtype=gal.dtype)
#             star = np.empty(100000000,dtype=star[['ind','sca','dither','x','y','ra','dec','mag']].dtype)
#         tmp = fio.FITS(filename)[-1].read()
#         for name in gal.dtype.names:
#             gal[name][gal_i:gal_i+len(tmp)] = tmp[name]
#         gal_i+=len(tmp)
#         tmp = fio.FITS(filename_star)[-1].read()
#         for name in star.dtype.names:
#             star[name][star_i:star_i+len(tmp)] = tmp[name]
#         star_i+=len(tmp)
# gal=gal[gal['ind']!=0]
# star=star[star['ind']!=0]
# gal = np.sort(gal,order=['ind','dither','sca'])
# star = np.sort(star,order=['ind','dither','sca'])
# gal = gal[(gal['x']<4088+256)&(gal['x']>-256)&(gal['y']<4088+256)&(gal['y']>-256)]
# star = star[(star['x']<4088+256)&(star['x']>-256)&(star['y']<4088+256)&(star['y']>-256)]
# fio.write(filename_,gal)
# f = fio.FITS(filename_[:-3],'rw')
# f.write(gal)
# f.close()
# f = fio.FITS(filename_star_[:],'rw')
# f.write(star)
# f.close()

##########

def write_fits(filename,img):

    hdr={}
    img.wcs.writeToFitsHeader(hdr,img.bounds)
    hdr['GS_XMIN'] = hdr['GS_XMIN'][0]
    hdr['GS_XMIN'] = hdr['GS_YMIN'][0]
    hdr['GS_WCS']  = hdr['GS_WCS'][0]
    hdr['extname']  = 'sca_'+str(sca)
    fits = fio.FITS(filename,'rw')
    fits.write(img.array)
    fits[0].write_keys(hdr)
    fits.close()

    return

import galsim
import wfirst_imsim
sim = wfirst_imsim.wfirst_sim('dc2_local.yaml')
for filter_ in ['Y106','J129','H158','F184']:
    low = fio.FITS('PSF_model_'+filter_+'_low.fits','rw')
    high = fio.FITS('PSF_model_'+filter_+'_high.fits','rw')
    for sca in range(1,19):
        sim.setup(filter_,15459,sca=sca)
        for star in [False,True]:
            psf_stamp = galsim.Image(256,256, wcs=sim.pointing.WCS)
            st_model = galsim.DeltaFunction(flux=1.)
            gsparams = galsim.GSParams( maximum_fft_size=16384 )
            st_model = st_model.evaluateAtWavelength(sim.pointing.bpass.effective_wavelength)
            st_model  = st_model.withFlux(1.)
            if star:
                psf = sim.pointing.load_psf(None,star=True)
                psf = psf.withGSParams(galsim.GSParams(folding_threshold=1e-3))
                st_model = galsim.Convolve(st_model, psf)
            else:
                psf = sim.pointing.load_psf(None,star=False)
                st_model = galsim.Convolve(st_model, psf)
            st_model.drawImage(image=psf_stamp,wcs=sim.pointing.WCS)
            hdr={}
            psf_stamp.wcs.writeToFitsHeader(hdr,psf_stamp.bounds)
            hdr['extname']  = 'sca_'+str(sca)
            if star:
                high.write(psf_stamp.array,header=hdr)
            else:
                low.write(psf_stamp.array,header=hdr)

