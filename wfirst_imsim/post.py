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
start=True
for d in dither:
    print(int(d))
    f = filter_dither_dict_[fio.FITS(params['dither_file'])[-1][int(d)]['filter']]
    for sca in range(1,19):
        filename = get_filename(params['out_path'],
                                'truth',
                                params['output_meds'],
                                var='index',
                                name2=f+'_'+str(int(d))+'_'+str(sca),
                                ftype='fits',
                                overwrite=False)
        filename_star = get_filename(params['out_path'],
                                'truth',
                                params['output_meds'],
                                var='index',
                                name2=f+'_'+str(int(d))+'_'+str(sca)+'_star',
                                ftype='fits',
                                overwrite=False)
        if start:
            gal = fio.FITS(filename)[-1].read()
            star = fio.FITS(filename_star)[-1].read()
            start=False
        else:
            gal = np.append(gal,fio.FITS(filename)[-1].read())
            star = np.append(star,fio.FITS(filename_star)[-1].read())
gal = np.sort(gal,order=['ind','dither','sca'])
star = np.sort(star,order=['ind','dither','sca'])
f = fio.FITS(filename_,'rw')
f.write(gal)
f.close()
f = fio.FITS(filename_star_,'rw')
f.write(star)
f.close()
