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

for d in dither:
    print(d)
    f = filter_dither_dict_[fio.FITS(params['dither_file'])[-1][d]['filter']]
    filename_ = get_filename(params['out_path'],
                            'images/visits',
                            params['output_meds'],
                            var=f+'_'+str(d),
                            ftype='fits',
                            overwrite=True)
    out = fio.FITS(filename_,'rw')
    for sca in range(1,19):
        filename = get_filename(params['out_path'],
                                'images',
                                params['output_meds'],
                                var=f+'_'+str(d),
                                name2=str(sca),
                                ftype='fits.gz',
                                overwrite=False)
        if not os.path.exists(filename):
            continue
        filename = filename[:-3]
        tmp = fio.FITS(filename,'r')[-1]
        hdr = tmp.read_header()
        hdr['extname'] = 'sca_'+str(sca)
        data = tmp.read()
        out.write(data,header=hdr)
    out.close()
    os.system('gzip '+filename_)
