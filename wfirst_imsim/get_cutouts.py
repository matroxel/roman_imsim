
from re import L
import fitsio as fio
import numpy as np
import galsim
import os, sys
import pickle

def get_coadd_psf_stamp(coadd_file,coadd_psf_file,x,y,stamp_size,oversample_factor=1):

    xy = galsim.PositionD(x,y)
    hdr = fio.FITS(coadd_file)['CTX'].read_header()
    if hdr['NAXIS']==3:
        nplane = 2
    else:
        nplane = 1
    if nplane<2:
        ctx = fio.FITS(coadd_file)['CTX'][int(x),int(y)].astype('uint32')
    elif nplane<3:
        ctx = np.left_shift(fio.FITS(coadd_file)['CTX'][1,int(x),int(y)].astype('uint64'),32)+fio.FITS(coadd_file)['CTX'][0,int(x),int(y)].astype('uint32')
    else:
        # if nplane>2:
        #     for i in range(nplane-2):
        #         cc += np.left_shift(ctx[i+2,:,:].astype('uint64'),32*(i+2))
        print('Not designed to work with more than 64 images.')
    print(ctx)
    hdu_ = fio.FITS(coadd_psf_file)[str(ctx)].get_extnum()
    psf_coadd = galsim.InterpolatedImage(coadd_psf_file,hdu=hdu_,x_interpolant='lanczos5')
    b_psf = galsim.BoundsI( xmin=1,
                    ymin=1,
                    xmax=stamp_size*oversample_factor,
                    ymax=stamp_size*oversample_factor)
    wcs = galsim.AstropyWCS(file_name=coadd_file,hdu=1).local(xy)
    wcs = galsim.JacobianWCS(dudx=wcs.dudx/oversample_factor,
                                dudy=wcs.dudy/oversample_factor,
                                dvdx=wcs.dvdx/oversample_factor,
                                dvdy=wcs.dvdy/oversample_factor)
    psf_stamp = galsim.Image(b_psf, wcs=wcs)
    # psf_coadd.drawImage(image=psf_stamp,offset=xy-psf_stamp.true_center)
    psf_coadd.drawImage(image=psf_stamp)

    return psf_coadd

def main(argv):
    base = sys.argv[1]
    filter_ = sys.argv[2]
    simset = sys.argv[3]
    work_filter = os.path.join(base, 'roman_'+filter_)
    work_truth = os.path.join(work_filter, simset+'/truth')
    work_coadd = os.path.join(work_filter, simset+'/images/coadd')
    work_psf = os.path.join(work_filter, simset+'/psf/coadd')

    truth_galaxies = fio.read(os.path.join(work_truth, 'fiducial_lensing_galaxia_'+simset+'_truth_gal.fits'))
    truth_simulated = fio.read(os.path.join(work_truth, 'fiducial_'+filter_+'_index_sorted.fits.gz'))
    truth_unique_objects = truth_simulated[truth_simulated['dither'] == -1]
    coadd_list = fio.read(os.path.join(work_truth, 'fiducial_coaddlist.fits.gz'))
    for tilename in ['26.96_-26.8']: # coadd_list['tilename']:

        if not os.path.exists(os.path.join(work_coadd, 'fiducial_H158_'+tilename+'.fits.gz')):
            continue

        out_fname = os.path.join(work_coadd, 'coadd_cutouts/fiducial_'+filter_+'_'+tilename+'_cutouts.pickle')

        ra_cen = coadd_list[coadd_list['tilename'] == tilename]['coadd_ra']
        dec_cen = coadd_list[coadd_list['tilename'] == tilename]['coadd_dec']
        ra_d = coadd_list[coadd_list['tilename'] == tilename]['d_ra']
        dec_d = coadd_list[coadd_list['tilename'] == tilename]['d_dec']
        radec_limit = [ra_cen - ra_d, ra_cen + ra_d, dec_cen - dec_d, dec_cen + dec_d]
        mask_objects = ((truth_unique_objects['ra'] >= radec_limit[0]) & (truth_unique_objects['ra'] <= radec_limit[1])
                        & (truth_unique_objects['dec'] >= radec_limit[2]) & (truth_unique_objects['dec'] <= radec_limit[3]))
        potential_coadd_objects = truth_unique_objects[mask_objects]


        coadd_fname = os.path.join(work_coadd, 'fiducial_H158_'+tilename+'.fits.gz')
        coadd_psf_fname = os.path.join(work_psf, 'fiducial_H158_'+tilename+'_psf.fits')
        coadd = fio.FITS(coadd_fname)
        image_info = coadd['SCI'].read()
        # weight_info = coadd['WHT'].read()
        noise_info = coadd['ERR'].read()
        wcs = galsim.AstropyWCS(file_name=coadd_fname, hdu=1)
        output = {}
        print('Getting ', len(potential_coadd_objects), 'cutouts. ')
        fail = 0
        for i in range(len(potential_coadd_objects)):
            
            sky = galsim.CelestialCoord(ra=potential_coadd_objects['ra'][i]*galsim.degrees, dec=potential_coadd_objects['dec'][i]*galsim.degrees)
            stamp_size = potential_coadd_objects['stamp'][i]
            xy = wcs.toImage(sky)
            xyI = galsim.PositionI(int(xy.x),int(xy.y))
            offset = xy - xyI
            local_wcs = wcs.local(xy)

            psf = get_coadd_psf_stamp(coadd_fname,coadd_psf_fname,xy.x,xy.y,stamp_size,oversample_factor=8)
            try:
                image_cutout = image_info[xyI.y-stamp_size//2:xyI.y+stamp_size//2, xyI.x-stamp_size//2:xyI.x+stamp_size//2]
                noise_cutout = noise_info[xyI.y-stamp_size//2:xyI.y+stamp_size//2, xyI.x-stamp_size//2:xyI.x+stamp_size//2]
            except:
                print('Object centroid is within the boundary but the cutouts are outside the boundary.')
                fail += 1
                continue

            data = np.zeros(1, dtype=[('ind', int), ('ra', float), ('dec', float), ('stamp', int), ('g1',float), ('g2',float), ('int_e1', float), ('int_e2', float), ('rot',float), ('size',float), ('redshift',float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('x', int), ('y', int), ('offset_x', float), ('offset_y', float), ('mag', float), ('dudx', float), ('dudy', float), ('dvdx', float), ('dvdy', float)])

            gind = potential_coadd_objects['ind'][i]
            t = truth_galaxies[truth_galaxies['gind'] == gind]
            data['ind']         = gind
            data['ra']          = potential_coadd_objects['ra'][i]
            data['dec']         = potential_coadd_objects['dec'][i]
            data['mag']         = potential_coadd_objects['mag'][i]
            data['stamp']       = stamp_size
            data['g1']          = t['g1']
            data['g2']          = t['g2']
            data['int_e1']      = t['int_e1']
            data['int_e2']      = t['int_e2']
            data['rot']         = t['rot']
            data['size']        = t['size']
            data['redshift']    = t['z']
            data['pind']        = t['pind']
            data['bulge_flux']  = t['bflux']
            data['disk_flux']   = t['dflux']

            data['x']           = xyI.x
            data['y']           = xyI.y
            data['offset_x']    = offset.x
            data['offset_y']    = offset.y
            data['dudx']        = local_wcs.dudx
            data['dudy']        = local_wcs.dudy
            data['dvdx']        = local_wcs.dvdx
            data['dvdy']        = local_wcs.dvdy

            output[gind] = {'image_cutouts': image_cutout, 'psf_cutouts': psf, 'noise_cutouts': noise_cutout, 'object_data': data}

        print('failed to get cutouts, ', fail)
        # dump image_cutouts, weight_cutouts, other info in FITS. 
        with open(out_fname, 'wb') as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL) 

if __name__ == "__main__":
    main(sys.argv)