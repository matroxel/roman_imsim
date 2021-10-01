import fitsio as fio
import numpy as np
import galsim
import os, sys
import pickle
import shutil

import wfirst_imsim

def main(argv):

    params_file = sys.argv[1]
    coadd_number = sys.argv[2]
    base = sys.argv[3]
    filter_ = sys.argv[4]
    simset = sys.argv[5]
    
    sim = wfirst_imsim.postprocessing(params_file)

    work_filter = os.path.join(base, 'roman_'+filter_)
    work_truth = os.path.join(work_filter, simset+'/truth')
    work_coadd = os.path.join(work_filter, simset+'/images/coadd')
    work_psf = os.path.join(work_filter, simset+'/psf/coadd')
    truth_galaxies = fio.read(os.path.join(work_truth, 'fiducial_lensing_galaxia_'+simset+'_truth_gal.fits'))
    truth_simulated = fio.read(os.path.join(work_truth, 'fiducial_'+filter_+'_index_sorted.fits.gz'))
    truth_unique_objects = truth_simulated[truth_simulated['dither'] == -1]
    coadd_list = fio.read(os.path.join(work_truth, 'fiducial_coaddlist.fits.gz'))
    
    tilename = coadd_list['tilename'][int(coadd_number)-1]

    if not os.path.exists(os.path.join(work_coadd, 'fiducial_H158_'+tilename+'.fits.gz')):
        print('this tile does not have a coadd. exiting.')
        return 
    else:
        filename = os.path.join(work_coadd, 'fiducial_H158_'+tilename+'.fits.gz')
        tmp_filename_ = os.path.join('/scratch', 'fiducial_H158_'+tilename+'.fits')
        shutil.copy(filename, tmp_filename_+'.gz')
        os.system('gunzip '+tmp_filename_+'.gz')
        os.chdir('/scratch/')

    out_fname = os.path.join(work_coadd, 'coadd_cutouts/fiducial_'+filter_+'_'+tilename+'_cutouts.pickle')

    ra_cen = coadd_list[coadd_list['tilename'] == tilename]['coadd_ra']
    dec_cen = coadd_list[coadd_list['tilename'] == tilename]['coadd_dec']
    ra_d = coadd_list[coadd_list['tilename'] == tilename]['d_ra']
    dec_d = coadd_list[coadd_list['tilename'] == tilename]['d_dec']
    radec_limit = [ra_cen - ra_d, ra_cen + ra_d, dec_cen - dec_d, dec_cen + dec_d]
    mask_objects = ((truth_unique_objects['ra'] >= radec_limit[0]) & (truth_unique_objects['ra'] <= radec_limit[1])
                    & (truth_unique_objects['dec'] >= radec_limit[2]) & (truth_unique_objects['dec'] <= radec_limit[3]))
    potential_coadd_objects = truth_unique_objects[mask_objects]


    # coadd_fname = os.path.join(work_coadd, 'fiducial_H158_'+tilename+'.fits.gz')
    coadd_fname = tmp_filename_ # os.path.join(work_coadd, 'fiducial_H158_'+tilename+'.fits.gz')
    coadd_psf_fname = os.path.join(work_psf, 'fiducial_H158_'+tilename+'_psf.fits')
    coadd = fio.FITS(coadd_fname)
    image_info = coadd['SCI'].read()
    weight_info = coadd['WHT'].read()
    noise_info = coadd['ERR'].read()
    wcs = galsim.AstropyWCS(file_name=coadd_fname, hdu=1)
    output = {}
    print('Getting ', len(potential_coadd_objects), 'cutouts. ')
    fail = 0
    for i in range(len(potential_coadd_objects)):

        if i%100==0:
            print(str(i)+'th cutouts')

        sky = galsim.CelestialCoord(ra=potential_coadd_objects['ra'][i]*galsim.degrees, dec=potential_coadd_objects['dec'][i]*galsim.degrees)
        stamp_size = potential_coadd_objects['stamp'][i]
        xy = wcs.toImage(sky)
        xyI = galsim.PositionI(int(xy.x), int(xy.y))
        offset = xy - xyI
        local_wcs = wcs.local(xy)
        psf = sim.get_coadd_psf_stamp(coadd_fname, coadd_psf_fname, xy.x, xy.y, stamp_size)
        if psf is None:
            print('PSF does not exist???')
            fail += 1
            continue
        try:
            if weight_info[xyI.x, xyI.y] == 0:
                continue
            image_cutout = image_info[xyI.y-stamp_size//2:xyI.y+stamp_size//2, xyI.x-stamp_size//2:xyI.x+stamp_size//2]
            noise_cutout = noise_info[xyI.y-stamp_size//2:xyI.y+stamp_size//2, xyI.x-stamp_size//2:xyI.x+stamp_size//2]
            weight_cutout = weight_info[xyI.y-stamp_size//2:xyI.y+stamp_size//2, xyI.x-stamp_size//2:xyI.x+stamp_size//2]
            # Get the right galaxy center for the cutouts. 
            new_pos  = galsim.PositionD(xy.x-(xyI.x-stamp_size//2), xy.y-(xyI.y-stamp_size//2))
            new_local_wcs = wcs.affine(image_pos=new_pos)
        except:
            print('Object centroid is within the boundary but the cutouts are outside the boundary.')
            fail += 1
            continue
        data = np.zeros(1, dtype=[('ind', int), ('ra', float), ('dec', float), ('stamp', int), ('g1',float), ('g2',float), ('int_e1', float), ('int_e2', float), ('rot',float), ('size',float), ('redshift',float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('orig_x', float), ('orig_y', float), ('cutout_x', float), ('cutout_y', float), ('mag', float), ('dudx', float), ('dudy', float), ('dvdx', float), ('dvdy', float)])
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
        data['orig_x']      = xy.x
        data['orig_y']      = xy.y
        data['cutout_x']    = new_pos.x
        data['cutout_y']    = new_pos.y
        data['dudx']        = new_local_wcs.dudx
        data['dudy']        = new_local_wcs.dudy
        data['dvdx']        = new_local_wcs.dvdx
        data['dvdy']        = new_local_wcs.dvdy
        output[gind] = {'image_cutouts': image_cutout, 'psf_cutouts': psf, 'weight_cutouts': weight_cutout, 'noise_cutouts': noise_cutout, 'object_data': data}
    print('failed to get cutouts, ', fail)
    
    if len(potential_coadd_objects) == 0:
        print('No cutout files saved.')
        os.remove(tmp_filename_)
        return 
    # dump image_cutouts, weight_cutouts, other info in FITS. 
    with open(out_fname, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    os.remove(tmp_filename_)

if __name__ == "__main__":
    main(sys.argv) 