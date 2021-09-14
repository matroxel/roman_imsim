
from re import L
import fitsio as fio
import numpy as np
import galsim
import os, sys

def main(argv):
    base = sys.argv[1]
    filter_ = sys.argv[2]
    simset = sys.argv[3]
    work_filter = os.path.join(base, 'roman_'+filter_)
    work_truth = os.path.join(work_filter, simset+'/truth')
    work_coadd = os.path.join(work_filter, simset+'/images/coadd')

    truth_galaxies = fio.read(os.path.join(work_truth, 'fiducial_lensing_galaxia_'+simset+'_truth_gal.fits'))
    truth_simulated = fio.read(os.path.join(work_truth, 'fiducial_'+filter_+'_index_sorted.fits.gz'))
    truth_unique_objects = truth_simulated[truth_simulated['dither'] == -1]
    coadd_list = fio.read(os.path.join(work_truth, 'fiducial_coaddlist.fits.gz'))
    for tilename in ['26.96_-26.8']: # coadd_list['tilename']:

        if not os.path.exists(os.path.join(work_coadd, 'fiducial_H158_'+tilename+'.fits.gz')):
            continue

        out_fname = os.path.join(work_coadd, 'coadd_cutouts/fiducial_'+filter_+'_'+tilename+'_cutouts.fits')

        ra_cen = coadd_list[coadd_list['tilename'] == tilename]['coadd_ra']
        dec_cen = coadd_list[coadd_list['tilename'] == tilename]['coadd_dec']
        ra_d = coadd_list[coadd_list['tilename'] == tilename]['d_ra']
        dec_d = coadd_list[coadd_list['tilename'] == tilename]['d_dec']
        radec_limit = [ra_cen - ra_d, ra_cen + ra_d, dec_cen - dec_d, dec_cen + dec_d]
        mask_objects = ((truth_unique_objects['ra'] >= radec_limit[0]) & (truth_unique_objects['ra'] <= radec_limit[1])
                        & (truth_unique_objects['dec'] >= radec_limit[2]) & (truth_unique_objects['dec'] <= radec_limit[3]))
        potential_coadd_objects = truth_unique_objects[mask_objects]


        coadd_fname = os.path.join(work_coadd, 'fiducial_H158_'+tilename+'.fits.gz')
        coadd = fio.FITS(coadd_fname)
        image_info = coadd[1].read()
        weight_info = coadd[2].read()
        wcs = galsim.AstropyWCS(file_name=coadd_fname, hdu=1)
        data = np.zeros(len(potential_coadd_objects), dtype=[('ind', int), ('ra', float), ('dec', float), ('stamp', int), ('g1',float), ('g2',float), ('int_e1', float), ('int_e2', float), ('rot',float), ('size',float), ('redshift',float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('x', int), ('y', int), ('offset_x', float), ('offset_y', float), ('mag', float), ('dudx', float), ('dudy', float), ('dvdx', float), ('dvdy', float)])
        image_data = []
        weight_data = []
        print('Getting ', len(potential_coadd_objects), 'cutouts. ')
        fail = 0
        for i in range(len(potential_coadd_objects)):
            
            sky = galsim.CelestialCoord(ra=potential_coadd_objects['ra'][i]*galsim.degrees, dec=potential_coadd_objects['dec'][i]*galsim.degrees)
            stamp_size = potential_coadd_objects['stamp'][i]
            xy = wcs.toImage(sky)
            xyI = galsim.PositionI(int(xy.x),int(xy.y))
            offset = xy - xyI
            local_wcs = wcs.local(xy)
            try:
                image_cutout = image_info[xyI.y-stamp_size//2:xyI.y+stamp_size//2, xyI.x-stamp_size//2:xyI.x+stamp_size//2]
                weight_cutout = weight_info[xyI.y-stamp_size//2:xyI.y+stamp_size//2, xyI.x-stamp_size//2:xyI.x+stamp_size//2]
            except:
                print('Object centroid is within the boundary but the cutouts are outside the boundary.')
                fail += 1
                continue

            gind = potential_coadd_objects['ind'][i]
            t = truth_galaxies[truth_galaxies['gind'] == gind]
            data['ind'][i]         = gind
            data['ra'][i]          = potential_coadd_objects['ra'][i]
            data['dec'][i]         = potential_coadd_objects['dec'][i]
            data['mag'][i]         = potential_coadd_objects['mag'][i]
            data['stamp'][i]       = stamp_size
            data['g1'][i]          = t['g1']
            data['g2'][i]          = t['g2']
            data['int_e1'][i]      = t['int_e1']
            data['int_e2'][i]      = t['int_e2']
            data['rot'][i]         = t['rot']
            data['size'][i]        = t['size']
            data['redshift'][i]    = t['z']
            data['pind'][i]        = t['pind']
            data['bulge_flux'][i]  = t['bflux']
            data['disk_flux'][i]   = t['dflux']

            data['x'][i]           = xyI.x
            data['y'][i]           = xyI.y
            data['offset_x'][i]    = offset.x
            data['offset_y'][i]    = offset.y
            data['dudx'][i]        = local_wcs.dudx
            data['dudy'][i]        = local_wcs.dudy
            data['dvdx'][i]        = local_wcs.dvdx
            data['dvdy'][i]        = local_wcs.dvdy

            image_data.append(image_cutout)
            weight_data.append(weight_cutout)

            # if i==1000:
            #     np.savetxt('image_cutout_'+str(i)+'.txt', image_cutout)
            #     np.savetxt('weight_cutout_'+str(i)+'.txt', weight_cutout)
        print('failed to get cutouts, ', fail)
        # dump image_cutouts, weight_cutouts, other info in FITS. 
        fits = fio.FITS(out_fname,'rw')
        fits.write(image_data)
        fits.write(weight_data)
        fits.write(data)

        # os.system('gzip '+out_fname)

if __name__ == "__main__":
    main(sys.argv)