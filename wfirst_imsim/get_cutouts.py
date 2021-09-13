
from re import L
import fitsio as fio
import numpy as np
import galsim
import os, sys
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

def main(argv):
    base = sys.argv[1]
    filter_ = sys.argv[2]
    simset = sys.argv[3]
    work_filter = os.path.join(base, 'roman_'+filter_)
    work_truth = os.path.join(work_filter, simset+'/truth')
    work_coadd = os.path.join(work_filter, simset+'/images/coadd')

    truth_info = fio.read(os.path.join(work_truth, 'fiducial_lensing_galaxia_'+simset+'_truth_gal.fits'))
    # truth_unique_objects = truth_info[truth_info['dither'] == -1]
    coadd_list = fio.read(os.path.join(work_truth, 'fiducial_coaddlist.fits.gz'))
    for tilename in ['26.96_-26.8']: # coadd_list['tilename']:

        if not os.path.exists(os.path.join(work_coadd, 'fiducial_H158_'+tilename+'.fits.gz')):
            continue

        out_fname = os.path.join(work_coadd, '/coadd_cutouts/fiducial_'+filter_+'_'+tilename+'_cutouts.fits')


        ra_cen = coadd_list[coadd_list['tilename'] == tilename]['coadd_ra']
        dec_cen = coadd_list[coadd_list['tilename'] == tilename]['coadd_dec']
        ra_d = coadd_list[coadd_list['tilename'] == tilename]['d_ra']
        dec_d = coadd_list[coadd_list['tilename'] == tilename]['d_dec']
        radec_limit = [ra_cen - ra_d, ra_cen + ra_d, dec_cen + dec_d, dec_cen - dec_d]
        mask_objects = ((truth_info['ra'] >= radec_limit[0]) & (truth_info['ra'] <= radec_limit[1])
                        & (truth_info['dec'] >= radec_limit[2]) & (truth_info['dec'] >= radec_limit[3]))
        potential_coadd_objects = truth_info[mask_objects]


        coadd = fio.FITS(os.path.join(work_coadd, 'fiducial_H158_'+tilename+'.fits.gz'))
        hdulist = fits.open(os.path.join(work_coadd, 'fiducial_H158_'+tilename+'.fits.gz'))
        image_info = coadd[1].read()
        weight_info = coadd[2].read()
        astropy_wcs = WCS(hdulist[1].header)
        wcs = galsim.AstropyWCS(wcs=astropy_wcs)
        data = np.zeros(len(potential_coadd_objects), dtype=[('ind', int), ('ra', float), ('dec', float), ('stamp', int), ('g1',float), ('g2',float), ('rot',float), ('size',float), ('redshift',float), ('pind',int), ('bulge_flux',float), ('disk_flux',float), ('x', int), ('y', int), ('offset_x', float), ('offset_y', float), ('mag', float), ('dudx', float), ('dudy', float), ('dvdx', float), ('dvdy', float)])
        print('Getting ', len(potential_coadd_objects), 'cutouts. ')
        fail = 0
        for i in range(len(potential_coadd_objects)):

            sky = galsim.CelestialCoord(ra=potential_coadd_objects['ra'][i]*galsim.degrees, dec=potential_coadd_objects['dec'][i]*galsim.degrees)
            stamp_size = potential_coadd_objects['stamp'][i]
            xy = wcs.toImage(sky)
            xyI = galsim.PositionI(int(xy.x),int(xy.y))
            offset = xy - xyI
            local_wcs = wcs.local(xy)
            print(xy)
            try:
                image_cutout = image_info[xyI.x-stamp_size//2:xyI.x+stamp_size//2, xyI.y-stamp_size//2:xyI.y+stamp_size//2]
                weight_cutout = weight_info[xyI.x-stamp_size//2:xyI.x+stamp_size//2, xyI.y-stamp_size//2:xyI.y+stamp_size//2]
            except:
                print('Object centroid is within the boundary but the cutouts are outside the boundary.')
                fail += 1
                continue


            data['ind'][i]         = potential_coadd_objects['gind'][i]
            data['ra'][i]          = potential_coadd_objects['ra'][i]
            data['dec'][i]         = potential_coadd_objects['dec'][i]
            data['mag'][i]         = potential_coadd_objects['mag'][i]
            data['stamp'][i]       = stamp_size
            data['g1'][i]          = potential_coadd_objects['g1'][i]
            data['g2'][i]          = potential_coadd_objects['g2'][i]
            data['int_e1'][i]      = potential_coadd_objects['int_e1'][i]
            data['int_e2'][i]      = potential_coadd_objects['int_e2'][i]
            data['rot'][i]         = potential_coadd_objects['rot'][i]
            data['size'][i]        = potential_coadd_objects['size'][i]
            data['z'][i]           = potential_coadd_objects['redshift'][i]
            data['pind'][i]        = potential_coadd_objects['pind'][i]
            data['bulge_flux'][i]  = potential_coadd_objects['bflux'][i]
            data['disk_flux'][i]  = potential_coadd_objects['dflux'][i]

            data['x'][i]           = xyI.x
            data['y'][i]           = xyI.y
            data['offset_x'][i]    = offset.x
            data['offset_y'][i]    = offset.y
            data['dudx'][i]        = local_wcs.dudx
            data['dudy'][i]        = local_wcs.dudy
            data['dvdx'][i]        = local_wcs.dvdx
            data['dvdy'][i]        = local_wcs.dvdy

            # dump image_cutouts, weight_cutouts, other info in FITS. 
            if i==1000:
                np.savetxt('image_cutout_'+str(i)+'.txt', image_cutout)
                np.savetxt('weight_cutout_'+str(i)+'.txt', weight_cutout)
        print('failed to get cutouts, ', fail)

if __name__ == "__main__":
    main(sys.argv)