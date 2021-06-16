import numpy as np
import galsim
import galsim.wfirst as wfirst
import os
import fitsio as fio

# These are raw fits forms fo the relevant instance cats that I (slowly) translated from txt form 
d = fio.FITS('disk_gal_cat_1616897.fits')[-1].read()
b = fio.FITS('bulge_gal_cat_1616897.fits')[-1].read()
n = fio.FITS('knots_cat_1616897.fits')[-1].read()
b = b[np.argsort(b['ID'])]
d = d[np.argsort(d['ID'])]
n = n[np.argsort(n['ID'])]

# Extract unique object id to combine non-delta models into one entry
bitmask = 0b1111111111
oldid   = np.hstack((b['ID'],d['ID'],n['ID']))
id      = oldid >> 10
comp    = oldid & bitmask
uid,i   = np.unique(id,return_index=True)

# Tags for each component for reference
disk  = 107
bulge = 97
knots = 127

# Combined fits truth file format. Some  columns are arrays of length 3 for bulge/disk/knot components.
store = np.zeros(len(uid), dtype=[('gind','i8')]
                            +[('ra',float)]
                            +[('dec',float)]
                            +[('g1','f4')]
                            +[('g2','f4')]
                            +[('k','f4')]
                            +[('z','f4')]
                            +[('size','f4',(3))]
                            +[('q','f4',(3))]
                            +[('pa','f4',(3))]
                            +[('knots','i4')]
                            +[('mag_norm','f4',(3))]
                            +[('sed_b','S56')]
                            +[('sed_d','S56')]
                            +[('sed_n','S56')]
                            +[('A_v','f4',(3))]
                            +[('R_v','f4',(3))])

# Add elements common to each component
store['gind'] = uid
store['ra'] = np.hstack((b['RA'],d['RA'],n['RA']))[i]/180.*np.pi
store['dec'] = np.hstack((b['DEC'],d['DEC'],n['DEC']))[i]/180.*np.pi
store['g1'] = np.hstack((b['GAMMA1'],d['GAMMA1'],n['GAMMA1']))[i]
store['g2'] = np.hstack((b['GAMMA2'],d['GAMMA2'],n['GAMMA2']))[i]
store['k'] = np.hstack((b['KAPPA'],d['KAPPA'],n['KAPPA']))[i]
store['z'] = np.hstack((b['REDSHIFT'],d['REDSHIFT'],n['REDSHIFT']))[i]

# Loop over each component and add relevant stuff to appropriate entries
m = np.where(np.in1d(store['gind'], b['ID']>>10))[0]
store['size'][m,0] = b['major']
store['q'][m,0]    = b['major']/b['minor']
store['pa'][m,0]   = b['pa']
store['mag_norm'][m,0]   = b['MAG_NORM']
store['sed_b'][m]   = b['SED_NAME']
store['A_v'][m,0]   = b['dust_rest_1']
store['R_v'][m,0]   = b['dust_rest_2']

m = np.where(np.in1d(store['gind'], d['ID']>>10))[0]
store['size'][m,1] = d['major']
store['q'][m,1]    = d['major']/d['minor']
store['pa'][m,1]   = d['pa']
store['mag_norm'][m,1]   = d['MAG_NORM']
store['sed_d'][m]   = d['SED_NAME']
store['A_v'][m,1]   = d['dust_rest_1']
store['R_v'][m,1]   = d['dust_rest_2']

m = np.where(np.in1d(store['gind'], n['ID']>>10))[0]
store['knots'][m]  = n['number']
store['size'][m,2] = n['major']
store['q'][m,2]    = n['major']/n['minor']
store['pa'][m,2]   = n['pa']
store['mag_norm'][m,2]   = n['MAG_NORM']
store['sed_n'][m]   = n['SED_NAME']
store['A_v'][m,2]   = n['dust_rest_1']
store['R_v'][m,2]   = n['dust_rest_2']

fio.write('dc2_truth_gal.fits',store,clobber=True)

# These are raw fits forms fo the relevant instance cats that I (slowly) translated from txt form 
agn = fio.FITS('agn_gal_cat_1616897.fits')[-1].read()
s1 = fio.FITS('bright_stars_1616897.fits')[-1].read()
s2 = fio.FITS('star_cat_1616897.fits')[-1].read()
sne = fio.FITS('sne_cat_1616897.fits')[-1].read()

# Combined fits truth file format for point source components, which are simulated together in a separate stage than extended sources in my sim. 
length = len(s1)+len(s2)#+len(agn)+len(sne)
store = np.zeros(length, dtype=[('gind','i8')]
                            +[('ra',float)]
                            +[('dec',float)]
                            +[('g1','f4')]
                            +[('g2','f4')]
                            +[('k','f4')]
                            +[('z','f4')]
                            +[('mag_norm','f4')]
                            +[('sed','S56')]
                            +[('A_v','f4')]
                            +[('R_v','f4')])

#Loop over and just dump everything to file.
start = 0
store['gind'][start:start+len(s1)] = s1['ID']
store['ra'][start:start+len(s1)]   = s1['RA']/180.*np.pi
store['dec'][start:start+len(s1)]  = s1['DEC']/180.*np.pi
store['g1'][start:start+len(s1)]   = s1['GAMMA1']
store['g2'][start:start+len(s1)]   = s1['GAMMA2']
store['k'][start:start+len(s1)]    = s1['KAPPA']
store['z'][start:start+len(s1)]    = s1['REDSHIFT']
store['mag_norm'][start:start+len(s1)] = s1['MAG_NORM']
store['sed'][start:start+len(s1)]   = s1['SED_NAME']
store['A_v'][start:start+len(s1)]   = s1['dust_rest_1']
store['R_v'][start:start+len(s1)]   = s1['dust_rest_2']
start+=len(s1)

store['gind'][start:start+len(s2)] = s2['ID']
store['ra'][start:start+len(s2)]   = s2['RA']/180.*np.pi
store['dec'][start:start+len(s2)]  = s2['DEC']/180.*np.pi
store['g1'][start:start+len(s2)]   = s2['GAMMA1']
store['g2'][start:start+len(s2)]   = s2['GAMMA2']
store['k'][start:start+len(s2)]    = s2['KAPPA']
store['z'][start:start+len(s2)]    = s2['REDSHIFT']
store['mag_norm'][start:start+len(s2)] = s2['MAG_NORM']
store['sed'][start:start+len(s2)]   = s2['SED_NAME']
store['A_v'][start:start+len(s2)]   = s2['dust_rest_1']
store['R_v'][start:start+len(s2)]   = s2['dust_rest_2']
start+=len(s2)

# store['gind'][start:start+len(agn)] = agn['ID']
# store['ra'][start:start+len(agn)]   = agn['RA']/180.*np.pi
# store['dec'][start:start+len(agn)]  = agn['DEC']/180.*np.pi
# store['g1'][start:start+len(agn)]   = agn['GAMMA1']
# store['g2'][start:start+len(agn)]   = agn['GAMMA2']
# store['k'][start:start+len(agn)]    = agn['KAPPA']
# store['z'][start:start+len(agn)]    = agn['REDSHIFT']
# store['mag_norm'][start:start+len(agn)] = agn['MAG_NORM']
# store['sed'][start:start+len(agn)]   = agn['SED_NAME']
# store['A_v'][start:start+len(agn)]   = agn['dust_rest_1']
# store['R_v'][start:start+len(agn)]   = agn['dust_rest_2']
# start+=len(agn)

# store['gind'][start:start+len(sne)] = np.arange(len(sne))
# store['ra'][start:start+len(sne)]   = sne['RA']/180.*np.pi
# store['dec'][start:start+len(sne)]  = sne['DEC']/180.*np.pi
# store['g1'][start:start+len(sne)]   = sne['GAMMA1']
# store['g2'][start:start+len(sne)]   = sne['GAMMA2']
# store['k'][start:start+len(sne)]    = sne['KAPPA']
# store['z'][start:start+len(sne)]    = sne['REDSHIFT']
# store['mag_norm'][start:start+len(sne)] = sne['MAG_NORM']
# store['sed'][start:start+len(sne)]   = sne['SED_NAME']
# store['A_v'][start:start+len(sne)]   = sne['dust_rest_1']
# store['R_v'][start:start+len(sne)]   = sne['dust_rest_2']
# start+=len(sne)

fio.write('dc2_truth_star.fits',store,clobber=True)

