
bad=array([571133, 571134, 571135, 571277, 571278, 571279, 571282, 571283,571286, 571288, 571289, 571290, 571291, 571292, 571293, 571294,571295, 571297, 571298, 571299, 571300, 571301, 571302, 571303,571304, 571305, 571306, 571307, 571308, 571309, 571310, 571311,571312, 571313, 571314, 571315, 571316, 571317, 571318, 571319,571320, 571321, 571322, 571323, 571324, 571325, 571326, 571327,571338, 571360, 571361, 571362, 571363, 571366, 571368, 571369,571370, 571371, 571372, 571373, 571374, 571375, 571386, 572500,572501, 572503, 572672, 572673, 572674, 572675, 572676, 572677,572678, 572679, 572680, 572681, 572683, 572684, 572685, 572686,572687, 572688, 572689, 572690, 572691, 572692, 572693, 572694,572695, 572696, 572697, 572698, 572699, 572700, 572701])


import simulate as sim0
import fitsio as fio
import numpy as np
sim = sim0.wfirst_sim('H158_los_i3.yaml')
pix = [ 572700, 572701]
for p in pix:
    print p
    if sim.params['psf_meds'] == sim.params['output_meds']:
        continue
    new = fio.FITS(sim.meds_filename(p),'rw')
    psf = fio.FITS(sim.meds_filename_psf(p),'rw')
    np.save(sim.meds_filename(p)+'.back.npy',new['psf'].read())
    new['psf'].write(psf['psf'].read(), start=0)
    new.close()
    psf.close()

r_min= 1.
r_max= 200.
Nrbins = 120
dlogr = np.log10(r_max/r_min) / Nrbins
rbin_array = np.arange(Nrbins)
r_array = 10**( (rbin_array+0.5)*dlogr+np.log10(r_min) )


for i in [1,2,3]:
  plt.errorbar(r_array,r_array**2*np.loadtxt('/Users/troxel/Downloads/Totroxel/FULLMonopole_True_Z'+str(i)+'_H3.txt'),yerr=r_array**2*np.diagonal(np.sqrt(np.load('/Users/troxel/Downloads/recorrelationfunctions-2/z'+str(i)+'h3Cov_00_XIAO.npy'))),color=cc[i-1],ls='',marker='.',label='z'+str(i)+'h3')

plt.legend()
#plt.yscale('log')
plt.xscale('log')
plt.xlabel('R')
plt.ylabel('R^2 Mono.')
plt.savefig('tmp.png')
plt.close()

for i in [1,2,3]:
  mask = np.loadtxt('/Users/troxel/Downloads/Totroxel/FULLQuadrupole_True_Z'+str(i)+'_H3.txt')>-999
  plt.errorbar(r_array[mask],r_array[mask]**2*np.loadtxt('/Users/troxel/Downloads/Totroxel/FULLQuadrupole_True_Z'+str(i)+'_H3.txt')[mask],yerr=r_array[mask]**2*np.diagonal(np.sqrt(np.load('/Users/troxel/Downloads/recorrelationfunctions-2/z'+str(i)+'h3Cov_00_XIAO.npy')))[mask],color=cc[i-1],ls='',marker='.',label='z'+str(i)+'h3')
  #plt.errorbar(r_array[~mask],-r_array[~mask]**2*np.loadtxt('/Users/troxel/Downloads/Totroxel/FULLQuadrupole_True_Z'+str(i)+'_H3.txt')[~mask],yerr=r_array[~mask]**2*np.diagonal(np.sqrt(np.load('/Users/troxel/Downloads/recorrelationfunctions-2/z'+str(i)+'h3Cov_00_XIAO.npy')))[~mask],color=cc[i-1],ls='',marker='x',label='')

plt.legend()
#plt.yscale('log')
plt.xscale('log')
plt.xlabel('R')
plt.ylabel('R^2 Quad.')
plt.savefig('tmp2.png')
plt.close()




import numpy as np
import fitsio as fio
import treecorr
import meds

import matplotlib
matplotlib.use ('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pylab

def mean_e(e,mask):
    return np.mean(e[mask]),np.std(e[mask])/np.sqrt(np.sum(mask))

def diff_e(e,e0,mask,d):
    return np.mean(e[mask]-e0[mask])/d,np.std(e[mask]-e0[mask])/np.sqrt(np.sum(mask))/d

los  = fio.FITS('/fs/scratch/cond0083/wfirst_sim_out/test_los_H158_main.fits')[-1].read()
z4p  = fio.FITS('/fs/scratch/cond0083/wfirst_sim_out/test_z4_p_H158_main.fits')[-1].read()
fid  = fio.FITS('/fs/scratch/cond0083/wfirst_sim_out/test_H158_main.fits')[-1].read()
filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_table.fits'
table = fio.FITS('/fs/scratch/cond0083/wfirst_sim_out/test_H158_table_wxy.fits')[-1].read()
mask = (fid['flags']==1)&(z4p['flags']==1)&(los['flags']==1)
maskfid = (fid['flags']==1)
maskz4p = (z4p['flags']==1)
masklos = (los['flags']==1)
print 'number in fid',np.sum(maskfid)
print 'number in z4p',np.sum(maskz4p)
print 'number in los',np.sum(masklos)
print 'joint number',np.sum(mask)
print 'fid mean e1',mean_e(fid['e1'],mask)
print 'fid mean e2',mean_e(fid['e2'],mask)
print 'z4p mean e1',mean_e(z4p['e1'],mask)
print 'z4p mean e2',mean_e(z4p['e2'],mask)
print 'los mean e1',mean_e(los['e1'],mask)
print 'los mean e2',mean_e(los['e2'],mask)
print 'de1/dz4',diff_e(z4p['e1'],fid['e1'],mask,6.465)
print 'de2/dz4',diff_e(z4p['e2'],fid['e2'],mask,6.465)
print 'de1/dlos',diff_e(los['e1'],fid['e1'],mask,0.015)
print 'de2/dlos',diff_e(los['e2'],fid['e2'],mask,0.015)

print 'de1 z4p',diff_e(z4p['e1'],fid['e1'],mask,1)
print 'de2 z4p',diff_e(z4p['e2'],fid['e2'],mask,1)
print 'de1 los',diff_e(los['e1'],fid['e1'],mask,1)
print 'de2 los',diff_e(los['e2'],fid['e2'],mask,1)

print 'fid mean psf e1',mean_e(fid['mean_hsm_psf_e1_sky'],mask)
print 'fid mean psf e2',mean_e(fid['mean_hsm_psf_e2_sky'],mask)
print 'z4p mean psf e1',mean_e(z4p['mean_hsm_psf_e1_sky'],mask)
print 'z4p mean psf e2',mean_e(z4p['mean_hsm_psf_e2_sky'],mask)
print 'los mean psf e1',mean_e(los['mean_hsm_psf_e1_sky'],mask)
print 'los mean psf e2',mean_e(los['mean_hsm_psf_e2_sky'],mask)

print 'fid mean psf fwhm',mean_e(fid['mean_hsm_psf_sigma'],mask)
print 'z4p mean psf fwhm',mean_e(z4p['mean_hsm_psf_sigma'],mask)
print 'los mean psf fwhm',mean_e(los['mean_hsm_psf_sigma'],mask)

print 'fid mean snr',mean_e(fid['snr'],mask)
print 'z4p mean snr',mean_e(z4p['snr'],mask)
print 'los mean snr',mean_e(los['snr'],mask)

print 'fid mean snr',mean_e(fid['radius'],mask)
print 'z4p mean snr',mean_e(z4p['radius'],mask)
print 'los mean snr',mean_e(los['radius'],mask)

print 'fid mean snr',mean_e(fid['n_exposure'],mask)
print 'z4p mean snr',mean_e(z4p['n_exposure'],mask)
print 'los mean snr',mean_e(los['n_exposure'],mask)


plt.hist(fid['e1'][mask],bins=500,histtype='stepfilled')
plt.axvline(x=0.,color='k')
plt.axvline(x=-0.2,color='k')
plt.axvline(x=0.2,color='k')
plt.xlabel('e1')
plt.ylabel('N')
plt.tight_layout()
plt.savefig('e1_hist.png',bbox_inches='tight')
plt.close()

plt.hist(fid['e2'][mask],bins=500,histtype='stepfilled')
plt.axvline(x=0.,color='k')
plt.axvline(x=-0.2,color='k')
plt.axvline(x=0.2,color='k')
plt.xlabel('e2')
plt.ylabel('N')
plt.tight_layout()
plt.savefig('e2_hist.png',bbox_inches='tight')
plt.close()

plt.hist(fid['mean_hsm_psf_e1_sky'][mask],bins=500,histtype='stepfilled')
plt.xlabel('PSF e1')
plt.ylabel('N')
plt.tight_layout()
plt.savefig('psfe1_hist.png',bbox_inches='tight')
plt.close()

plt.hist(fid['mean_hsm_psf_e2_sky'][mask],bins=500,histtype='stepfilled')
plt.xlabel('PSF e2')
plt.ylabel('N')
plt.tight_layout()
plt.savefig('psfe2_hist.png',bbox_inches='tight')
plt.close()

plt.hist(fid['mean_hsm_psf_sigma'][mask]*2.*np.sqrt(2.*np.log(2.)),bins=500,histtype='stepfilled')
plt.xlabel('PSF FWHM (pix)')
plt.ylabel('N')
plt.tight_layout()
plt.savefig('psffwhm_hist.png',bbox_inches='tight')
plt.close()

plt.hist(np.log10(fid['snr'][mask]),bins=500,histtype='stepfilled')
plt.xlabel('log10(signal-to-noise)')
plt.ylabel('N')
plt.tight_layout()
plt.savefig('snr_hist.png',bbox_inches='tight')
plt.close()

plt.hist(fid['radius'][mask],bins=500,histtype='stepfilled')
plt.xlabel('radius')
plt.ylabel('N')
plt.tight_layout()
plt.savefig('rad_hist.png',bbox_inches='tight')
plt.close()

plt.hist(fid['n_exposure'][mask],bins=np.arange(14)+0.5,histtype='stepfilled')
plt.xlabel('Number of exposures')
plt.ylabel('N')
plt.tight_layout()
plt.savefig('nexp_hist.png',bbox_inches='tight')
plt.close()


import galsim.wfirst as wfirst
sca_cen_mm = np.array([
[ -21.94,  13.12 ],
[ -22.09, -31.77 ],
[ -22.24, -81.15 ],
[ -65.82,  23.76 ],
[ -66.32, -20.77 ],
[ -66.82, -70.15 ],
[-109.7 ,  44.12 ],
[-110.46,   0.24 ],
[-111.56, -49.15 ],
[  21.94,  13.12 ],
[  22.09, -31.77 ],
[  22.24, -81.15 ],
[  65.82,  23.76 ],
[  66.32, -20.77 ],
[  66.82, -70.15 ],
[ 109.7 ,  44.12 ],
[ 110.46,   0.24 ],
[ 111.56, -49.15 ]
])

def sca_pix_to_mm_fov(x,y,sca):
    x = ( x - 4096 / 2 ) * 10e-3 + sca_cen_mm[sca,0]
    y = ( y - 4096 / 2 ) * 10e-3 + sca_cen_mm[sca,1]
    return x,y

def get_e_pix(cat,table,mask):
    e1 = np.zeros(len(table))
    e2 = np.zeros(len(table))
    diff=np.diff(table['gal'])
    diff=np.where(diff!=0)[0]+1
    diff=np.append([0],diff)
    diff=np.append(diff,[None])
    for i in xrange(len(diff)-1):
        idx = table['gal'][diff[i]]
        if mask[idx]:
            e1[diff[i]:diff[i+1]]=cat['e1'][idx]
            e2[diff[i]:diff[i+1]]=cat['e2'][idx]
    return e1,e2

def sca_map_diff(table,e,e0,mask,name):
    gal_mask = table['gal'][table['dither']==176230]
    sca_mask = table['sca'][table['dither']==176230]
    ra  = fid['ra'][gal_mask]
    dec = fid['dec'][gal_mask]
    sca_array = np.zeros(len(ra))
    std_array = np.zeros(len(ra))
    sca_array_norm = np.zeros(len(ra))
    for i in range(18):
        gal = table['gal'][table['sca']==i+1]
        gal = gal[np.in1d(gal,np.where(mask)[0],assume_unique=False)]
        sca_array[sca_mask==i+1] = np.mean(e[gal]-e0[gal])
        print i,sca_array
        std_array[sca_mask==i+1] = np.std(e[gal]-e0[gal])/np.sqrt(len(gal))
        sca_array_norm[sca_mask==i+1] = np.mean((e[gal]-e0[gal])/e0[gal])
    plt.scatter(ra,dec,c=sca_array,marker='o',edgecolors='none')
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.colorbar()
    plt.savefig(name+'_sca.png')
    plt.close()
    plt.scatter(ra,dec,c=sca_array_norm,marker='o',edgecolors='none')
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.colorbar()
    plt.savefig(name+'_sca_norm.png')
    plt.close()
    return sca_array,sca_array_norm,std_array

def sca_map(table,e,mask,name):
    gal_mask = table['gal'][table['dither']==176230]
    sca_mask = table['sca'][table['dither']==176230]
    ra  = fid['ra'][gal_mask]
    dec = fid['dec'][gal_mask]
    sca_array = np.zeros(len(ra))
    std_array = np.zeros(len(ra))
    sca_array_norm = np.zeros(len(ra))
    for i in range(18):
        gal = table['gal'][table['sca']==i+1]
        gal = gal[np.in1d(gal,np.where(mask)[0],assume_unique=False)]
        sca_array[sca_mask==i+1] = np.mean(e[gal])
        print i,np.mean(e[gal])
        std_array[sca_mask==i+1] = np.std(e[gal])/np.sqrt(len(gal))
        sca_array_norm[sca_mask==i+1] = np.mean(e[gal])/np.mean(e[mask])
    plt.scatter(ra,dec,c=sca_array,marker='o',edgecolors='none')
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.colorbar()
    plt.savefig(name+'_sca.png')
    plt.close()
    plt.scatter(ra,dec,c=sca_array_norm,marker='o',edgecolors='none')
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.colorbar()
    plt.savefig(name+'_sca_norm.png')
    plt.close()
    return sca_array,sca_array_norm,std_array

def fov_map_diff(table,cat,cat2,mask,name):
    e1,e2 = get_e_pix(cat,table,mask)
    e1b,e2b = get_e_pix(cat2,table,mask)
    pixmask = (e1!=0)&(e1b!=0)
    m,s=diff_e(cat2['e1'],cat['e1'],mask,1.)
    x,y=sca_pix_to_mm_fov(table['x'][pixmask],table['y'][pixmask],table['sca'][pixmask]-1)
    plt.hexbin(x,y,C=e1b[pixmask]-e1[pixmask],gridsize=(10,5),vmin=m-40*s,vmax=m+40*s)
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(name+'_e1_fov.png',bbox_inches='tight')
    plt.close()
    m,s=diff_e(cat2['e2'],cat['e2'],mask,1.)
    plt.hexbin(x,y,C=e2b[pixmask]-e2[pixmask],gridsize=(10,5),vmin=m-40*s,vmax=m+40*s)
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(name+'_e2_fov.png',bbox_inches='tight')
    plt.close()
    return 


fov_map_diff(table,fid,z4p,mask,'dz4p')
fov_map_diff(table,fid,los,mask,'dlos')


def get_sky_cf(cat,mask,cat2=None,diff=False,name='tmp'):
    gg = treecorr.GGCorrelation(nbins=20, min_sep=0.12, max_sep=120., sep_units='arcmin', bin_slop=0.1)
    if diff:
        tcat = treecorr.Catalog(g1=cat2['e1'][mask]-cat['e1'][mask], g2=cat2['e2'][mask]-cat['e2'][mask], ra=cat['ra'][mask], dec=cat['dec'][mask], ra_units='radians', dec_units='radians')
    else:
        tcat = treecorr.Catalog(g1=cat['e1'][mask], g2=cat['e2'][mask], ra=cat['ra'][mask], dec=cat['dec'][mask], ra_units='radians', dec_units='radians')
        if cat2 is not None:
            tcat2 = treecorr.Catalog(g1=cat2['e1'][mask], g2=cat2['e2'][mask], ra=cat2['ra'][mask], dec=cat2['dec'][mask], ra_units='radians', dec_units='radians')
    if (cat2 is None)|diff:
        gg.process(tcat)
    else:
        gg.process(tcat,tcat2)
    plt.errorbar(np.exp(gg.meanlogr)[gg.xip>0],gg.xip[gg.xip>0],yerr=np.sqrt(gg.varxi)[gg.xip>0],marker='o',ls='',color='b',label='xi+')
    plt.errorbar(np.exp(gg.meanlogr)[gg.xim>0],gg.xim[gg.xim>0],yerr=np.sqrt(gg.varxi)[gg.xim>0],marker='o',ls='',color='r',label='xi-')
    plt.errorbar(np.exp(gg.meanlogr)[gg.xip<0],-gg.xip[gg.xip<0],yerr=np.sqrt(gg.varxi)[gg.xip<0],marker='x',ls='',color='b')
    plt.errorbar(np.exp(gg.meanlogr)[gg.xim<0],-gg.xim[gg.xim<0],yerr=np.sqrt(gg.varxi)[gg.xim<0],marker='x',ls='',color='r')
    plt.xlabel('arcmin')
    plt.ylabel('xi (pos. val. o, neg. val. x')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper right',ncol=2, frameon=True,prop={'size':12})
    plt.tight_layout()
    plt.savefig(name+'_sky_cf.png',bbox_inches='tight')
    plt.close()
    return np.exp(gg.meanlogr),gg.xip,gg.xim,np.sqrt(gg.varxi)


def get_fov_cf(cat,mask,table,cat2=None,diff=False,name='tmp'):
    gg = treecorr.GGCorrelation(nbins=20, min_sep=0.0654, max_sep=65.45, bin_slop=0.1)
    if diff:
        e1,e2 = get_e_pix(cat,table,mask)
        e1b,e2b = get_e_pix(cat2,table,mask)
        pixmask = (e1!=0)&(e1b!=0)
        x,y=sca_pix_to_mm_fov(table['x'][pixmask],table['y'][pixmask],table['sca'][pixmask]-1)
        tcat = treecorr.Catalog(g1=e1b[pixmask]-e1[pixmask], g2=e2b[pixmask]-e2[pixmask], x=x, y=y)
    else:
        e1,e2 = get_e_pix(cat,table,mask)
        pixmaska = e1!=0
        x,y=sca_pix_to_mm_fov(table['x'][pixmaska],table['y'][pixmaska],table['sca'][pixmaska]-1)
        tcat = treecorr.Catalog(g1=e1[pixmaska], g2=e2[pixmaska], x=x, y=y)
        if cat2 is not None:
            e1b,e2b = get_e_pix(cat2,table,mask)
            pixmaskb = e1b!=0
            x,y=sca_pix_to_mm_fov(table['x'][pixmaskb],table['y'][pixmaskb],table['sca'][pixmaskb]-1)
            tcat2 = treecorr.Catalog(g1=e1b[pixmaskb], g2=e2b[pixmaskb], x=x, y=table['y'][pixmaskb])
    if (cat2 is None)|diff:
        gg.process(tcat)
    else:
        gg.process(tcat,tcat2)
    plt.errorbar(np.exp(gg.meanlogr)[gg.xip>0],gg.xip[gg.xip>0],yerr=np.sqrt(gg.varxi)[gg.xip>0],marker='.',ls='',color='b',label='xi+')
    plt.errorbar(np.exp(gg.meanlogr)[gg.xim>0],gg.xim[gg.xim>0],yerr=np.sqrt(gg.varxi)[gg.xim>0],marker='.',ls='',color='r',label='xi-')
    plt.errorbar(np.exp(gg.meanlogr)[gg.xip<0],-gg.xip[gg.xip<0],yerr=np.sqrt(gg.varxi)[gg.xip<0],marker='x',ls='',color='b')
    plt.errorbar(np.exp(gg.meanlogr)[gg.xim<0],-gg.xim[gg.xim<0],yerr=np.sqrt(gg.varxi)[gg.xim<0],marker='x',ls='',color='r')
    plt.xlabel('mm')
    plt.ylabel('xi (pos. val. o, neg. val. x')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper right',ncol=2, frameon=True,prop={'size':12})
    plt.tight_layout()
    plt.savefig(name+'_fov_cf.png',bbox_inches='tight')
    plt.close()
    return np.exp(gg.meanlogr),gg.xip,gg.xim,np.sqrt(gg.varxi)

sca_array,sca_array_norm,std = sca_map_diff(table,z4p['e1'],fid['e1'],mask,'de1z4p')
sca_array,sca_array_norm,std = sca_map_diff(table,z4p['e2'],fid['e2'],mask,'de2z4p')
sca_array,sca_array_norm,std = sca_map_diff(table,z4p['mean_hsm_psf_e1_sky'],fid['mean_hsm_psf_e1_sky'],mask,'dpsfe1z4p')
sca_array,sca_array_norm,std = sca_map_diff(table,z4p['mean_hsm_psf_e2_sky'],fid['mean_hsm_psf_e2_sky'],mask,'dpsfe2z4p')
sca_array,sca_array_norm,std = sca_map_diff(table,z4p['mean_hsm_psf_sigma'],fid['mean_hsm_psf_sigma'],mask,'dpsffwhmz4p')

sca_array,sca_array_norm,std = sca_map_diff(table,los['e1'],fid['e1'],mask,'de1los')
sca_array,sca_array_norm,std = sca_map_diff(table,los['e2'],fid['e2'],mask,'de2los')
sca_array,sca_array_norm,std = sca_map_diff(table,los['mean_hsm_psf_e1_sky'],fid['mean_hsm_psf_e1_sky'],mask,'dpsfe1los')
sca_array,sca_array_norm,std = sca_map_diff(table,los['mean_hsm_psf_e2_sky'],fid['mean_hsm_psf_e2_sky'],mask,'dpsfe2los')
sca_array,sca_array_norm,std = sca_map_diff(table,los['mean_hsm_psf_sigma'],fid['mean_hsm_psf_sigma'],mask,'dpsffwhmlos')

sca_array,sca_array_norm,std = sca_map(table,fid['e1'],mask,'e1')
sca_array,sca_array_norm,std = sca_map(table,fid['e2'],mask,'e2')
sca_array,sca_array_norm,std = sca_map(table,fid['mean_hsm_psf_e1_sky'],mask,'psfe1')
sca_array,sca_array_norm,std = sca_map(table,fid['mean_hsm_psf_e2_sky'],mask,'psfe2')
sca_array,sca_array_norm,std = sca_map(table,fid['mean_hsm_psf_sigma'],mask,'psffwhm')

get_sky_cf(fid,mask,name='fid')
get_sky_cf(z4p,mask,name='z4p')
get_sky_cf(los,mask,name='los')
get_sky_cf(fid,mask,cat2=z4p,name='cross_z4p')
get_sky_cf(fid,mask,cat2=los,name='cross_los')
get_sky_cf(fid,mask,cat2=z4p,diff=True,name='diff_z4p')
get_sky_cf(fid,mask,cat2=los,diff=True,name='diff_los')

get_fov_cf(fid,mask,table,name='fid')
get_fov_cf(z4p,mask,table,name='z4p')
get_fov_cf(los,mask,table,name='los')
get_fov_cf(fid,mask,table,cat2=z4p,name='cross_z4p')
get_fov_cf(fid,mask,table,cat2=los,name='cross_los')
get_fov_cf(fid,mask,table,cat2=z4p,diff=True,name='diff_z4p')
get_fov_cf(fid,mask,table,cat2=los,diff=True,name='diff_los')


fid=np.load('fid_psf.npy')
los=np.load('los_psf.npy')
z4p=np.load('z4p_psf.npy')

plt.imshow(fid,origin='lower',cmap=plt.get_cmap('inferno'))
plt.colorbar()
plt.xlabel('x (pix*8)')
plt.ylabel('y (pix*8)')
plt.title('FID PSF')
plt.tight_layout()
plt.savefig('/users/PCON0003/cond0083/wfirst_imsim/psf_fid.png', bbox_inches='tight')
plt.close()

plt.imshow(fid,origin='lower',cmap=plt.get_cmap('inferno'),norm = LogNorm())
plt.colorbar()
plt.xlabel('x (pix*8)')
plt.ylabel('y (pix*8)')
plt.title('log FID PSF')
plt.tight_layout()
plt.savefig('/users/PCON0003/cond0083/wfirst_imsim/psf_fid_log.png', bbox_inches='tight')
plt.close()

plt.imshow((z4p-fid)/np.max(fid),origin='lower',cmap=plt.get_cmap('inferno'))
plt.colorbar()
plt.xlabel('x (pix*8)')
plt.ylabel('y (pix*8)')
plt.title('Z4p PSF - FID PSF')
plt.tight_layout()
plt.savefig('/users/PCON0003/cond0083/wfirst_imsim/psf_z4p.png', bbox_inches='tight')
plt.close()

plt.imshow((los-fid)/np.max(fid),origin='lower',cmap=plt.get_cmap('inferno'))
plt.colorbar()
plt.xlabel('x (pix*8)')
plt.ylabel('y (pix*8)')
plt.title('LOS PSF - FID PSF')
plt.tight_layout()
plt.savefig('/users/PCON0003/cond0083/wfirst_imsim/psf_los.png', bbox_inches='tight')
plt.close()



import cProfile, pstats
pr = cProfile.Profile()
pr.enable()

import simulate as sim0
import fitsio as fio
import numpy as np
import galsim
import galsim.wfirst as wfirst
from astropy.time import Time
from numpy.lib.recfunctions import append_fields
sim = sim0.wfirst_sim('H158_i3.yaml')
sim.store = sim.init_galaxy()
store = fio.FITS(sim.out_path+'/'+sim.params['output_meds']+'_'+sim.params['filter']+'_truth_gal.fits')[-1].read()
table = fio.FITS('/fs/scratch/cond0083/wfirst_sim_out/test_H158_table.fits')[-1].read()
table=append_fields(table,['x','y'],[np.zeros(len(table)),np.zeros(len(table))],usemask=False)
fits    = fio.FITS(sim.params['dither_file'])[-1]
date    = fits.read(columns='date')
date    = Time(date,format='mjd').datetime
dfilter = fits.read(columns='filter')
dither  = fits.read(columns=['ra','dec','pa'])
for name in dither.dtype.names:
    dither[name] *= np.pi/180.

import cProfile, pstats
pr = cProfile.Profile()
pr.enable()
ud,uidx = np.unique(table['dither'],return_index=True)
for d,idx in tuple(zip(ud,uidx)):
    print d
    uinvd = np.where(table['dither']==d)[0]
    WCS = wfirst.getWCS(world_pos=galsim.CelestialCoord(ra=dither['ra'][d]*galsim.radians, dec=dither['dec'][d]*galsim.radians),PA=dither['pa'][d]*galsim.radians, date=date[d],SCAs=np.unique(table['sca'][uinvd]),PA_is_FPA=True)
    for i in uinvd:
        radec = galsim.CelestialCoord(store['ra'][table['gal'][i]]*galsim.radians,store['dec'][table['gal'][i]]*galsim.radians)
        xy = WCS[table['sca'][i]].toImage(radec)
        table['x'][i]=xy.x
        table['y'][i]=xy.y

pr.disable()
ps = pstats.Stats(pr).sort_stats('time')
ps.print_stats(20)



for i,p in enumerate(pix):
    print i,p
    obj = fio.FITS(sim.meds_filename(p))['object_data'].read(columns = 'number')
    try:
        os.remove('/fs/scratch/cond0083/wfirst_sim_out/test_los_H158_'+str(p)+'_obj.fits')
    except:
        pass
    fits = fio.FITS('/fs/scratch/cond0083/wfirst_sim_out/test_los_H158_'+str(p)+'_obj.fits','rw')
    fits.write(obj)
    fits.close()



import numpy as np
import fitsio as fio
import os
import yaml
params     = yaml.load(open('H158_los_i3.yaml'))
out_path = params['out_path']

def meds_filename(chunk):
    return out_path+'/'+params['output_meds']+'_'+params['filter']+'_'+str(chunk)+'.fits'

def get_totpix():
    import healpy as hp
    return np.unique(hp.ang2pix(params['nside'], np.pi/2.-store['dec'],store['ra'], nest=True))

def load_truth_gal():
    """
    Accepts a list of meds MultiExposureObject's and writes to meds file.
    """
    filename = out_path+'/'+params['output_meds']+'_'+params['filter']+'_truth_gal.fits'
    store = np.ones(1500000, dtype=[('rot','i2')]+[('e','i2')]+[('size','f4')]+[('z','f4')]+[('mag','f4')]+[('ra',float)]+[('dec',float)])
    out = fio.FITS(filename)[-1].read()
    store['rot']  = out['rot_angle']
    store['e']    = out['e_index']
    store['size'] = out['gal_size']
    store['z']    = out['redshift']
    store['mag']  = out['magnitude']
    store['ra']   = out['ra']
    store['dec']  = out['dec']
    return store

store = load_truth_gal()

from numpy.lib.recfunctions import append_fields
pix = get_totpix()
for i,p in enumerate(pix):
    obj = fio.FITS('/global/cscratch1/sd/troxel/wfirst_tmp/test_los_H158_'+str(p)+'_obj.fits')[-1].read(columns = 'number')
    for j in range (20):
        print i,p,j
        try:
            tmp=np.genfromtxt(meds_filename(p)+'.'+str(j)+'.main.txt',names=True)
            tmp2=np.genfromtxt(meds_filename(p)+'.'+str(j)+'.epoch.txt',names=True)
        except:
            print 'trouble reading ',meds_filename(p)+'.'+str(j)+'.main.txt'
            continue
        if len(tmp)==0:
            print 'empty file ',meds_filename(p)+'.'+str(j)+'.main.txt'
        if (i==0)&(j==0):
            main=np.empty(1500000,dtype=tmp.dtype)
            main=append_fields(main,['res','sige','chi2_pixel','flags'],[np.zeros(len(main)),np.zeros(len(main)),np.zeros(len(main)),np.zeros(len(main)).astype(int)],usemask=False)
        tmp_ind  = len(tmp)-np.unique(tmp['identifier'][::-1],return_index=True)[1]-1
        if not (len(tmp_ind)==len(tmp['identifier'])):
            tmp      = tmp[tmp_ind[0]:]
            tmp_idx  = tmp['identifier'][0]
            tmp2_idx = np.where(tmp_idx==tmp2['ID'])[0]
            tmp2_ind = tmp2_idx[np.where(np.diff(tmp2_idx)>1)[0][0]+1]
            tmp2     = tmp2[tmp2_ind:]
        u,uinv,ucnt=np.unique(tmp2['ID'].astype(int),return_inverse=True,return_counts=True)
        res = 1.-np.bincount(uinv,weights=tmp2['psf_fwhm']**2)/np.bincount(uinv,weights=tmp2['fwhm']**2)
        idx = obj[tmp['identifier'].astype(int)].astype(int)
        main[idx] = tmp
        main['identifier'][idx] = idx
        try:
            main['res'][idx]=res
        except:
            continue
        main['sige'][idx]=np.sqrt((tmp['covmat_1_1']+tmp['covmat_2_2'])/2)
        main['chi2_pixel'][idx]= -2 * tmp['likelihood'] / (tmp['stamp_size']**2 * tmp['n_exposure'] * (1-tmp['mean_mask_fraction']))
        main['flags'][idx] = (np.isnan(tmp['tilename']))&(tmp['snr']>18)&(main['res'][idx]>0.4)&(tmp['levmar_reason']!=3)&(tmp['levmar_reason']!=4)&(tmp['levmar_reason']!=5)&(tmp['levmar_reason']!=7)&(np.abs(tmp['e1'])>1e-4)&(np.abs(tmp['e2'])>1e-4)&(np.abs(tmp['e1'])<1)&(np.abs(tmp['e2'])<1)&(tmp['radius']<20)&(tmp['mean_rgpp_rp']>0)&(tmp['mean_rgpp_rp']<20)&(main['chi2_pixel'][idx]<10)&(np.abs(tmp['min_residuals'])<20)&(tmp['fails_rgpp_rp']==0)&(np.abs(tmp['ra_as'])<10)&(np.abs(tmp['dec_as'])<10)&(main['sige'][idx]<0.2)

fio.write(meds_filename('main'),main)


import simulate as sim0
import numpy as np
import galsim
import galsim.wfirst as wfirst
pix=571308
sim = sim0.wfirst_sim('H158.yaml')
sim.store = sim.init_galaxy()
sim.compile_tab()
gals = sim.get_pix_gals(pix)
gals = np.sort(gals)
tablemask = np.in1d(sim.table['gal'],gals,assume_unique=False)
gal_      = sim.table['gal'][tablemask]
sca_      = sim.table['sca'][tablemask]
dither_   = sim.table['dither'][tablemask]
dither,date_ = sim.setup_dither(dither_list = dither_)
cnt   = 0
dumps = 0
sim.PSF = wfirst.getPSF(SCAs=np.unique(sca_), 
                        approximate_struts=sim.params['approximate_struts'], 
                        n_waves=sim.params['n_waves'], 
                        logger=sim.logger, 
                        wavelength=sim.bpass,
                        extra_aberrations=sim.params['extra_aberrations'])
igal=0
gal=gals[0]
galmask = np.where(gal_==gal)
date = date_[galmask]
sca = sca_[galmask]
for idither,d in enumerate(dither[galmask]):
    sim.date = date[idither]
    sim.WCS = wfirst.getWCS(world_pos=galsim.CelestialCoord(ra=d['ra']*galsim.radians, 
                                                            dec=d['dec']*galsim.radians), 
                            PA=d['pa']*galsim.radians, 
                            date=sim.date,
                            SCAs=sca,
                            PA_is_FPA=True)
    fid = sim.draw_galaxy(gal,sca[idither],None)
    break

sim.params['extra_aberrations']=[0.,0.,0.,0.,0.005,0.,0.,0.,0.,0.,0.,0.]
sim.PSF = wfirst.getPSF(SCAs=np.unique(sca_), 
                        approximate_struts=sim.params['approximate_struts'], 
                        n_waves=sim.params['n_waves'], 
                        logger=sim.logger, 
                        wavelength=sim.bpass,
                        extra_aberrations=sim.params['extra_aberrations'])
for idither,d in enumerate(dither[galmask]):
    sim.date = date[idither]
    sim.WCS = wfirst.getWCS(world_pos=galsim.CelestialCoord(ra=d['ra']*galsim.radians, 
                                                            dec=d['dec']*galsim.radians), 
                            PA=d['pa']*galsim.radians, 
                            date=sim.date,
                            SCAs=sca,
                            PA_is_FPA=True)
    z4p = sim.draw_galaxy(gal,sca[idither],None)
    break

sim.params['extra_aberrations']=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
sim.params['los_motion']=0.015
for idither,d in enumerate(dither[galmask]):
    sim.date = date[idither]
    sim.WCS = wfirst.getWCS(world_pos=galsim.CelestialCoord(ra=d['ra']*galsim.radians, 
                                                            dec=d['dec']*galsim.radians), 
                            PA=d['pa']*galsim.radians, 
                            date=sim.date,
                            SCAs=sca,
                            PA_is_FPA=True)
    los = sim.draw_galaxy(gal,sca[idither],None)
    break



def linear_regression(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    rxy = np.mean((x-mean_x)*(y-mean_y))/np.sqrt(np.mean((x-mean_x)**2))/np.sqrt(np.mean((y-mean_y)**2))
    slope = rxy * np.std(y)/np.std(x) # beta in Wikipedia
    intercept = mean_y - slope * mean_x  # alpha in Wikipedia
    ndata = len(x)
    sum_residual_sqr = sum((y-(slope*x+intercept))**2)/(ndata-2)
    sig_y = np.sqrt(sum_residual_sqr)
    sig_slope_sqr = sum_residual_sqr/sum((x-mean_x)**2)
    sig_slope = np.sqrt(sig_slope_sqr)
    sig_intercept = sig_slope * np.sqrt(np.mean(x*x))
    return slope, intercept, sig_slope, sig_intercept, sig_y


m = np.zeros(200)
for i in range(200):
    print i
    mask2=np.random.choice(len(mask),len(mask),replace=True)
    p,c=curve_fit(func,truth['g1'][mask[mask2]],main['e1'][mask[mask2]],p0=(0.,0.),sigma=np.ones(np.sum(mask[mask2]))*10)
    m[i]=p[0]


>>> mask = (fid['flags']==1)&(z4p['flags']==1)&(los['flags']==1)
>>> maskfid = (fid['flags']==1)
>>> maskz4p = (z4p['flags']==1)
>>> masklos = (los['flags']==1)
>>> print 'number in fid',np.sum(maskfid)
number in fid 1156331
>>> print 'number in z4p',np.sum(maskz4p)
number in z4p 1405770
>>> print 'number in los',np.sum(masklos)
number in los 1157790
>>> print 'joint number',np.sum(mask)
joint number 1125504
>>> print 'fid mean e1',mean_e(fid['e1'],mask)
fid mean e1 (-0.00059979242012745293, 0.00014909012660272261)
>>> print 'fid mean e2',mean_e(fid['e2'],mask)
fid mean e2 (-0.0019922562148925803, 0.00014378424727308606)
>>> print 'z4p mean e1',mean_e(z4p['e1'],mask)
z4p mean e1 (-0.0020162938390795786, 0.00014922962234004814)
>>> print 'z4p mean e2',mean_e(z4p['e2'],mask)
z4p mean e2 (-0.0044493521657709284, 0.00014406139321663521)
>>> print 'los mean e1',mean_e(los['e1'],mask)
los mean e1 (0.0029352427442423112, 0.00014801806602264572)
>>> print 'los mean e2',mean_e(los['e2'],mask)
los mean e2 (-0.0021601965243566138, 0.00014248514813649466)
>>> print 'de1/dz4',diff_e(z4p['e1'],fid['e1'],mask,6.465)
^Cde1/dz4
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in diff_e
KeyboardInterrupt
>>> 
>>> filename = self.out_path+'/'+self.params['output_meds']+'_'+self.params['filter']+'_table.fits'
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'self' is not defined
>>> table = fio.FITS('/global/cscratch1/sd/troxel/wfirst_tmp/test_H158_table_wxy.fits')[-1].read()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/users/PCON0003/cond0083/.conda/envs/local/lib/python2.7/site-packages/fitsio/fitslib.py", line 351, in __init__
    raise IOError("File not found: '%s'" % filename)
IOError: File not found: '/global/cscratch1/sd/troxel/wfirst_tmp/test_H158_table_wxy.fits'
>>> mask = (fid['flags']==1)&(z4p['flags']==1)&(los['flags']==1)
>>> maskfid = (fid['flags']==1)
>>> maskz4p = (z4p['flags']==1)
>>> masklos = (los['flags']==1)
>>> print 'number in fid',np.sum(maskfid)
number in fid 1156331
>>> print 'number in z4p',np.sum(maskz4p)
number in z4p 1405770
>>> print 'number in los',np.sum(masklos)
number in los 1157790
>>> print 'joint number',np.sum(mask)
joint number 1125504
>>> print 'fid mean e1',mean_e(fid['e1'],mask)
fid mean e1 (-0.00059979242012745293, 0.00014909012660272261)
>>> print 'fid mean e2',mean_e(fid['e2'],mask)
fid mean e2 (-0.0019922562148925803, 0.00014378424727308606)
>>> print 'los mean e1',mean_e(los['e1'],mask)
los mean e1 (0.0029352427442423112, 0.00014801806602264572)
>>> print 'los mean e2',mean_e(los['e2'],mask)
los mean e2 (-0.0021601965243566138, 0.00014248514813649466)
>>> print 'de1/dlos',diff_e(los['e1'],fid['e1'],mask,0.015)
de1/dlos (0.23566901095798434, 0.0040068464468435323)
>>> print 'de2/dlos',diff_e(los['e2'],fid['e2'],mask,0.015)
de2/dlos (-0.011196020630935532, 0.0039049482491163466)
>>> 
>>> print 'de1 los',diff_e(los['e1'],fid['e1'],mask,1)
de1 los (0.0035350351643697649, 6.0102696702652982e-05)
>>> print 'de2 los',diff_e(los['e2'],fid['e2'],mask,1)
de2 los (-0.00016794030946403298, 5.8574223736745194e-05)


print 'de1/dlos',diff_e(los['e1'],fid['e1'],mask,0.015/np.sqrt(2))
print 'de2/dlos',diff_e(los['e2'],fid['e2'],mask,2*0.015/np.sqrt(2))
