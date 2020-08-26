mcal = fio.FITS('mcal-y1a1-combined-riz-unblind-v4-matched.fits')[-1].read(columns=['coadd_objects_id','e1','e2','R11','R22','flags_select','flags_select_1p','flags_select_1m','flags_select_2p','flags_select_2m'])
i3 = fio.FITS('y1a1-im3shape_v5_unblind_v2_matched_v4.fits')[-1].read(columns=['coadd_objects_id','e1','e2','m','weight','c1','c2','flags_select'])

for rmfile in ['y1a1_gold_1.0.3-d10-mof-001d_run_redmapper_v6.4.17_lgt5_desformat_catalog_members_sorted','y1a1_gold_1.0.3-d10-mof-001d_run_redmapper_v6.4.17_lgt20_desformat_catalog_members_sorted','y1a1_gold_1.0.3_wide+d10-mof-001d_run_redmapper_v6.4.17_lgt5_vl50_catalog_ubermembers_sorted']:
    new_file(rmfile)

def new_file(rmfile):
    rm = fio.FITS(rmfile+'.fit')[-1].read()
    mask = np.in1d(mcal['coadd_objects_id'],rm['ID'])
    tmp = append_fields(rm,  'e1', mcal['e1'][mask], usemask=False)
    tmp = append_fields(tmp, 'e2', mcal['e2'][mask], usemask=False)
    tmp = append_fields(tmp, 'R11', mcal['R11'][mask], usemask=False)
    tmp = append_fields(tmp, 'R22', mcal['R22'][mask], usemask=False)
    tmp = append_fields(tmp, 'flags_select', mcal['flags_select'][mask], usemask=False)
    tmp = append_fields(tmp, 'flags_select_1p', mcal['flags_select_1p'][mask], usemask=False)
    tmp = append_fields(tmp, 'flags_select_1m', mcal['flags_select_1m'][mask], usemask=False)
    tmp = append_fields(tmp, 'flags_select_2p', mcal['flags_select_2p'][mask], usemask=False)
    tmp = append_fields(tmp, 'flags_select_2m', mcal['flags_select_2m'][mask], usemask=False)
    fio.write(rmfile+'_mcal.fits.gz',tmp,clobber=True)
    rm = fio.FITS(rmfile+'.fit')[-1].read()
    mask = np.in1d(i3['coadd_objects_id'],rm['ID'])
    tmp = append_fields(rm,  'e1', i3['e1'][mask], usemask=False)
    tmp = append_fields(tmp, 'e2', i3['e2'][mask], usemask=False)
    tmp = append_fields(tmp, 'c1', i3['c1'][mask], usemask=False)
    tmp = append_fields(tmp, 'c2', i3['c2'][mask], usemask=False)
    tmp = append_fields(tmp, 'm', i3['m'][mask], usemask=False)
    tmp = append_fields(tmp, 'shapeweight', i3['weight'][mask], usemask=False)
    tmp = append_fields(tmp, 'flags_select', i3['flags_select'][mask], usemask=False)
    fio.write(rmfile+'_i3.fits.gz',tmp,clobber=True)

for name in tmp.dtype.names:
  fig.plot_methods.plot_hexbin_base(tmp[name],tmp['ra'],tmp['dec'],label=name,bins=1,part='_')


selection_1 = (tmp['ngmix_cm_t'] + 5. * tmp['ngmix_cm_t_err']) > 0.1
selection_2 = (tmp['ngmix_cm_t'] + 1. * tmp['ngmix_cm_t_err']) > 0.05
selection_3 = (tmp['ngmix_cm_t'] - 1. * tmp['ngmix_cm_t_err']) > 0.02
ext_mof = selection_1.astype(int) + selection_2.astype(int) + selection_3.astype(int)
mask2 = ext_mof==0



def skymap(ra0,dec0,val=None,nside=512,label=''):
    fig = plt.figure(figsize=(6.5,6))
    decsep=10
    rasep=10
    if val is None:
        bc, ra, dec, vertices = skm.getCountAtLocations(ra0, dec0, nside=nside, return_vertices=True)
    else:
        bc, ra, dec, vertices = skm.reduceAtLocations(ra0, dec0, val, nside=nside, return_vertices=True)
    # setup figure
    import matplotlib.cm as cm
    cmap = cm.YlOrRd
    ax = fig.add_subplot(111, aspect='equal')
    # setup map: define AEA map optimal for given RA/Dec
    proj = skm.createConicMap(ax, ra0, dec0, proj_class=skm.AlbersEqualAreaProjection)
    # add lines and labels for meridians/parallels (separation 5 deg)
    sep = 5
    meridians = np.arange(-90, 90+decsep, decsep)
    parallels = np.arange(0, 360+rasep, rasep)
    skm.setMeridianPatches(ax, proj, meridians, linestyle='-', lw=0.5, alpha=0.3, zorder=2)
    skm.setParallelPatches(ax, proj, parallels, linestyle='-', lw=0.5, alpha=0.3, zorder=2)
    skm.setMeridianLabels(ax, proj, meridians, loc="left", fmt=skm.pmDegFormatter)
    skm.setParallelLabels(ax, proj, parallels, loc="top")
    # add vertices as polygons
    vmin, vmax = np.percentile(bc,[10,90])
    poly = skm.addPolygons(vertices, proj, ax, color=bc, vmin=vmin, vmax=vmax, cmap=cmap, zorder=3, rasterized=True)
    # add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.0)
    cb = fig.colorbar(poly, cax=cax)
    cb.set_label('$n_g$ [arcmin$^{-2}$]')
    cb.solids.set_edgecolor("face")
    skm.addFootprint('DES', proj, ax, zorder=10, edgecolor='#2222B2', facecolor='None', lw=2)
    plt.savefig('footprint_'+label+'.png', bbox_inches='tight')
    plt.close()

for col in ['ngmix_flags','ngmix_cm_flags','flag_gold','flag_foreground']:
for col in ['flag_gold','flag_foreground']:
    maxbit = len(bin(np.max(tmp[col])))-2
    for val in range(maxbit):
        mask = (tmp[col] & 2**val) != 0
        if np.sum(mask)==0:
            continue
        skymap(tmp['ra'][mask],tmp['dec'][mask],label=col+'_'+str(val))



0.02 s per galaxy per pointing
2 min for psf per pointing
2 min to find galaxies in pointing




before wcs 520.711508036
after wcs 520.711951971
after gal stamp 520.712011099
after gal eff lambda 520.836121082
after gal draw 521.538080931
after add effects 521.539847136
before psf 521.539855957
after psf stamp 521.539906025
after psf eff lambda 521.663483143
after psf draw 522.339436054


for i in range(100):
    im = m.get_mosaic(i)
    print np.min(im),np.max(im)
    plt.imshow(im)
    plt.savefig('h_mosaic_'+str(i)+'.png')
    plt.close()


tmp0=load_obj('tmp0.pickle')
tmp1=load_obj('tmp1.pickle')
tmp3=load_obj('tmp2.pickle')
tmp2=load_obj('tmp2.pickle')
tmp3=load_obj('tmp3.pickle')
for i,var in enumerate([tmp0,tmp1,tmp2,tmp3]):
    plt.imshow(var.image.array)
    plt.colorbar()
    plt.savefig('tmp_'+str(i)+'.png')
    plt.close()

for n in ['0','a','a1','a2','a3','a4','a5','a6','b','c','d','e','f','g','h','i']:
  tmp=fio.FITS('tmp'+n+'.fits')[-1].read()
  plt.imshow(tmp)
  plt.colorbar()
  plt.savefig('tmp'+n+'.png')
  plt.close()

obj = galsim.Sersic(2, half_light_radius=1.)
obj = obj.rotate(0.*galsim.degrees)
obj = obj.shear(g1=0.0,g2=0.2)
sedpath = os.path.join(galsim.meta_data.share_dir, 'SEDs', 'CWW_Sbc_ext.sed')
galaxy_sed = galsim.SED(sedpath, wave_type='Ang', flux_type='flambda')
galaxy_sed = galaxy_sed.atRedshift(0.5)
galaxy_sed = galaxy_sed.withMagnitude(22,wfirst.getBandpasses()['F184']) * galsim.wfirst.collecting_area * galsim.wfirst.exptime
obj = obj * galaxy_sed



t0=time.time()
galsim.Convolve(obj, psf)
print t0-time.time()

sca=1
t0=time.time()

print t0-time.time()
wfirst.getWCS(world_pos=galsim.CelestialCoord(ra=100.*galsim.degrees, dec=-50*galsim.degrees), SCAs=sca+1, PA_is_FPA=True)[sca+1]
print t0-time.time()
psf = wfirst.getPSF(SCAs=sca+1, approximate_struts=True, n_waves=10, wavelength=wfirst.getBandpasses(AB_zeropoint=True)['F184'])[sca+1]
print t0-time.time()




import fitsio as fio
for i in range(18):
  tmp=fio.FITS('test_H158_image_22537.fits')[i].read()
  plt.imshow(tmp,interpolation='none')
  plt.colorbar()
  plt.savefig('test_sca_'+str(i)+'.png')
  plt.close()


tmp=fio.FITS('/users/PCON0003/cond0080/src/REM/out_test_H158_0.fits')[1].read()
meds = fio.FITS('/fs/scratch/cond0083/wfirst_sim_out/test_H158_0.fits')[1].read()
truth = fio.FITS('/fs/scratch/cond0083/wfirst_sim_out/test_H158_truth_gal.fits')[1].read()
mask=(tmp['flags']==0)&(tmp['gauss_flags']==0)&(tmp['psf_flags']==0)&(tmp['gauss_s2n_w']>10)&(tmp['gauss_T_r']/tmp['gauss_psf_T_r']>0.7)
match = tmp['id'][mask]
truth=truth[match]
tmp=tmp[mask]

mask = np.ones(len(tmp)).astype(bool)
mask = tmp['nimage_tot']>1

plt.hist(tmp['gauss_g'][mask,0],bins=500,histtype='step',label='e1')
plt.hist(tmp['gauss_g'][mask,1],bins=500,histtype='step',label='e2')
plt.savefig('tmp.png')
plt.close()


mask1 = (truth['g1']<-0.1)&(truth['g1']>-0.3)
mask2 = (truth['g2']<-0.1)&(truth['g2']>-0.3)
plt.hist(tmp['gauss_g'][:,0][mask1],bins=500,histtype='step',label='e1')
plt.hist(tmp['gauss_g'][:,1][mask2],bins=500,histtype='step',label='e2')
plt.savefig('tmp1.png')
plt.close()


mask1 = (truth['g1']>-0.1)&(truth['g1']<0.1)
mask2 = (truth['g2']>-0.1)&(truth['g2']<0.1)
plt.hist(tmp['gauss_g'][:,0][mask1],bins=500,histtype='step',label='e1')
plt.hist(tmp['gauss_g'][:,1][mask2],bins=500,histtype='step',label='e2')
plt.savefig('tmp2.png')
plt.close()


mask1 = (truth['g1']<0.3)&(truth['g1']>0.1)
mask2 = (truth['g2']<0.3)&(truth['g2']>0.1)
plt.hist(tmp['gauss_g'][:,0][mask1],bins=500,histtype='step',label='e1')
plt.hist(tmp['gauss_g'][:,1][mask2],bins=500,histtype='step',label='e2')
plt.savefig('tmp3.png')
plt.close()

from scipy.optimize import curve_fit
def lin_fit(x,y,sig):
  """
  Find linear fit with errors for two arrays.
  """
  def func(x,m,b):
    return m*x+b
  params=curve_fit(func,x,y,p0=(0.,0.),sigma=sig)
  m,b=params[0]
  merr,berr=np.sqrt(np.diagonal(params[1]))
  return m,b,merr,berr



import pickle
import fitsio as fio
import glob
def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)


for name in glob.glob('/fs/scratch/cond0083/wfirst_sim_out/test_H158_image_*.pickle'):
    im,wt=load_obj(name)
    plt.imshow(im.array,vmax=200)
    plt.savefig(name+'.png',dpi=500)
    plt.close()

for n_wave in [2,4,8]:
    for sca in range(18):
        for oversample in [1,2,4,8,16,32]:
            for filter_ in filter_dither_dict.keys():
                if not os.path.exists('psf_test_'+str(n_wave)+'_'+str(oversample)+'_'+str(sca)+'.pickle'):
                    print n_wave,sca,oversample



fits1 = '/users/PCON0003/cond0083/cosmosis5/test.fits'
xip1   = tp.TwoPointFile.from_fits(fits1,covmat_name=None).get_spectrum('xip')
xim1   = tp.TwoPointFile.from_fits(fits1,covmat_name=None).get_spectrum('xim')

fits2 = '/users/PCON0003/cond0083/cosmosis5k/test.fits'
xip2   = tp.TwoPointFile.from_fits(fits2,covmat_name=None).get_spectrum('xip')
xim2   = tp.TwoPointFile.from_fits(fits2,covmat_name=None).get_spectrum('xim')

for i in range(10):
    val = ((xip2.value-xip1.value)/xip1.value)[i*1000:(i+1)*1000]
    plt.plot(xip1.angle[i*1000:(i+1)*1000][val>=0],val[val>=0],ls='-')
    plt.plot(xip1.angle[i*1000:(i+1)*1000][val<0],-val[val<0],ls=':')

plt.xscale('log')
plt.yscale('log')
plt.savefig('xip.png')
plt.close()

for i in range(10):
    val = ((xim2.value-xim1.value)/xim1.value)[i*1000:(i+1)*1000]
    plt.plot(xip1.angle[i*1000:(i+1)*1000][val>=0],val[val>=0],ls='-')
    plt.plot(xip1.angle[i*1000:(i+1)*1000][val<0],-val[val<0],ls=':')

plt.xscale('log')
plt.yscale('log')
plt.savefig('xim.png')
plt.close()


def matched_metacal_cut_live():
    cuts=catalog.CatalogMethods.add_cut(np.array([]),'flags',catalog.noval,0,catalog.noval)
    cuts=catalog.CatalogMethods.add_cut(cuts,'snr',10.,catalog.noval,100.)
    cuts=catalog.CatalogMethods.add_cut(cuts,'rgp',np.log(np.sqrt(0.5)),catalog.noval,catalog.noval)
    # cuts=catalog.CatalogMethods.add_cut(cuts,'psf1',-0.02,catalog.noval,0.03)
    return cuts

import src.config as config
config.cfg['lbins']=10
import src.catalog as catalog
catdir = '/global/project/projectdirs/des/wl/desdata/users/esheldon/matched-catalogs/'
cols = ['coadd','e1','e2','snr','flags','mask_frac','psf1','psf2','psfsize','size','R11','R22','nimage_tot_r','nimage_use_r']
# mcala = catalog.CatalogStore('t001a',cattype='mcal',catfile=catdir+'y3v02-mcal-t001a-combined-blind-v1.fits',goldfile=catdir+'y3v02-mcal-t001a-combined-blind-v1.fits',cols=cols,goldcols=[])
# mcalb = catalog.CatalogStore('t002a',cattype='mcal',catfile=catdir+'y3v02-mcal-t002a-combined-blind-v1.fits',goldfile=catdir+'y3v02-mcal-t002a-combined-blind-v1.fits',cols=cols,goldcols=[])
# mcalc = catalog.CatalogStore('t003a',cattype='mcal',catfile=catdir+'y3v02-mcal-t003a-combined-blind-v1.fits',goldfile=catdir+'y3v02-mcal-t003a-combined-blind-v1.fits',cols=cols,goldcols=[])
# mcald = catalog.CatalogStore('t003b',cattype='mcal',catfile=catdir+'y3v02-mcal-t003b-combined-blind-v1.fits',goldfile=catdir+'y3v02-mcal-t003b-combined-blind-v1.fits',cols=cols,goldcols=[])
mcal = catalog.CatalogStore('mcal_001_0',cattype='mcal',catfile=catdir+'y3v02-mcal-001-0-combined-blind-v1.fits',goldfile=catdir+'y3v02-mcal-001-0-combined-blind-v1.fits',cols=cols,goldcols=[])


import src.sys_split as sys_split
import src.lin as lin
import numpy as np

for cat in [mcala,mcalb,mcalc]:
    cols = ['snr','mask_frac','psf1','psf2','psfsize','size','rgp','e1','e2','nimage_tot_r','nimage_use_r']
    cat.rgp = cat.size/cat.psfsize
    cat.rgp_1p = cat.size_1p/cat.psfsize
    cat.rgp_2p = cat.size_2p/cat.psfsize
    cat.rgp_1m = cat.size_1m/cat.psfsize
    cat.rgp_2m = cat.size_2m/cat.psfsize
    cat.size = np.log(np.sqrt(cat.size))
    cat.rgp = np.log(np.sqrt(cat.rgp))
    cat.livecuts = matched_metacal_cut_live()
    lin.summary_stats.i3_flags_vals_check(cat,flags=['flags'])
    mask = catalog.CatalogMethods.get_cuts_mask(cat,full=False)
    mask = np.in1d(np.arange(len(cat.coadd)),mask)
    # lin.summary_stats.val_stats(cat,cols = cols, mask = mask)
    lin.hist.hist_tests(cat, cols=cols,mask = mask)
    lin.hist.hist_2D_tests(cat,colsx=cols,colsy=cols,mask = mask)
    cols = ['snr','mask_frac','psf1','psf2','psfsize','size','rgp']
    sys_split.split.cat_splits_lin_e(cat,cols=cols)


# lin.summary_stats.val_stats(mcala,cols = cols, mask = mask)
lin.hist.hist_tests(mcala, cols=cols,mask = mask)
lin.hist.hist_2D_tests(mcala,colsx=cols,colsy=cols,mask = mask)


zm = np.loadtxt('../../cosmosis5/demo7_output/matter_power_lin/z.txt')
pk = np.loadtxt('../../cosmosis5/demo7_output/matter_power_lin/p_k.txt')
zg = np.loadtxt('../../cosmosis5/demo7_output/growth_parameters/z.txt')
d = np.loadtxt('../../cosmosis5/demo7_output/growth_parameters/d_z.txt')

zmk = np.loadtxt('matter_power_lin/z.txt')
pkk = np.loadtxt('matter_power_lin/p_k.txt')
zgk = np.loadtxt('growth_parameters/z.txt')
dk = np.loadtxt('growth_parameters/d_z.txt')

plt.plot(zg+.2,(d/d[0])**2,color='r',ls='-')
plt.plot(zgk+.4,(dk/dk[0])**2,color='r',ls=':')
plt.plot(zm,pk[:,0]/pk[0,0],color='b',ls='-')
plt.plot(zmk,pkk[:,0]/pkk[0,0],color='b',ls=':')
plt.yscale('log')
plt.savefig('tmp.png')
plt.close()

def hsm(im, psf=None, wt=None):
    MAX_CENTROID_SHIFT = 999.
    BAD_MEASUREMENT = 2
    CENTROID_SHIFT = 1
    out = np.zeros(1,dtype=[('e1','f4')]+[('e2','f4')]+[('T','f4')]+[('dx','f4')]+[('dy','f4')]+[('flag','i2')])
    try:
        if psf is not None:
            shape_data = galsim.hsm.EstimateShear(im, psf, weight=wt, strict=False)
        else:
            shape_data = im.FindAdaptiveMom(weight=wt, strict=False)
    except:
        # print ' *** Bad measurement (caught exception).  Mask this one.'
        out['flag'] |= BAD_MEASUREMENT
        return out
    if shape_data.moments_status != 0:
        # print 'status',shape_data.moments_status
        out['flag'] |= BAD_MEASUREMENT
    out['dx'] = shape_data.moments_centroid.x - im.trueCenter().x
    out['dy'] = shape_data.moments_centroid.y - im.trueCenter().y
    if out['dx']**2 + out['dy']**2 > MAX_CENTROID_SHIFT**2:
        # print 'bad centroid',shape_data.moments_centroid.x,shape_data.moments_centroid.y
        out['flag'] |= CENTROID_SHIFT
    # Account for the image wcs
    if im.wcs.isPixelScale():
        out['e1'] = shape_data.observed_shape.g1
        out['e2'] = shape_data.observed_shape.g2
        out['T']  = 2 * shape_data.moments_sigma**2 * im.scale**2
    else:
        e1 = shape_data.observed_shape.e1
        e2 = shape_data.observed_shape.e2
        s = shape_data.moments_sigma
        jac = im.wcs.jacobian(im.trueCenter())
        M = np.matrix( [[ 1 + e1, e2 ], [ e2, 1 - e1 ]] ) * s*s
        J = jac.getMatrix()
        M = J * M * J.T
        scale = np.sqrt(M/2./s/s)
        e1 = (M[0,0] - M[1,1]) / (M[0,0] + M[1,1])
        e2 = (2.*M[0,1]) / (M[0,0] + M[1,1])
        out['T'] = M[0,0] + M[1,1]
        shear = galsim.Shear(e1=e1, e2=e2)
        out['e1'] = shear.g1
        out['e2'] = shear.g2
    return out

import simulate as sim0
import cPickle as pickle
import galsim
import numpy as np
sim=sim0.wfirst_sim('H158.yaml')
# sim.accumulate_stamps(0,ignore_missing_files=False)
def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

filename = sim.out_path+'/'+sim.params['output_meds']+'_'+sim.params['filter']+'_stamps_'+str(0)+'_'+str(0)+'_'+str(0)+'.pickle'
gal_exps_,wcs_exps_,wgt_exps_,psf_exps_,dither_list_,sca_list_,hsm_list_ = load_obj(filename)

for i0,i in enumerate(gal_exps_.keys()):
    out0 = sim0.hsm(psf_exps_[i][0])
    for c in [1,2,3,4,5,6,7,8]:
        b = galsim.BoundsI(xmin=1-(2**c-256)/2,
                            ymin=1-(2**c-256)/2,
                            xmax=256+(2**c-256)/2,
                            ymax=256+(2**c-256)/2)
        out = sim0.hsm(psf_exps_[i][0][b])
        if out['flag']==0:
            print 2**c,i0,[out[name][0]/out0[name][0] for name in ['e1','e2','T','dx','dy']]
    if i0>327:
        break

hsm(gal_exps_[490667][0], psf=psf_exps_[490667][0], wt=wgt_exps_[490667][0])
shape_data = galsim.hsm.EstimateShear(gal_exps_[490667][0], psf_exps_[490667][0], weight=wgt_exps_[490667][0], strict=False)



b = galsim.BoundsI(xmin=97,
                    ymin=97,
                    xmax=160,
                    ymax=160)
 

tmp = fio.FITS('mcal-y1a1-combined-riz-unblind-v4-matched.fits')[-1].read(columns=['coadd_objects_id', 'e1','e2', 'R11', 'R22', 'ra', 'dec', 'tilename', 'flags_select', 'flags_select_1p', 'flags_select_1m', 'flags_select_2p', 'flags_select_2m','covmat_0_0','covmat_1_1'])

z    = fio.FITS('mcal-y1a1-combined-griz-blind-v3-matched_BPZ.fits')[-1].read(columns='MEAN_Z')
z_1p = fio.FITS('mcal-y1a1-combined-griz-blind-v3-matched_BPZ_1p.fits')[-1].read(columns='MEAN_Z')
z_1m = fio.FITS('mcal-y1a1-combined-griz-blind-v3-matched_BPZ_1m.fits')[-1].read(columns='MEAN_Z')
z_2p = fio.FITS('mcal-y1a1-combined-griz-blind-v3-matched_BPZ_2p.fits')[-1].read(columns='MEAN_Z')
z_2m = fio.FITS('mcal-y1a1-combined-griz-blind-v3-matched_BPZ_2m.fits')[-1].read(columns='MEAN_Z')

mask    = (tmp['flags_select']==0)&(tmp['dec']<-35)
mask_1p = (tmp['flags_select_1p']==0)&(tmp['dec']<-35)
mask_1m = (tmp['flags_select_1m']==0)&(tmp['dec']<-35)
mask_2p = (tmp['flags_select_2p']==0)&(tmp['dec']<-35)
mask_2m = (tmp['flags_select_2m']==0)&(tmp['dec']<-35)
umask   = mask|mask_1p|mask_1m|mask_2p|mask_2m

tmp     = tmp[umask]
mask    = mask[umask]
mask_1p = mask_1p[umask]
mask_1m = mask_1m[umask]
mask_2p = mask_2p[umask]
mask_2m = mask_2m[umask]

z    = z[umask]
z_1p = z_1p[umask]
z_1m = z_1m[umask]
z_2p = z_2p[umask]
z_2m = z_2m[umask]

ran = np.random.choice(np.arange(len(tmp)),500000,replace=False)
km  = km0.kmeans_sample(np.vstack((tmp['ra'][ran],tmp['dec'][ran])).T, 200, maxiter=100)
jk  = km.find_nearest(np.vstack((tmp['ra'],tmp['dec'])).T)
np.save('jk_mcal_y1.npy',jk)
jk=np.load('jk_mcal_y1.npy')

a=np.zeros((4,200))
b=np.zeros((4,200))
c=np.zeros((4,200))
d=np.zeros((4,200))
zrange = [0.2,0.43,0.63,0.9,1.3]
sigma_ec_orig = [0.24768695494279713, 0.28168178843826547, 0.27215546947259356, 0.275742843736973]
for i in range(4):
    mask_    = (z>zrange[i])&(z<=zrange[i+1])
    mask_1p_ = (z_1p>zrange[i])&(z_1p<=zrange[i+1])
    mask_1m_ = (z_1m>zrange[i])&(z_1m<=zrange[i+1])
    mask_2p_ = (z_2p>zrange[i])&(z_2p<=zrange[i+1])
    mask_2m_ = (z_2m>zrange[i])&(z_2m<=zrange[i+1])
    for ij,j in enumerate(np.unique(jk)):
        print i,ij
        jkmask   = jk==j
        mask0 = mask_&mask&jkmask
        m1  = np.mean(tmp['R11'][mask0])
        m2  = np.mean(tmp['R22'][mask0])
        m1  += (np.mean(tmp['e1'][mask_1p_&mask_1p&jkmask]) - np.mean(tmp['e1'][mask_1m_&mask_1m&jkmask])) / (0.02)
        m2  += (np.mean(tmp['e2'][mask_2p_&mask_2p&jkmask]) - np.mean(tmp['e2'][mask_2m_&mask_2m&jkmask])) / (0.02)
        s   = (m1+m2)/2.
        var = tmp['covmat_0_0'][mask0] + tmp['covmat_1_1'][mask0]
        var[var>2] = 2.
        a[i,ij] = np.sum(((tmp['e1'][mask0]**2 + tmp['e2'][mask0]**2 - var)/2)**2)
        b[i,ij] = s**4*np.sum(mask0)**2
        c[i,ij] = s*np.sum(mask0)
        d[i,ij] = np.sum((s**2 * sigma_ec_orig[i] + var/2.))


sigma_ec_full = np.sum(a,axis=1) / (np.sum(b,axis=1)) 
neff_full = (np.sqrt(sigma_ec_full) * np.sum(c,axis=1)**2) / np.sum(d,axis=1)/1321./60/60
tot_full = sigma_ec_full/neff_full**2
sigma_ec = np.zeros((4,200))
neff     = np.zeros((4,200))
tot     = np.zeros((4,200))
for ij in range(200):
    sigma_ec[:,ij] =  (np.sum(a,axis=1)-a[:,ij]) / ((np.sum(b,axis=1)-b[:,ij])) 
    neff[:,ij] = (np.sqrt(sigma_ec[:,ij]) * (np.sum(c,axis=1)-c[:,ij])**2) / (np.sum(d,axis=1)-d[:,ij])/1321./60/60
    tot[:,ij] = sigma_ec[:,ij]/neff[:,ij]**2

sigma_ec_cov = np.zeros(4)
neff_cov     = np.zeros(4)
tot_cov      = np.zeros(4)
for i in range(4):
    sigma_ec_cov[i]=np.sum((sigma_ec[i,:]-np.mean(sigma_ec[i,:]))**2)*(200.-1.)/200.
    neff_cov[i]=np.sum((neff[i,:]-np.mean(neff[i,:]))**2)*(200.-1.)/200.
    tot_cov[i]=np.sum((tot[i,:]-np.mean(tot[i,:]))**2)*(200.-1.)/200.


np.sqrt(sigma_ec_cov)/sigma_ec_full
np.sqrt(neff_cov)/neff_full
np.sqrt(tot_cov)/tot_full


changing sigmae each jk region
>>> np.sqrt(sigma_ec_cov)/sigma_ec_full
array([ 0.0007891 ,  0.00110061,  0.0012034 ,  0.0017066 ])
>>> np.sqrt(neff_cov)/neff_full
array([ 0.00963071,  0.00877213,  0.00860968,  0.00939463])
>>> np.sqrt(tot_cov)/tot_full
array([ 0.0100776 ,  0.00928545,  0.00945519,  0.0106562 ])

0.0016, 0.0036
0.0001/0.0025



def get_bins(x,nbins):
    xmin = x.min()
    if xmin<0:
        cumsum = (x+xmin).cumsum() / (x+xmin).sum()
    else:
        cumsum = x.cumsum() / x.sum()
    return np.searchsorted(cumsum, np.linspace(0, 1, nbins+1, endpoint=True))



gold = fio.FITS('/project/projectdirs/des/jderose/catalog/mergedcats/y3/buzzard/flock/buzzard-0/a/Buzzard_v1.6_Y3_gold.fits')[-1].read()
pz = fio.FITS('/project/projectdirs/des/jderose/catalog/mergedcats/y3/buzzard/flock/buzzard-0/a/Buzzard_v1.6_Y3_pz.fits')[-1].read(columns='mode-z')

gr   = gold['mag_g']-gold['mag_r']
ri   = gold['mag_r']-gold['mag_i']
iz   = gold['mag_i']-gold['mag_z']
mask = (gr>-1)&(gr<4)&(ri>-1)&(ri<4)&(iz>-1)&(iz<4)
gold = gold[mask]
pz   = pz[mask]
gr   = gold['mag_g']-gold['mag_r']
ri   = gold['mag_r']-gold['mag_i']
iz   = gold['mag_i']-gold['mag_z']
i    = gold['mag_i']*1.
redshift = gold['redshift']*1.

s  = np.argsort(gr)
gr = gr[s]
ri = ri[s]
iz = iz[s]
i  =  i[s]
pz = pz[s]
redshift = redshift[s]

t0=time.time()
slices=20
store = np.zeros((7,slices,slices,slices,slices))
edges = get_bins(1.*np.arange(len(gr)),slices)
for gr_ in range(slices):
    tmp_z = redshift[edges[gr_]:edges[gr_+1]]
    tmp_pz = pz[edges[gr_]:edges[gr_+1]]
    ri_s = np.argsort(ri[edges[gr_]:edges[gr_+1]])
    iz_s = np.argsort(iz[edges[gr_]:edges[gr_+1]])
    i_s  = np.argsort(i[edges[gr_]:edges[gr_+1]])
    print gr_
    edges = get_bins(1.*np.arange(len(tmp)),slices)
    for ri_ in range(slices):
        ri_slice = ri_s[edges[ri_]:edges[ri_+1]]
        ri_slice = np.intersect1d(ri_slice,gr_slice,assume_unique=True)
        if len(ri_slice) == 0:
            continue
        for iz_ in range(slices):
            iz_slice=iz_s[edges[iz_]:edges[iz_+1]]
            iz_slice = np.intersect1d(iz_slice,ri_slice,assume_unique=True)
            if len(iz_slice) == 0:
                continue
            for i_ in range(slices):
                i_slice = i_s[edges[i_]:edges[i_+1]]
                i_slice = np.intersect1d(i_slice,iz_slice,assume_unique=True)
                if len(i_slice) == 0:
                    continue
                store[0,gr_,ri_,iz_,i_] = len(i_slice)
                tmp_z_ = tmp_z[i_slice]
                tmp_pz_ = tmp_pz[i_slice]
                store[1,gr_,ri_,iz_,i_] = np.mean(tmp_z_)
                store[2,gr_,ri_,iz_,i_] = np.std(tmp_z_)
                store[3,gr_,ri_,iz_,i_] = np.mean(tmp_pz_)
                store[4,gr_,ri_,iz_,i_] = np.std(tmp_pz_)
                store[5,gr_,ri_,iz_,i_] = np.mean(tmp_pz_-tmp_z_)
                store[6,gr_,ri_,iz_,i_] = np.std(tmp_pz_-tmp_z_)
                print time.time()-t0,gr_,ri_,iz_,i_,store[:,gr_,ri_,iz_,i_]

import numpy as np
import twopoint as tp

for i in range(4):
    for j in range(4):
        try:
            theta= np.loadtxt('2pt_'+str(i)+'_'+str(j)+'_0.txt')[:,0]
            new  = np.loadtxt('2pt_'+str(i)+'_'+str(j)+'_0.txt')[:,1]
            new2 = np.loadtxt('2pt_'+str(i)+'_'+str(j)+'_0.txt')[:,2]
            old  = tp.TwoPointFile.from_fits('../deswlwg-y1cosmicshear/data/des-y1/mcal/2pt_NG.fits').get_spectrum('xip').get_pair(i+1,j+1)[1]
            old2 = tp.TwoPointFile.from_fits('../deswlwg-y1cosmicshear/data/des-y1/mcal/2pt_NG.fits').get_spectrum('xim').get_pair(i+1,j+1)[1]
            sig  = tp.TwoPointFile.from_fits('../deswlwg-y1cosmicshear/data/des-y1/mcal/2pt_NG.fits').get_spectrum('xip').get_error(i+1,j+1)
            sig2  = tp.TwoPointFile.from_fits('../deswlwg-y1cosmicshear/data/des-y1/mcal/2pt_NG.fits').get_spectrum('xim').get_error(i+1,j+1)
            niall  = tp.TwoPointFile.from_fits('/users/PCON0003/cond0083/des-mpp/covariance_tests/all2pt_bs0.05_v0/2pt_wpc_v0.fits').get_spectrum('xip').get_pair(i+1,j+1)[1]
            niall2 = tp.TwoPointFile.from_fits('/users/PCON0003/cond0083/des-mpp/covariance_tests/all2pt_bs0.05_v0/2pt_wpc_v0.fits').get_spectrum('xim').get_pair(i+1,j+1)[1]
            plt.plot(theta,new*theta,ls='',color='b',marker='.')
            plt.errorbar(theta*1.1,old*theta,yerr=sig,ls='',color='r',marker='.')
            plt.xscale('log')
            plt.savefig('comp_xip'+str(i)+'_'+str(j)+'.png')
            plt.close()
            plt.plot(theta,new2*theta,ls='',color='b',marker='.')
            plt.errorbar(theta*1.1,old2*theta,yerr=sig2,ls='',color='r',marker='.')
            plt.xscale('log')
            plt.savefig('comp_xim'+str(i)+'_'+str(j)+'.png')
            plt.close()
        except:
            pass
        try:
            sig3  = tp.TwoPointFile.from_fits('../deswlwg-y1cosmicshear/data/des-y1/mcal/2pt_NG.fits').get_spectrum('gammat').get_error(i+1,j+1)
            old3 = tp.TwoPointFile.from_fits('../deswlwg-y1cosmicshear/data/des-y1/mcal/2pt_NG.fits').get_spectrum('gammat').get_pair(i+1,j+1)[1]
            new3 = np.loadtxt('2pt_'+str(i)+'_'+str(j)+'_1.txt')[:,1]
            niall3 = tp.TwoPointFile.from_fits('/users/PCON0003/cond0083/des-mpp/covariance_tests/all2pt_bs0.05_v0/2pt_wpc_v0.fits').get_spectrum('gammat').get_pair(i+1,j+1)[1]
            plt.plot(theta,new3*theta,ls='',color='b',marker='.')
            plt.errorbar(theta*1.1,old3*theta,yerr=sig3,ls='',color='r',marker='.')
            plt.xscale('log')
            plt.savefig('comp_gammat'+str(i)+'_'+str(j)+'.png')
            plt.close()
        except:
            pass

