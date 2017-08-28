# imports, etc.
import galsim
import galsim.wfirst as wf
import datetime
import numpy as np
import matplotlib.pyplot as plt
from radec_to_chip import *


#create coords.txt
ralims = [20,31]
declims = [-85.3,-84.7]
step = 1./60.
ra=np.arange((ralims[1]-ralims[0])/step)/((ralims[1]-ralims[0])/step)*(ralims[1]-ralims[0])+ralims[0]
dec=np.arange((declims[1]-declims[0])/step)/((declims[1]-declims[0])/step)*(declims[1]-declims[0])+declims[0]
x,y=np.meshgrid(ra,dec)
x=x.flatten()
y=y.flatten()
np.savetxt('coords.txt',np.vstack((x,y)).T)


ra_cen = 25.817233259290145 # degrees
dec_cen = -85.0 # degrees
ra_cen_rad = ra_cen*np.pi/180. # radians
dec_cen_rad = dec_cen*np.pi/180. # radians
date = datetime.datetime(2025, 1, 12)
seed = 314159
testing = False
if testing:
    n_rand = 10000
    output = 'show'
else:
    n_rand = 100000
    output = 'wfirst_wcs_test_675_dec%.3f.png'%dec_cen

fpa_center = galsim.CelestialCoord(ra=ra_cen*galsim.degrees, dec=dec_cen*galsim.degrees)

# get WFIRST WCS pointed with a specific position and PA
pa = wf.bestPA(fpa_center, date)

print 'Getting WCS for position, date, FPA PA:',fpa_center, date, pa
if pa is None:
    raise RuntimeError("Cannot look here on this date!")
wcs = wf.getWCS(fpa_center, PA=pa, date=date, PA_is_FPA=True)

# generate a bunch of random points there
print 'Generating random points'
ud = galsim.UniformDeviate(seed)
ra_vals = []
dec_vals = []
for i in range(n_rand):
    if i % 1000 == 0: print '   ',i
    ra_vals.append(ra_cen + (ud() - 0.5)/np.cos((dec_cen*galsim.degrees)/galsim.radians))
    dec_vals.append(dec_cen + ud() - 0.5)
ra_vals = np.array(ra_vals)
dec_vals = np.array(dec_vals)

# Find the SCAs from Chris's code (Python version) for the same points
sca_ch = radec_to_chip(ra_cen_rad, dec_cen_rad, pa,
                       ra_vals*np.pi/180., dec_vals*np.pi/180.)
m2 = sca_ch>0
ra_vals_ch = ra_vals[m2]
dec_vals_ch = dec_vals[m2]
sca_vals_ch = sca_ch[m2]

# Find the SCAs
sca_vals = np.zeros_like(ra_vals)-1
print 'Finding SCA value for random points'
for i in range(n_rand):
    if i % 1000 == 0: print '   ',i
    sca = wf.findSCA(wcs, galsim.CelestialCoord(ra=ra_vals[i]*galsim.degrees,
                                                dec=dec_vals[i]*galsim.degrees))
    if sca is not None:
        sca_vals[i] = sca
m = sca_vals > 0
ra_vals = ra_vals[m]
dec_vals = dec_vals[m]
sca_vals = sca_vals[m]
print n_rand, len(ra_vals)

# make a plot showing the points colored by their WCS, also with original pointing position shown

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(121)
sc=ax.scatter(ra_vals, dec_vals, c=sca_vals, s=1, 
              lw=0, cmap=plt.cm.viridis)
# The previous line is a change to make defaults like the newer matplotlib
# since the Ohio Supercomputer Center comp seems to have an older mpl by default
ax.scatter([ra_cen], [dec_cen], c='w', marker='o', s=40)
plt.xlabel('RA')
plt.ylabel('dec')
plt.colorbar(sc)
plt.title('GalSim #675')
xlim = ax.get_xlim()
ylim = ax.get_ylim()

ax2 = fig.add_subplot(122)
sc2 = ax2.scatter(ra_vals_ch, dec_vals_ch, c=sca_vals_ch, s=1,
                  lw=0, cmap=plt.cm.viridis)
ax2.scatter([ra_cen], [dec_cen], c='w', marker='o', s=40)
plt.xlabel('RA')
plt.ylabel('dec')
plt.colorbar(sc2)
plt.title('Python vers of CH code')
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)

if output=='show':
    plt.show()
else:
    print 'Writing to file ',output
    plt.savefig(output)
