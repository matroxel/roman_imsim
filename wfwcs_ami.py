# imports, etc.
import galsim
import galsim.wfirst as wf
import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
from radec_to_chip import *

ra_cen = 25.817233259290145 # degrees
dec_cen = -85.0 # degrees
ra_cen_rad = 0.45059572412999993 # radians
dec_cen_rad = -1.4835298641951802 # radians
pa_rad = 0.0696662245219 #radians
date = datetime.datetime(2025, 1, 12)
seed = 314159

#create coords.txt
ralims = [20,31]
declims = [-85.3,-84.7]
ud = galsim.UniformDeviate(seed)
ra_vals = []
dec_vals = []
for i in range(100000):
    ra_vals.append(ra_cen + (ud() - 0.5)/np.cos((dec_cen*galsim.degrees)/galsim.radians))
    dec_vals.append(dec_cen + ud() - 0.5)
ra_vals = np.array(ra_vals)
dec_vals = np.array(dec_vals)
mask = (ra_vals>ralims[0])&(ra_vals<ralims[1])&(dec_vals>declims[0])&(dec_vals<declims[1])
ra_vals=ra_vals[mask]*np.pi/180.
dec_vals=dec_vals[mask]*np.pi/180.
np.savetxt('coords.txt',np.vstack((ra_vals,dec_vals)).T)


fpa_center = galsim.CelestialCoord(ra=ra_cen*galsim.degrees, dec=dec_cen*galsim.degrees)

wcs = wf.getWCS(fpa_center, PA=pa_rad*galsim.radians, date=date, PA_is_FPA=True)

# Find the SCAs from Chris's code (Python version) for the same points
sca_ch = radec_to_chip(ra_cen_rad, dec_cen_rad, pa_rad,
                       ra_vals, dec_vals)
sca_ch[sca_ch==0]=None
np.savetxt('python.txt',sca_ch)

# Find the SCAs
sca_vals = []
for i in range(n_rand):
    if i % 1000 == 0: print '   ',i
    sca.append(wf.findSCA(wcs, galsim.CelestialCoord(ra=ra_vals[i]*galsim.radians,
                                                    dec=dec_vals[i]*galsim.radians)))
sca=np.array(sca)
np.savetxt('galsim.txt',sca)

sca_c = np.load('c.txt')

# make a plot showing the points colored by their WCS, also with original pointing position shown

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(131)
sc=ax.scatter(ra_vals, dec_vals, c=sca, s=1, 
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

ax2 = fig.add_subplot(132)
sc2 = ax2.scatter(ra_vals, dec_vals, c=sca_ch, s=1,
                  lw=0, cmap=plt.cm.viridis)
ax2.scatter([ra_cen], [dec_cen], c='w', marker='o', s=40)
plt.xlabel('RA')
plt.ylabel('dec')
plt.colorbar(sc2)
plt.title('Python vers of CH code')
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)

ax2 = fig.add_subplot(133)
sc2 = ax2.scatter(ra_vals, dec_vals, c=sca_c, s=1,
                  lw=0, cmap=plt.cm.viridis)
ax2.scatter([ra_cen], [dec_cen], c='w', marker='o', s=40)
plt.xlabel('RA')
plt.ylabel('dec')
plt.colorbar(sc2)
plt.title('Python vers of CH code')
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)

plt.savefig('panel.png')
