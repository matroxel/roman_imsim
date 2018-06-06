""" Run galsim fits from the ngmix pixels branch
This script does a fit over multiple observations of
a given galaxy
Written specifically to split the Healpix file over N<=20 nodes
and then concatenate into one file to write out output
"""
from multiprocessing import Pool, Manager
import numpy as np
import ngmix
from ngmix.observation import Observation, ObsList, MultiBandObsList
from numpy.random import uniform as urand
from ngmix.jacobian import Jacobian
import galsim
from ngmix.galsimfit import GalsimRunner,GalsimSimple,GalsimTemplateFluxFitter
from ngmix.guessers import R50FluxGuesser
import sys
import time
import os
import meds

# Read in the name of the MEDS file with full path
if (len(sys.argv)<3):
    print "Specify a MEDS file name and image type."
    print "e.g. /n/home00/choi.1442/data/wfirst_imsim/test_los_H158_572680.fits"
    df
else:
    medsfile=sys.argv[1]
    key=sys.argv[2]

# ngmix processing function given a meds structure
def f(args):
    print os.getpid(),"working"
    a_list = args
    out = []

    meds_file=medsfile
    meds_data=meds.MEDS(meds_file)

    for n in a_list:
        # Identify how many cutouts there are for this index n
        index=n
        num=meds_data['number'][index]
        ncutout=meds_data['ncutout'][index]
        obs_list=ObsList()
        print "num, index: ", num, index
        # For each of these objects create an observation
        for cdx in range(ncutout):
            cutout_index=cdx
            image=meds_data.get_cutout(index, cutout_index)
            weight=meds_data.get_cweight_cutout(index, cutout_index)
            meds_jacob=meds_data.get_jacobian(index, cutout_index)
            gal_jacob=Jacobian(
                row=meds_jacob['row0'],col=meds_jacob['col0'],
                dvdrow=meds_jacob['dvdrow'],
                dvdcol=meds_jacob['dvdcol'], dudrow=meds_jacob['dudrow'],
                dudcol=meds_jacob['dudcol'])
            psf_image=meds_data.get_psf(index, cutout_index)
            psf_jacob=Jacobian(
                row=31.5,col=31.5,dvdrow=meds_jacob['dvdrow'],
                dvdcol=meds_jacob['dvdcol'], dudrow=meds_jacob['dudrow'],
                dudcol=meds_jacob['dudcol'])
            # Create an obs for each cutout
            psf_obs = Observation(psf_image, jacobian=psf_jacob)
            obs = Observation(
                image, weight=weight, jacobian=gal_jacob, psf=psf_obs)
            # Append the obs to the ObsList
            obs_list.append(obs)

        # With the observations in hand, run ngmix
        guesser=R50FluxGuesser(1.0,100.0)  # Need to work on these guesses?
        ntry=5  # also to be fiddled with
        runner=GalsimRunner(obs_list,'exp',guesser=guesser)
        runner.go(ntry=ntry)
        fitter=runner.get_fitter()
        res=fitter.get_result()
        print res['pars']
        df
        # If there was a problem, save a bunch of -99
        if res['flags'] != 0:
            print "an error occurred with flags:',res['flags']"
            flag_str = " -99"*6
            full_str = '%s %s %s'%(str(index), str(num),flag_str)
        else:
            #print res['flags']
            str_respars = ' '.join(map(str, res['pars']))
            full_str = '%s %s %s'%(str(index), str(num), str_respars)
        out.append(full_str)

    return out

# This is the "main" part of the script, probs want to specify that better

# Write in something to identify if a file isn't specified, or if it isn't 
# found

# Various inits
eps=0.01
np.random.seed(8381)
outputlines=[]
nproc=2  # for debugging purposes
#nproc=16  # hard-coded to suit cosmos52
#nproc=32 # hard-coded to suit cosmos43

# Save the output to a file
hp_idx = medsfile.split("H158_")[-1].split(".fits")[0]
fileo = open('%s_ngmix_out_%s.txt'%(str(key),hp_idx),'w')

# Identify how many objects are in this file and split over the 20 nodes
mfile = medsfile
m=meds.MEDS(mfile)
totid = m['id'][-1]
print "Total objects: ", totid
#totid = 100
#df

id_list = np.arange(totid)
chunks = [(id_list[i::nproc]) for i in range(nproc)]

pool = Pool(processes=nproc)
result = pool.map_async(f, chunks)

while not result.ready():
    #print("Running...")
    time.sleep(0.5)

full_results = result.get()

pool.close()
pool.join()

# Write full results to the file
# first removing all brackets...
#str_test = ' '.join(map(str, full_results))
#print len(str_test)
#print str_test[0]

#df
for idx in range(len(full_results)):
    for jdx in range(len(full_results[idx])): 
        fileo.write("%s\n" %full_results[idx][jdx])

# Some helpful stack overflow refs:
# https://stackoverflow.com/questions/3217002/how-do-you-pass-a-queue-reference-to-a-function-managed-by-pool-map-async
