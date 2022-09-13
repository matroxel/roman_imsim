"""
An implementation of galaxy and star image simulations for Roman.
Built from the Roman GalSim module.
"""


import numpy as np
import healpy as hp
import sys, os, io
import math
import copy
import logging
import time
import yaml
import copy
import galsim as galsim
import galsim.roman as roman
import galsim.config.process as process
import galsim.des as des
# import ngmix
import fitsio as fio
import pickle as pickle
import pickletools
from astropy.time import Time
#import mpi4py
#mpi4py.rc.recv_mprobe = False
#from mpi4py import MPI
# from mpi_pool import MPIPool
import cProfile, pstats, psutil
import glob
import shutil
import h5py

import roman_imsim


# Uncomment for profiling
# pr = cProfile.Profile()

if __name__ == "__main__":
    """
    """

    t0 = time.time()

    # process = psutil.Process(os.getpid())
    # print(process.memory_info().rss/2**30)
    # print(process.memory_info().vms/2**30)

    try:
        param_file = sys.argv[1]
        filter_ = sys.argv[2]
        dither = sys.argv[3]
    except:
        syntax_proc()

    if (param_file.lower() == 'help') or (filter_.lower()=='help') or (dither.lower()=='help'):
        syntax_proc()

    # Uncomment for profiling
    # pr.enable()

    # This instantiates the simulation based on settings in input param file
    sim = roman_imsim.roman_sim(param_file)
    # print(sim.params.keys())

    if sim.params['condor']==True:
        condor=True
    else:
        condor=False

    # This sets up some things like input truth catalogs and empty objects
    if dither=='setup':
        sim.setup(filter_,dither,setup=True)
    elif dither=='meds':
        condor_build = False
        if len(sys.argv)<5:
            syntax_proc()
        if sys.argv[4]=='setup':
            setup = True
            pix = -1
        elif sys.argv[4]=='condor_build':
            condor_build = True
            setup = False
            pix = -1
        elif sys.argv[4]=='cleanup':
            setup = True
            pix = -1
            m = accumulate_output_disk( param_file, filter_, pix, sim.comm, ignore_missing_files = False, setup = setup )
            m.cleanup()
        elif sys.argv[4]=='shape':
            print(sys.argv)
            if (sim.params['meds_from_file'] is not None) & (sim.params['meds_from_file'] != 'None'):
                pix = int(np.loadtxt(sim.params['meds_from_file'])[int(sys.argv[5])-1])
            else:
                pix = int(sys.argv[5])
            m = roman_imsim.accumulate_output_disk( param_file, filter_, pix, sim.comm, shape=True, shape_iter = int(sys.argv[6]), shape_cnt = int(sys.argv[7]))
            #m.get_coadd_shape_mcal() 
            #m.get_coadd_shape_coadd()
            m.get_coadd_shape_multiband_coadd()
            print('out of coadd_shape')
            del(m)
            sys.exit()
        else:
            setup = False
            if (sim.params['meds_from_file'] is not None) & (sim.params['meds_from_file'] != 'None'):
                pix = int(np.loadtxt(sim.params['meds_from_file'])[int(sys.argv[4])-1])
            else:
                pix = int(sys.argv[4])
        m = roman_imsim.accumulate_output_disk( param_file, filter_, pix, sim.comm, ignore_missing_files = False, setup = setup, condor_build = condor_build)
        if not setup:
        	m.finish(condor=sim.params['condor'])
        	sys.exit()

        if setup or condor_build:
            print('exiting')
            sys.exit()
        if m.comm is not None:
            m.comm.Barrier()
        skip = False
        if sim.rank==0:
            for i in range(1,sim.size):
                m.comm.send(m.skip, dest=i)
            skip = m.skip
        else:
            skip = m.comm.recv(source=0)
        if not skip:
            if m.comm is not None:
                m.comm.Barrier()
            if not condor:
                #m.get_coadd_shape_mcal()
                #m.get_coadd_shape_coadd()
                m.get_coadd_shape_multiband_coadd()
            print('out of coadd_shape')
            # print 'commented out finish()'
            m.finish(condor=sim.params['condor'])
            # pr.disable()
            # ps = pstats.Stats(pr).sort_stats('time')
            # ps.print_stats(200)
            del(m)

    else:


        if 'coadd' in sys.argv:
            dither,filter_=np.loadtxt(sim.params['coadd_from_file'])[int(dither)-1].astype(int)
            sim = roman_imsim.postprocessing(param_file)
            if 'verify_output' in sys.argv:
                if sim.check_coaddfile(int(dither),filter_):
                    print('coadd exists',dither,filter_)
                    sys.exit()
            sim.get_coadd(dither,filter_)
            sys.exit()

        if 'psf' in sys.argv:
            dither,filter_=np.loadtxt(sim.params['coadd_from_file'])[int(dither)-1].astype(int)
            sim = roman_imsim.postprocessing(param_file)
            if 'verify_output' in sys.argv:
                if sim.check_coaddfile(int(dither),filter_):
                    print('coadd exists',dither,filter_)
                    sys.exit()
            sim.get_coadd_psf(dither,filter_)
            sys.exit()

        if 'detection' in sys.argv:
            dither=np.unique(np.loadtxt(sim.params['coadd_from_file'])[:,0])[int(dither)-1].astype(int)
            sim = roman_imsim.postprocessing(param_file)
            sim.get_detection(dither)
            sys.exit()

        if (sim.params['dither_from_file'] is not None) & (sim.params['dither_from_file'] != 'None'):
            if sim.params['dither_and_sca']:
                dither,sca=np.loadtxt(sim.params['dither_from_file'])[int(dither)-1].astype(int) # Assumes array starts with 1
            else:
                dither=np.loadtxt(sim.params['dither_from_file'])[int(dither)-1] # Assumes array starts with 1
                sca = int(sys.argv[4])
        else:
            sca = int(sys.argv[4])
        print(sys.argv)

        if 'detector_physics' in sys.argv:
            sim.setup(filter_,int(dither),sca=sca,load_cats=False)
            sim.modify_image = roman_imsim.modify_image(sim.params,sim.pointing)
            sim.iterate_detector_image()
            # sim.iterate_detector_stamps('gal')
            # sim.iterate_detector_stamps('star')
            sys.exit()

        if 'verify_output' in sys.argv:
            if sim.check_file(sca,int(dither),filter_):
                print('exists',dither,sca)
                sys.exit()
        skip = sim.setup(filter_,int(dither),sca=sca)

        #tmp_name_id = int(sys.argv[6])

        # Loop over SCAs
        if sim.comm is not None:
            sim.comm.Barrier()
        # This sets up the object that will simulate various roman detector effects, noise, etc. Instantiation creates a noise realisation for the image.
        sim.modify_image = roman_imsim.modify_image(sim.params,sim.pointing)
        print('modified image', sca)
        # This is the main thing - iterates over galaxies for a given pointing and SCA and simulates them all
        if sim.comm is not None:
            sim.comm.Barrier()
        print(time.time()-t0)
        # print(process.memory_info().rss/2**30)
        # print(process.memory_info().vms/2**30)
        sim.iterate_image()
        if sim.comm is not None:
            sim.comm.Barrier()

        # Uncomment for profiling
        # pr.disable()
        # pr.dump_stats(param_file+'.pstats')
        # ps = pstats.Stats(pr).sort_stats('time')
        # ps.print_stats(50)

    # if sim.params['condor']==True:
    #     condor_cleanup(sim.params['out_path'])

# test round galaxy recovered to cover wcs errors

# same noise in different runs? same noise
# distance to edge to reject images?

