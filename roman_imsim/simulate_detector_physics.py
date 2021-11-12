import numpy as np
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
import fitsio as fio
import pickle as pickle
import pickletools
from astropy.time import Time
import cProfile, pstats, psutil
import glob
import shutil
import h5py
import roman_imsim


if __name__ == "__main__":
    param_file = sys.argv[1]
    filter_ = sys.argv[2]
    dither = sys.argv[3]

    # read yaml file
    sim = roman_imsim.roman_sim(param_file)

    # dither number and sca
    if (sim.params['dither_from_file'] is not None) & (sim.params['dither_from_file'] != 'None'):
        if sim.params['dither_and_sca']:
            dither,sca=np.loadtxt(sim.params['dither_from_file'])[int(dither)-1].astype(int) # Assumes array starts with 1
            print('dither: ', dither, ' sca: ', sca)
        else:
            dither=np.loadtxt(sim.params['dither_from_file'])[int(dither)-1] # Assumes array starts with 1
            sca = int(sys.argv[4])
    else:
        sca = int(sys.argv[4])

    # setup dither
    sim.setup(filter_,int(dither),sca=sca,load_cats=False)
    sim.modify_image = roman_imsim.modify_image(sim.params)
    sim.iterate_detector_image()
