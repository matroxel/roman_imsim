import astropy
import pickle
import io
from matplotlib import pyplot as plt 
import fitsio


filename='../debug/fiducial_H158_22531_1_0.cPickle'
filename1='../debug/fiducial_H158_22531_1_1.cPickle'


with io.open(filename, 'rb') as p :
    unpickler = pickle.Unpickler(p)
    while p.peek(1) :
        gal1 = unpickler.load()
        if gal1['ind']==901802:
            gal_stamp1=gal1
        
with io.open(filename1, 'rb') as p2 :
    unpickler = pickle.Unpickler(p2)
    while p2.peek(1) :
        gal2 = unpickler.load()
        if gal2['ind']==901802:
            gal_stamp2=gal2

gal_stamp1.write('old_stamp.fits')
gal_stamp2.write('new_stamp.fits')