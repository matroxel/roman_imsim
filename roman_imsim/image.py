# from __future__ import division
# from __future__ import print_function

# from future import standard_library
# standard_library.install_aliases()
# from builtins import str
# from builtins import range
# from past.builtins import basestring
# from builtins import object
# from past.utils import old_div

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
from mpi4py import MPI
# from mpi_pool import MPIPool
import cProfile, pstats, psutil
import glob
import shutil
import h5py

from .universe import setupCCM_ab
from .universe import addDust
from .misc import ParamError
from .misc import except_func
from .misc import save_obj
from .misc import load_obj
from .misc import convert_dither_to_fits
from .misc import convert_gaia
from .misc import convert_galaxia
from .misc import create_radec_fits
from .misc import hsm
from .misc import get_filename
from .misc import get_filenames
from .misc import write_fits

path, filename = os.path.split(__file__)
sedpath_Star   = os.path.join(galsim.meta_data.share_dir, 'SEDs', 'vega.txt')


class draw_image(object):
    """
    This is where the management of drawing happens (basicaly all the galsim interaction).
    The general process is that 1) a galaxy model is specified from the truth catalog, 2) rotated, sheared, and convolved with the psf, 3) its drawn into a postage samp, 4) that postage stamp is added to a persistent image of the SCA, 5) the postage stamp is finalized by going through make_image(). Objects within the SCA are iterated using the iterate_*() functions, and the final SCA image (self.im) can be completed with self.finalize_sca().
    """

    def __init__(self, params, pointing, modify_image, cats, logger, image_buffer=1024, rank=0, comm=None):
        """
        Sets up some general properties, including defining the object index lists, starting the generator iterators, assigning the SEDs (single stand-ins for now but generally red to blue for bulg/disk/knots), defining SCA bounds, and creating the empty SCA image.

        Input
        params          : parameter dict
        pointing        : Pointing object
        modify_image    : modify_image object
        cats            : init_catalots object
        logger          : logger instance
        gal_ind_list    : List of indices from gal truth catalog to attempt to simulate
        star_ind_list   : List of indices from star truth catalog to attempt to simulate
        image_buffer    : Number of pixels beyond SCA to attempt simulating objects that may overlap SCA
        rank            : process rank
        """

        self.params       = params
        self.pointing     = pointing
        self.modify_image = modify_image
        self.cats         = cats
        self.gal_iter     = 0
        self.star_iter    = 0
        self.supernova_iter = 0
        self.gal_done     = False
        self.star_done    = False
        self.supernova_done = False
        self.lightcurves = self.cats.lightcurves
        self.rank         = rank
        self.rng          = galsim.BaseDeviate(self.params['random_seed'])
        self.star_stamp   = None
        self.t0           = time.time()
        self.gal_stamp_too_large = False
        self.gal_model = None
        self.gal_stamp = None
        self.weight = None

        # Option to change exposure time (in seconds)
        if 'exposure_time' in self.params:
            if self.params['exposure_time'] == 'deep':
                if self.pointing.filter[0] == 'Y':
                    roman.exptime = 230
                if self.pointing.filter[0] == 'J':
                    roman.exptime = 230
                if self.pointing.filter[0] == 'H':
                    roman.exptime = 340
                if self.pointing.filter[0] == 'F':
                    roman.exptime = 1000
        
        # Setup galaxy SED
        # Need to generalize to vary sed based on input catalog
        if not self.params['dc2']:
            self.galaxy_sed_b = galsim.SED(self.params['sedpath_E'], wave_type='Ang', flux_type='flambda')
            self.galaxy_sed_d = galsim.SED(self.params['sedpath_Scd'], wave_type='Ang', flux_type='flambda')
            self.galaxy_sed_n = galsim.SED(self.params['sedpath_Im'],  wave_type='Ang', flux_type='flambda')
            # Setup star SED
            self.star_sed     = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')
            self.supernova_sed = galsim.SED(sedpath_Star, wave_type='nm', flux_type='flambda')

        # Galsim bounds object to specify area to simulate objects that might overlap the SCA
        self.b0  = galsim.BoundsI(  xmin=1-int(image_buffer/2),
                                    ymin=1-int(image_buffer/2),
                                    xmax=roman.n_pix+int(image_buffer/2),
                                    ymax=roman.n_pix+int(image_buffer/2))
        # Galsim bounds object to specify area to simulate objects that would have centroids that fall on the SCA to save as postage stamps (pixels not on the SCA have weight=0)
        self.b   = galsim.BoundsI(  xmin=1,
                                    ymin=1,
                                    xmax=roman.n_pix,
                                    ymax=roman.n_pix)

        # SCA image (empty right now)
        if self.params['draw_sca']:
            self.im = galsim.Image(self.b, wcs=self.pointing.WCS)
        else:
            self.im = None

        # Get sky background for pointing
        self.sky_level = roman.getSkyLevel(self.pointing.bpass,
                                            world_pos=self.pointing.WCS.toWorld(
                                                        galsim.PositionI(roman.n_pix/2,
                                                                        roman.n_pix/2)),
                                            date=self.pointing.date)
        self.sky_level *= (1.0 + roman.stray_light_fraction)*roman.pixel_scale**2 # adds stray light and converts to photons/cm^2
        self.sky_level *= 32*32 # Converts to photons, but uses smallest stamp size to do so - not optimal

        if self.params['dc2']:
            self.ax={}
            self.bx={}
            wavelen = np.arange(3000.,11500.+1.,1., dtype='float')
            sb = np.zeros(len(wavelen), dtype='float')
            sb[abs(wavelen-5000.)<1./2.] = 1.
            self.imsim_bpass = galsim.Bandpass(galsim.LookupTable(x=wavelen,f=sb,interpolant='nearest'),'a',blue_limit=3000., red_limit=11500.).withZeropoint('AB')

    def iterate_gal(self):
        """
        Iterator function to loop over all possible galaxies to draw
        """

        if self.gal_iter==0:
            self.t0 = time.time()


        # Check if the end of the galaxy list has been reached; return exit flag (gal_done) True
        # You'll have a bad day if you aren't checking for this flag in any external loop...
        # self.gal_done = True
        # return
        if self.gal_iter == self.cats.get_gal_length():
            self.gal_done = True
            print('Proc '+str(self.rank)+' done with galaxies.',time.time()-self.t0)
            return

        # Reset galaxy information
        del(self.gal_model)
        del(self.gal_stamp)
        del(self.weight)
        self.gal_model = None
        self.gal_stamp = None
        self.weight    = None

        # if self.gal_iter>1000:
        #     self.gal_done = True
        #     return


        if self.gal_iter%100==0:
            print('Progress '+str(self.rank)+': Attempting to simulate galaxy '+str(self.gal_iter)+' in SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.')

        # Galaxy truth index and array for this galaxy
        self.ind,self.gal = self.cats.get_gal(self.gal_iter)
        self.gal_iter    += 1
        self.rng          = galsim.BaseDeviate(self.params['random_seed']+self.ind+self.pointing.dither)

        # if self.ind != 157733:
        #     return

        # if self.ind != 144078:
        #     return

        # If galaxy image position (from wcs) doesn't fall within simulate-able bounds, skip (slower)
        # If it does, draw it
        if self.check_position(self.gal['ra'],self.gal['dec'],gal=True):
            #print('good position')
            # print('iterate',self.gal_iter,time.time()-t0)
            # print(process.memory_info().rss/2**30)
            # print(process.memory_info().vms/2**30)
            self.draw_galaxy()

    def iterate_star(self):
        """
        Iterator function to loop over all possible stars to draw
        """

        if self.star_iter==0:
            self.t0 = time.time()

        # self.star_done = True
        # return
        # Don't draw stars into postage stamps
        if not self.params['draw_sca']:
            self.star_done = True
            print('Proc '+str(self.rank)+' done with stars.')
            return
        if not self.params['draw_stars']:
            self.star_done = True
            print('Proc '+str(self.rank)+' not doing stars.')
            return
        # Check if the end of the star list has been reached; return exit flag (gal_done) True
        # You'll have a bad day if you aren't checking for this flag in any external loop...
        if self.star_iter == self.cats.get_star_length():
            self.star_done = True
            print('Proc '+str(self.rank)+' done with stars.',time.time()-self.t0)
            return

        # Not participating in star parallelisation
        if self.rank == -1:
            self.star_done = True
            return

        # Reset star information
        self.st_model = None
        self.star_stamp = None

        # if self.star_iter%10==0:
        print('Progress '+str(self.rank)+': Attempting to simulate star '+str(self.star_iter)+' in SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.')

        # Star truth index for this galaxy
        self.ind,self.star = self.cats.get_star(self.star_iter)
        self.star_iter    += 1
        self.rng        = galsim.BaseDeviate(self.params['random_seed']+self.ind+self.pointing.dither)

        # If star image position (from wcs) doesn't fall within simulate-able bounds, skip (slower)
        # If it does, draw it
        #print(self.ind, self.star, self.star_iter)
        if self.check_position(self.star['ra'],self.star['dec']):
            self.draw_star()

    def iterate_supernova(self):
        if not self.params['draw_sca']:
            self.supernova_done = True
            print('Proc '+str(self.rank)+' done with supernova.')
            return 
        if not self.params['draw_supernova']:
            self.supernova_done = True
            print('Proc '+str(self.rank)+' not doing stars.')
            return             
        # Check if the end of the supernova list has been reached; return exit flag (supernova_done) True
        # You'll have a bad day if you aren't checking for this flag in any external loop...
        if self.supernova_iter == self.cats.get_supernova_length():
            self.supernova_done = True
            return 

        # Not participating in star parallelisation
        if self.rank == -1:
            self.supernova_done = True
            return 
            
        self.supernova_stamp = None
        # if self.star_iter%10==0:
        print('Progress '+str(self.rank)+': Attempting to simulate supernova '+str(self.supernova_iter)+' in SCA '+str(self.pointing.sca)+' and dither '+str(self.pointing.dither)+'.')
        
        #Host galaxy variable
        self.hostid = None
        
        # Supernova truth index for this supernova
        self.ind,self.supernova = self.cats.get_supernova(self.supernova_iter)
        self.supernova_iter    += 1
        self.rng        = galsim.BaseDeviate(self.params['random_seed']+self.ind+self.pointing.dither)

        # If supernova image position (from wcs) doesn't fall within simulate-able bounds, skip (slower) 
        # If it does, draw it
        if self.check_position(self.supernova['ra'],self.supernova['dec']) and self.lightcurves['field'][self.supernova['ptrobs_min'] - 1] == 'DEEP':
            self.draw_supernova()

    def check_position(self, ra, dec, gal=False):
        """
        Create the world and image position galsim objects for obj, as well as the local WCS. Return whether object is in SCA (+half-stamp-width border).

        Input
        ra  : RA of object
        dec : Dec of object
        """

        # Galsim world coordinate object (ra,dec)
        self.radec = galsim.CelestialCoord(ra*galsim.radians, dec*galsim.radians)

        # Galsim image coordinate object
        self.xy = self.pointing.WCS.toImage(self.radec)

        # Galsim integer image coordinate object
        self.xyI = galsim.PositionI(int(self.xy.x),int(self.xy.y))

        # Galsim image coordinate object holding offset from integer pixel grid
        # troxel needs to change this
        self.offset = self.xy-self.xyI

        # Define the local_wcs at this world position
        self.local_wcs = self.pointing.WCS.local(self.xy)

        # Discard objects too far from SCA
        if self.xy.x<1:
            dboundsx = -(self.xy.x-1)
        else:
            dboundsx = self.xy.x-roman.n_pix
        if self.xy.y<1:
            dboundsy = -(self.xy.y-1)
        else:
            dboundsy = self.xy.y-roman.n_pix

        if gal:
            if dboundsx>10*(np.max(self.gal['size'])/.11):
                return False

            if dboundsy>10*(np.max(self.gal['size'])/.11):
                return False

        # Return whether object is in SCA (+half-stamp-width border)
        #print('is the object in SCA', self.b0.includes(self.xyI))
        return self.b0.includes(self.xyI)

    def make_sed_model(self, model, sed):
        """
        Modifies input SED to be at appropriate redshift and magnitude, then applies it to the object model.

        Input
        model : Galsim object model
        sed   : Template SED for object
        """

        # Apply correct flux from magnitude for filter bandpass
        sed_ = sed.atRedshift(self.gal['z'])
        sed_ = sed_.withMagnitude(self.gal[self.pointing.filter], self.pointing.bpass)

        # Return model with SED applied
        return model * sed_

    def make_sed_model_dc2(self, model, obj, i):
        """
        Modifies input SED to be at appropriate redshift and magnitude, deals with dust model, then applies it to the object model.

        Input
        model : Galsim object model
        i     : component index to extract truth params
        """

        if i==-1:
            sedname = obj['sed'].lstrip().rstrip()
            magnorm = obj['mag_norm']
            Av = obj['A_v']
            Rv = obj['R_v']
        else:
            if i==2:
                sedname = obj['sed'][1].lstrip().rstrip()
            else:
                sedname = obj['sed'][i].lstrip().rstrip()
            magnorm = obj['mag_norm'][i]
            Av = obj['A_v'][i]
            Rv = obj['R_v'][i]

        sed = self.cats.seds[sedname]
        sed_lut = galsim.LookupTable(x=sed[:,0],f=sed[:,1])
        sed = galsim.SED(sed_lut, wave_type='nm', flux_type='flambda',redshift=0.)
        sed_ = sed.withMagnitude(magnorm, self.imsim_bpass) # apply mag
        if len(sed_.wave_list) not in self.ax:
            ax,bx = setupCCM_ab(sed_.wave_list)
            self.ax[len(sed_.wave_list)] = ax
            self.bx[len(sed_.wave_list)] = bx
        dust = addDust(self.ax[len(sed_.wave_list)], self.bx[len(sed_.wave_list)], A_v=Av, R_v=Rv)
        sed_ = sed_._mul_scalar(dust) # Add dust extinction. Same function from lsst code for testing right now
        sed_ = sed_.atRedshift(obj['z']) # redshift

        # Return model with SED applied
        return model * sed_

    def galaxy_model(self):
        """
        Generate the intrinsic galaxy model based on truth catalog parameters
        """


        if self.params['dc2']:

            # loop over components, order bulge, disk, knots
            for i in range(3):
                if self.gal['size'][i] == 0:
                    continue
                # If any flux, build component and apply appropriate SED
                if i<2:
                    if i==0:
                        n=4
                    else:
                        n=1
                    component = galsim.Sersic(n, half_light_radius=1.*self.gal['size'][i], flux=1., trunc=10.*self.gal['size'][i])
                else:
                    rng   = galsim.BaseDeviate((int(self.gal['gind'])<<10)+127) #using orig phosim unique id as random seed, which requires bit appending 127 to represent knots model
                    component = galsim.RandomKnots(npoints=self.gal['knots'], half_light_radius=1.*self.gal['size'][i], flux=1., rng=rng)
                # Apply intrinsic ellipticity to the component.
                s         = galsim.Shear(q=1./self.gal['q'][i], beta=(90.+self.gal['pa'][i])*galsim.degrees)
                s         = galsim._Shear(complex(s.g1,-s.g2)) # Fix -g2
                component = component.shear(s)
                # Apply the SED
                component = self.make_sed_model_dc2(component, self.gal, i)

                if self.gal_model is None:
                    # No model, so save the galaxy model as the component
                    self.gal_model = component
                else:
                    # Existing model, so save the galaxy model as the sum of the components
                    self.gal_model = galsim.Add([self.gal_model, component])

        else:

            # Generate galaxy model
            # Calculate flux fraction of disk portion 
            flux = (1.-self.gal['bflux']) * self.gal['dflux']
            if flux > 0:
                # If any flux, build Sersic disk galaxy (exponential) and apply appropriate SED
                self.gal_model = galsim.Sersic(1, half_light_radius=1.*self.gal['size'], flux=flux, trunc=10.*self.gal['size'])
                self.gal_model = self.make_sed_model(self.gal_model, self.galaxy_sed_d)
                # self.gal_model = self.gal_model.withScaledFlux(flux)

            # Calculate flux fraction of knots portion 
            flux = (1.-self.gal['bflux']) * (1.-self.gal['dflux'])
            if flux > 0:
                # If any flux, build star forming knots model and apply appropriate SED
                rng   = galsim.BaseDeviate(self.params['random_seed']+self.ind)
                knots = galsim.RandomKnots(npoints=self.params['knots'], half_light_radius=1.*self.gal['size'], flux=flux, rng=rng) 
                knots = self.make_sed_model(galsim.ChromaticObject(knots), self.galaxy_sed_n)
                # knots = knots.withScaledFlux(flux)
                # Sum the disk and knots, then apply intrinsic ellipticity to the disk+knot component. Fixed intrinsic shape, but can be made variable later.
                self.gal_model = galsim.Add([self.gal_model, knots])
                self.gal_model = self.gal_model.shear(e1=self.gal['int_e1'], e2=self.gal['int_e2'])

            # Calculate flux fraction of bulge portion 
            flux = self.gal['bflux']
            if flux > 0:
                # If any flux, build Sersic bulge galaxy (de vacaleurs) and apply appropriate SED
                bulge = galsim.Sersic(4, half_light_radius=1.*self.gal['size'], flux=flux, trunc=10.*self.gal['size']) 
                # Apply intrinsic ellipticity to the bulge component. Fixed intrinsic shape, but can be made variable later.
                bulge = bulge.shear(e1=self.gal['int_e1'], e2=self.gal['int_e2'])
                # Apply the SED
                bulge = self.make_sed_model(bulge, self.galaxy_sed_b)
                # bulge = bulge.withScaledFlux(flux)

                if self.gal_model is None:
                    # No disk or knot component, so save the galaxy model as the bulge part
                    self.gal_model = bulge
                else:
                    # Disk/knot component, so save the galaxy model as the sum of two parts
                    self.gal_model = galsim.Add([self.gal_model, bulge])

    def galaxy(self):
        """
        Call galaxy_model() to get the intrinsic galaxy model, then apply properties relevant to its observation
        """

        # Build intrinsic galaxy model
        self.galaxy_model()

        # print('model1',time.time()-t0)
        # print(process.memory_info().rss/2**30)
        # print(process.memory_info().vms/2**30)

        if self.params['dc2']:
            g1 = self.gal['g1']/(1. - self.gal['k'])
            g2 = -self.gal['g2']/(1. - self.gal['k'])
            mu = 1./((1. - self.gal['k'])**2 - (self.gal['g1']**2 + self.gal['g2']**2))
            # Apply a shear
            self.gal_model = self.gal_model.lens(g1=g1,g2=g2,mu=mu)
            # Rescale flux appropriately for roman
            self.mag = self.gal_model.calculateMagnitude(self.pointing.bpass)
            self.gal_model = self.gal_model * galsim.roman.collecting_area * galsim.roman.exptime
        else:
            # Random rotation (pairs of objects are offset by pi/2 to cancel shape noise)
            self.gal_model = self.gal_model.rotate(self.gal['rot']*galsim.radians) 
            # Apply a shear
            self.gal_model = self.gal_model.shear(g1=self.gal['g1'],g2=self.gal['g2'])
            # Rescale flux appropriately for roman
            self.mag = self.gal_model.calculateMagnitude(self.pointing.bpass)
            self.gal_model = self.gal_model * galsim.roman.collecting_area * galsim.roman.exptime

        # Ignoring chromatic stuff for now for speed, so save correct flux of object
        flux = self.gal_model.calculateFlux(self.pointing.bpass)
        # print(flux,self.mag)
        # print 'galaxy flux',flux
        # Evaluate the model at the effective wavelength of this filter bandpass (should change to effective SED*bandpass?)
        # This makes the object achromatic, which speeds up drawing and convolution
        # self.gal_model  = self.gal_model.evaluateAtWavelength(self.pointing.bpass.effective_wavelength)
        # Reassign correct flux
        # self.gal_model  = self.gal_model.withFlux(flux) # reapply correct flux

        if self.sky_level/flux < galsim.GSParams().folding_threshold:
            gsparams = galsim.GSParams( folding_threshold=self.sky_level/flux,
                                        maximum_fft_size=16384 )
        else:
            gsparams = galsim.GSParams( maximum_fft_size=16384 )
        gsparams = galsim.GSParams( maximum_fft_size=16384 )

        # Convolve with PSF
        # self.gal_model = galsim.Convolve(self.gal_model.withGSParams(gsparams), self.pointing.load_psf(self.xyI), propagate_gsparams=False)

        # Convolve with additional los motion (jitter), if any
        if self.pointing.los_motion is not None:
            self.gal_model = galsim.Convolve(self.gal_model, self.pointing.los_motion)

        # chromatic stuff replaced by above lines
        # # Draw galaxy igal into stamp.
        # self.gal_list[igal].drawImage(self.pointing.bpass[self.params['filter']], image=gal_stamp)
        # # Add detector effects to stamp.

        return flux

    def get_stamp_size(self,obj,flux):
        """
        Select the stamp size multiple to use.

        Input
        obj    : Galsim object
        factor : Factor to multiple suggested galsim stamp size by
        """

        #return int(obj.getGoodImageSize(roman.pixel_scale)/self.stamp_size)
        #return int(obj.getGoodImageSize(roman.pixel_scale)/(2**factor))
        # return 2*np.ceil(1.*np.ceil(self.gal['size']/(np.sqrt(2*np.log(2)))*1.25)/self.stamp_size)
        if self.params['dc2']:
            # gal array size is 3, (bulge, disk, knots)
            galsize = 2*10*max(self.gal['size'])

        else:
            galsize = 2*10*self.gal['size']

        stamp_size = int(2**(np.ceil(np.log2(galsize/roman.pixel_scale))+1))
        # This makes the object achromatic, which speeds up drawing and convolution
        tmp_obj  = obj.evaluateAtWavelength(self.pointing.bpass.effective_wavelength)
        # Reassign correct flux
        tmp_obj  = tmp_obj.withFlux(flux) # reapply correct flux
        stamp_image_size = tmp_obj.getGoodImageSize(roman.pixel_scale)
        if stamp_image_size<stamp_size:
            stamp_image_size = stamp_size
        return stamp_size,stamp_image_size

    def draw_galaxy(self):
        """
        Draw the galaxy model into the SCA (neighbors and blending) and/or the postage stamp (isolated).
        """

        self.gal_stamp_too_large = False

        # Build galaxy model that will be drawn into images
        flux = self.galaxy()

        # print('draw_galaxy1',time.time()-t0)
        # print(process.memory_info().rss/2**30)
        # print(process.memory_info().vms/2**30)

        stamp_size,stamp_image_size = self.get_stamp_size(self.gal_model,flux)
        self.stamp_size = stamp_size

        # # Skip drawing some really huge objects (>twice the largest stamp size)
        # if stamp_size>2.*self.num_sizes:
        #     return

        # Create postage stamp bounds at position of object
        b = galsim.BoundsI( xmin=self.xyI.x-int(stamp_image_size/2)+1,
                            ymin=self.xyI.y-int(stamp_image_size/2)+1,
                            xmax=self.xyI.x+int(stamp_image_size/2),
                            ymax=self.xyI.y+int(stamp_image_size/2))
        b2 = galsim.BoundsI( xmin=self.xyI.x-int(stamp_size/2)+1,
                            ymin=self.xyI.y-int(stamp_size/2)+1,
                            xmax=self.xyI.x+int(stamp_size/2),
                            ymax=self.xyI.y+int(stamp_size/2))

        # If this postage stamp doesn't overlap the SCA bounds at all, no reason to draw anything
        if not (b&self.b).isDefined():
            return

        # Create postage stamp for galaxy
        gal_stamp = galsim.Image(b, wcs=self.pointing.WCS)

        # print('draw_galaxy2',time.time()-t0)
        # print(process.memory_info().rss/2**30)
        # print(process.memory_info().vms/2**30)

        # Draw galaxy model into postage stamp. This is the basis for both the postage stamp output and what gets added to the SCA image. This will obviously create biases if the postage stamp is too small - need to monitor that.
        #self.gal_model.drawImage(self.pointing.bpass,image=gal_stamp,offset=self.xy-b.true_center,method='phot',rng=self.rng)

        # gal_stamp.write(str(self.ind)+'.fits')

        # print('draw_galaxy3',time.time()-t0)
        # print(process.memory_info().rss/2**30)
        # print(process.memory_info().vms/2**30)

        # Add galaxy stamp to SCA image
        if self.params['draw_sca']:
            self.im[b&self.b] = self.im[b&self.b] + gal_stamp[b&self.b]

        # If object too big for stamp sizes, or not saving stamps, skip saving a stamp
        if stamp_size>256:
            print('too big stamp',stamp_size)
            self.gal_stamp_too_large = True
            return

        if 'no_stamps' in self.params:
            if self.params['no_stamps']:
                print('test stamps line')
                self.gal_stamp_too_large = True
                self.gal_stamp = -1
                return

        # print('draw_galaxy4',time.time()-t0)
        # print(process.memory_info().rss/2**30)
        # print(process.memory_info().vms/2**30)

        # Check if galaxy center falls on SCA
        # Apply background, noise, and Roman detector effects
        # Get final galaxy stamp and weight map
        if self.b.includes(self.xyI):
            # if 
            #     self.gal_stamp_too_large = True
            #     return

            gal_stamp = gal_stamp[b&b2]
            gal_stamp = gal_stamp[b2&self.b]
            self.gal_b = None#b2
            self.gal_stamp            = None#galsim.Image(b2, wcs=self.pointing.WCS)
            # self.gal_stamp[b2&self.b] = gal_stamp
            self.weight            = None#galsim.Image(b2, wcs=self.pointing.WCS,init_value=0.)
            # self.weight[b2&self.b].array[:,:] = 1
            # self.weight            = self.weight.array

            # # Copy part of postage stamp that falls on SCA - set weight map to zero for parts outside SCA
            # self.gal_stamp = galsim.Image(b, wcs=self.pointing.WCS)
            # self.gal_stamp[b&self.b] = self.gal_stamp[b&self.b] + gal_stamp[b&self.b]
            # self.weight_stamp = galsim.Image(b, wcs=self.pointing.WCS)
            # if weight != None:
            #     self.weight_stamp[b&self.b] = self.weight_stamp[b&self.b] + weight[b&self.b]


            # If we're saving the true PSF model, simulate an appropriate unit-flux star and draw it (oversampled) at the position of the galaxy
            if (self.params['draw_true_psf']) and (not self.params['skip_stamps']):
                self.star_model() #Star model for PSF (unit flux)
                # Create modified WCS jacobian for super-sampled pixelisation
                wcs = galsim.JacobianWCS(dudx=self.local_wcs.dudx/self.params['oversample'],
                                         dudy=self.local_wcs.dudy/self.params['oversample'],
                                         dvdx=self.local_wcs.dvdx/self.params['oversample'],
                                         dvdy=self.local_wcs.dvdy/self.params['oversample'])
                # Create postage stamp bounds at position of object
                b_psf = galsim.BoundsI( xmin=self.xyI.x-int(self.params['psf_stampsize'])/2+1,
                                    ymin=self.xyI.y-int(self.params['psf_stampsize'])/2+1,
                                    xmax=self.xyI.x+int(self.params['psf_stampsize'])/2,
                                    ymax=self.xyI.y+int(self.params['psf_stampsize'])/2)
                # Create postage stamp bounds at position of object
                b_psf2 = galsim.BoundsI( xmin=self.xyI.x-int(self.params['psf_stampsize']*self.params['oversample'])/2+1,
                                    ymin=self.xyI.y-int(self.params['psf_stampsize']*self.params['oversample'])/2+1,
                                    xmax=self.xyI.x+int(self.params['psf_stampsize']*self.params['oversample'])/2,
                                    ymax=self.xyI.y+int(self.params['psf_stampsize']*self.params['oversample'])/2)
                # Create psf stamp with oversampled pixelisation
                self.psf_stamp = galsim.Image(b_psf, wcs=self.pointing.WCS)
                # print('draw_galaxy5',time.time()-t0)
                # print(process.memory_info().rss/2**30)
                # print(process.memory_info().vms/2**30)
                self.psf_stamp2 = galsim.Image(b_psf2, wcs=wcs)
                # Draw PSF into postage stamp
                # self.st_model.drawImage(self.pointing.bpass,image=self.psf_stamp,wcs=self.pointing.WCS)
                # self.st_model.drawImage(self.pointing.bpass,image=self.psf_stamp2,wcs=wcs,method='no_pixel')
                self.st_model.drawImage(image=self.psf_stamp,wcs=self.pointing.WCS)
                self.st_model.drawImage(image=self.psf_stamp2,wcs=wcs,method='no_pixel')
            # print('draw_galaxy6',time.time()-t0)
            # print(process.memory_info().rss/2**30)
            # print(process.memory_info().vms/2**30)

    def star_model(self, sed = None, mag = 0.):
        """
        Create star model for PSF or for drawing stars into SCA

        Input
        sed  : The stellar SED
        mag  : The magnitude of the star
        """

        # Generate star model (just a delta function) and apply SED
        if sed is not None:
            if self.params['dc2']:
                self.st_model = galsim.DeltaFunction()
                self.st_model = self.make_sed_model_dc2(self.st_model, self.star, -1)
                mag = self.st_model.calculateMagnitude(self.pointing.bpass)
                self.st_model = self.st_model * roman.collecting_area * roman.exptime
            else:
                sed_ = sed.withMagnitude(mag, self.pointing.bpass)
                self.st_model = galsim.DeltaFunction() * sed_  * roman.collecting_area * roman.exptime
            flux = self.st_model.calculateFlux(self.pointing.bpass)
            ft = int(self.sky_level/flux)
            # print mag,flux,ft
            # if ft<0.0005:
            #     ft = 0.0005
            if ft < galsim.GSParams().folding_threshold:
                gsparams = galsim.GSParams( folding_threshold=int(self.sky_level/flux),
                                            maximum_fft_size=16384 )
            else:
                gsparams = galsim.GSParams( maximum_fft_size=16384 )
        else:
            self.st_model = galsim.DeltaFunction(flux=1.)
            flux = 1.
            gsparams = galsim.GSParams( maximum_fft_size=16384 )

        # Evaluate the model at the effective wavelength of this filter bandpass (should change to effective SED*bandpass?)
        # This makes the object achromatic, which speeds up drawing and convolution
        # self.st_model = self.st_model.evaluateAtWavelength(self.pointing.bpass.effective_wavelength)
        # Reassign correct flux
        # self.st_model  = self.st_model.withFlux(flux) # reapply correct flux

        # Convolve with PSF
        if mag<12:
            psf = self.pointing.load_psf(self.xyI,pupil_bin=2)
            psf = psf.withGSParams(galsim.GSParams(folding_threshold=1e-4))
        elif mag<15:
            psf = self.pointing.load_psf(self.xyI,pupil_bin=4)
            psf = psf.withGSParams(galsim.GSParams(folding_threshold=1e-3))
        else:
            psf = self.pointing.load_psf(self.xyI)
        self.st_model = galsim.Convolve(self.st_model , psf)

        # Convolve with additional los motion (jitter), if any
        if self.pointing.los_motion is not None:
            self.st_model = galsim.Convolve(self.st_model, self.pointing.los_motion)

        # old chromatic version
        # self.psf_list[igal].drawImage(self.pointing.bpass[self.params['filter']],image=psf_stamp, wcs=local_wcs)

        return mag,flux

    def draw_star(self):
        """
        Draw a star into the SCA
        """

        # Get star model with given SED and flux
        if self.params['dc2']:
            self.mag,flux = self.star_model(sed=self.star['sed'].lstrip().rstrip())
        else:
            self.mag,flux = self.star_model(sed=self.star_sed,mag=self.star[self.pointing.filter])

        # Get good stamp size multiple for star
        # stamp_size = self.get_stamp_size(self.st_model)#.withGSParams(gsparams))
        stamp_size = 1600

        # Create postage stamp bounds for star
        # b = galsim.BoundsI( xmin=self.xyI.x-int(stamp_size*self.stamp_size)/2,
        #                     ymin=self.xyI.y-int(stamp_size*self.stamp_size)/2,
        #                     xmax=self.xyI.x+int(stamp_size*self.stamp_size)/2,
        #                     ymax=self.xyI.y+int(stamp_size*self.stamp_size)/2 )
        b = galsim.BoundsI( xmin=self.xyI.x-int(stamp_size/2),
                            ymin=self.xyI.y-int(stamp_size/2),
                            xmax=self.xyI.x+int(stamp_size/2),
                            ymax=self.xyI.y+int(stamp_size/2) )
        # Create postage stamp bounds at position of object
        b_psf = galsim.BoundsI( xmin=self.xyI.x-int(self.params['psf_stampsize'])/2+1,
                            ymin=self.xyI.y-int(self.params['psf_stampsize'])/2+1,
                            xmax=self.xyI.x+int(self.params['psf_stampsize'])/2,
                            ymax=self.xyI.y+int(self.params['psf_stampsize'])/2)        
        self.star_b = b_psf

        # If postage stamp doesn't overlap with SCA, don't draw anything
        if not (b&self.b).isDefined():
            return

        # Create star postage stamp
        star_stamp = galsim.Image(b, wcs=self.pointing.WCS)

        # print(self.star[self.pointing.filter],repr(self.st_model))
        # Draw star model into postage stamp
        if self.mag<15:
            self.st_model.drawImage(self.pointing.bpass,image=star_stamp,offset=self.xy-b.true_center)
        else:
            self.st_model.drawImage(self.pointing.bpass,image=star_stamp,offset=self.xy-b.true_center,method='phot',rng=self.rng,maxN=1000000)

        # star_stamp.write('/fs/scratch/cond0083/roman_sim_out/images/'+str(self.ind)+'.fits.gz')

        # Add star stamp to SCA image
        self.im[b&self.b] += star_stamp[b&self.b]
        # self.st_model.drawImage(image=self.im,add_to_image=True,offset=self.xy-self.im.true_center,method='phot',rng=self.rng,maxN=1000000)

        if self.b.includes(self.xyI):
            star_stamp = star_stamp[b&b_psf]
            star_stamp = star_stamp[b_psf&self.b]
            self.star_stamp = galsim.Image(b_psf, wcs=self.pointing.WCS)
            self.star_stamp[b_psf&self.b] = star_stamp
            self.weight            = galsim.Image(b_psf, wcs=self.pointing.WCS,init_value=0.)
            self.weight[b_psf&self.b].array[:,:] = 1
            self.weight            = self.weight.array
            self.supernova_stamp = self.star_stamp
            self.star_b = b_psf
            snr = 0.015*np.sum(star_stamp.array)
            saturated = star_stamp.array.max()/100000.>1.
            self.save_star_stamp = True
            if snr<100.:
                self.save_star_stamp = False
            if saturated:
                self.save_star_stamp = False

    def draw_supernova(self):
        
        # Start at the first entry in supernova's lightcurve
        index = self.supernova['ptrobs_min'] - 1
        
        # Figure out how many filters there are and move to the right one
        current_filter = self.lightcurves['flt'][index]
        filt_index = 0
        no_of_filters = 0
        filters = []
        while current_filter not in filters:
            if current_filter == self.pointing.filter[0]:
                filt_index = index
            filters.append(current_filter)
            no_of_filters += 1
            index += 1
            current_filter = self.lightcurves['flt'][index]
        # Move through the entries with the right folder looking for the right date
        current_date = self.lightcurves['mjd'][filt_index]
        while current_date <= self.pointing.mjd and filt_index <= self.supernova['ptrobs_max'] - 1 - no_of_filters:
            filt_index += no_of_filters
            current_date = self.lightcurves['mjd'][filt_index]
        # Find the two entries corresponding to dates immediately before and after the supernova observation date
        flux1 = 10 ** ((27.5 - self.lightcurves['sim_magobs'][filt_index - no_of_filters]) / 2.512)
        flux2 = 10 ** ((27.5 - self.lightcurves['sim_magobs'][filt_index]) / 2.512)
        # Interpolate the flux between the two dates (linear for now)
        flux = np.interp(self.pointing.mjd, [self.lightcurves['mjd'][filt_index - no_of_filters], current_date], [flux1, flux2])
        # This probably isn't necessary but doesn't hurt anything
        if flux <= 0.0:
            magnitude = 100
        else:
            magnitude = 27.5 - (2.512 * math.log10(flux))
        self.ind = self.supernova['snid']
        self.mag = magnitude
        self.hostid = self.supernova['hostgal_objid']
        print('remember to get real supernova sed')
            
        gsparams = self.star_model(sed=self.supernova_sed,mag=magnitude)

        # Get good stamp size multiple for supernova
        # stamp_size = self.get_stamp_size(self.st_model)#.withGSParams(gsparams))
        stamp_size = 256

        # Create postage stamp bounds for supernova
        # b = galsim.BoundsI( xmin=self.xyI.x-int(stamp_size*self.stamp_size)/2,
        #                     ymin=self.xyI.y-int(stamp_size*self.stamp_size)/2,
        #                     xmax=self.xyI.x+int(stamp_size*self.stamp_size)/2,
        #                     ymax=self.xyI.y+int(stamp_size*self.stamp_size)/2 )
        b = galsim.BoundsI( xmin=self.xyI.x-int(stamp_size/2),
                            ymin=self.xyI.y-int(stamp_size/2),
                            xmax=self.xyI.x+int(stamp_size/2),
                            ymax=self.xyI.y+int(stamp_size/2) )

        # If postage stamp doesn't overlap with SCA, don't draw anything
        if not (b&self.b).isDefined():
            return

        # Create star postage stamp
        star_stamp = galsim.Image(b, wcs=self.pointing.WCS)

        # Draw star model into postage stamp
        self.st_model.drawImage(self.pointing.bpass,image=star_stamp,offset=self.offset,method='phot',rng=self.rng,maxN=1000000)

        # star_stamp.write('/fs/scratch/cond0083/roman_sim_out/images/'+str(self.ind)+'.fits.gz')

        # Add star stamp to SCA image
        self.im[b&self.b] = self.im[b&self.b] + star_stamp[b&self.b]
        # self.st_model.drawImage(image=self.im,add_to_image=True,offset=self.xy-self.im.true_center,method='phot',rng=self.rng,maxN=1000000)

        if self.b.includes(self.xyI):
            self.supernova_stamp = star_stamp

    def retrieve_stamp(self):
        """
        Helper function to accumulate various information about a postage stamp and return it in dictionary form.
        """

        # print('stamp size',self.gal_stamp_too_large,self.gal_stamp)

        if self.gal_stamp is None:
            return None

        if self.gal_stamp_too_large:
            # stamp size too big
            return {'ind'    : self.ind, # truth index
                    'ra'     : self.gal['ra'], # ra of galaxy
                    'dec'    : self.gal['dec'], # dec of galaxy
                    'x'      : self.xy.x, # SCA x position of galaxy
                    'y'      : self.xy.y, # SCA y position of galaxy
                    'dither' : self.pointing.dither, # dither index
                    'mag'    : self.mag, #Calculated magnitude
                    'stamp'  : self.stamp_size, # Get stamp size in pixels
                    'gal'    : None, # Galaxy image object (includes metadata like WCS)
                    # 'psf'    : None, # Flattened array of PSF image
                    # 'psf2'    : None, # Flattened array of PSF image
                    'weight' : None } # Flattened array of weight map

        return {'ind'    : self.ind, # truth index
                'ra'     : self.gal['ra'], # ra of galaxy
                'dec'    : self.gal['dec'], # dec of galaxy
                'x'      : self.xy.x, # SCA x position of galaxy
                'y'      : self.xy.y, # SCA y position of galaxy
                'dither' : self.pointing.dither, # dither index
                'mag'    : self.mag, #Calculated magnitude
                'stamp'  : self.stamp_size, # Get stamp size in pixels
                'b'  : self.gal_b, # Galaxy bounds object
                'gal'    : self.gal_stamp, # Galaxy image object (includes metadata like WCS)
                'gal_model' : self.gal_model,
                # 'psf'    : self.psf_stamp.array.flatten(), # Flattened array of PSF image
                # 'psf2'   : self.psf_stamp2.array.flatten(), # Flattened array of PSF image
                'weight' : self.weight } # Flattened array of weight map

    def retrieve_star_stamp(self):

        if self.star_stamp is None:
            return None

        if self.save_star_stamp:

            return {'ind'    : self.ind, # truth index
                    'ra'     : self.star['ra'], # ra of galaxy
                    'dec'    : self.star['dec'], # dec of galaxy
                    'x'      : self.xy.x, # SCA x position of galaxy
                    'y'      : self.xy.y, # SCA y position of galaxy
                    'dither' : self.pointing.dither, # dither index
                    'mag'    : self.mag, #Calculated magnitude
                    'b'      : self.star_b, # Galaxy bounds object
                    'weight' : self.weight,
                    'star'   : self.star_stamp} 
        else:

            return {'ind'    : self.ind, # truth index
                    'ra'     : self.star['ra'], # ra of galaxy
                    'dec'    : self.star['dec'], # dec of galaxy
                    'x'      : self.xy.x, # SCA x position of galaxy
                    'y'      : self.xy.y, # SCA y position of galaxy
                    'dither' : self.pointing.dither, # dither index
                    'mag'    : self.mag, #Calculated magnitude
                    'star'   : None}

    
    def retrieve_supernova_stamp(self):
        
        if self.supernova_stamp is None:
            return None
        
        return {'ind'    : self.ind, # truth index
                'ra'     : self.supernova['ra'], # ra of supernova
                'dec'    : self.supernova['dec'], # dec of supernova
                'x'      : self.xy.x, # SCA x position of supernova
                'y'      : self.xy.y, # SCA y position of supernova
                'dither' : self.pointing.dither, # dither index
                'mag'    : self.mag, #Calculated magnitude
                'hostid' : self.hostid, #Host galaxy id number
                'supernova'    : self.supernova_stamp } # Supernova image object (includes metadata like WCS)

    def finalize_sca(self):
        """
        # Apply background, noise, and Roman detector effects to SCA image
        # Get final SCA image and weight map
        """

        # World coordinate of SCA center
        radec = self.pointing.WCS.toWorld(galsim.PositionI(int(roman.n_pix/2),int(roman.n_pix/2)))
        # Apply background, noise, and Roman detector effects to SCA image and return final SCA image and weight map
        return self.modify_image.add_effects(self.im,self.pointing,radec,self.pointing.WCS,self.rng,phot=True, ps_save=True)



class draw_detector(draw_image):
    def __init__(self, params, pointing, modify_image, cats, logger, image_buffer=1024, rank=0, comm=None, im=None):
        super().__init__(params, pointing, modify_image, cats, logger, image_buffer=image_buffer, rank=rank, comm=comm)

        self.im = galsim.Image(im, bounds=self.b, wcs=self.pointing.WCS)

    def finalize_stamp(self,gal,obj='gal'):
        """
        # Apply background, noise, and Roman detector effects to an image stamp
        # Get final image stamp and weight map
        """

        # World coordinate of SCA center
        radec = self.pointing.WCS.toWorld(galsim.PositionD(gal['x'],gal['y']))
        # Apply background, noise, and Roman detector effects to SCA image and return final SCA image and weight map
        im_,wt_,dq_ = self.modify_image.add_effects(gal[obj][gal['b']&self.b],self.pointing,radec,self.pointing.WCS,self.rng,phot=True, ps_save=True)
        im = galsim.Image(gal['b'], wcs=self.pointing.WCS)
        im[gal['b']&self.b] = im_
        wt = galsim.Image(gal['b'], wcs=self.pointing.WCS)
        wt[gal['b']&self.b] = galsim.Image(wt_,xmin=gal[obj].xmin,ymin=gal[obj].ymin)
        wt            = wt.array
        dq            = galsim.Image(gal['b'], wcs=self.pointing.WCS,init_value=4)
        dq[gal['b']&self.b] = galsim.Image(dq_,xmin=gal[obj].xmin,ymin=gal[obj].ymin)
        dq            = dq.array

        return im,wt,dq
