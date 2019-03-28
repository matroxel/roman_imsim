diff --cc wfirst_imsim/simulate.py
index 18981ff,ef8d31e..0000000
--- a/wfirst_imsim/simulate.py
+++ b/wfirst_imsim/simulate.py
@@@ -1512,7 -1512,7 +1512,11 @@@ class draw_image()
  
          # Apply correct flux from magnitude for filter bandpass
          sed_ = sed.atRedshift(self.gal['z'])
++<<<<<<< HEAD
 +        sed_ = sed_.withMagnitude(self.gal[self.pointing.filter], self.bpass)
++=======
+         sed_ = sed_.withMagnitude(self.gal[self.pointing.filter], self.pointing.bpass)
++>>>>>>> 787d4285d6d8374d330e185f16ab9f33589245c7
  
          # Return model with SED applied
          return model * sed_
@@@ -2029,7 -2022,7 +2033,11 @@@ class accumulate_output_disk()
          for i in range(len(self.steps)-1):
              data['box_size'][i] = np.min(self.index['stamp'][self.steps[i]:self.steps[i+1]])
          data['box_size'][i+1]   = np.min(self.index['stamp'][self.steps[-1]:])
++<<<<<<< HEAD
 +        data['psf_box_size'] = np.ones(n_obj)*self.params['psf_stampsize']#*self.params['oversample']
++=======
+         data['psf_box_size'] = np.ones(n_obj)*self.params['psf_stampsize']*self.params['oversample']
++>>>>>>> 787d4285d6d8374d330e185f16ab9f33589245c7
          m.write(data,extname='object_data')
  
          length = np.sum(bincount*data['box_size']**2)
@@@ -2270,23 -2263,23 +2278,40 @@@
                                          wcs.dvdx,
                                          wcs.dvdy)
  
++<<<<<<< HEAD
 +                # psf_wcs = galsim.JacobianWCS(dudx=wcs.dudx/self.params['oversample'],
 +                #                          dudy=wcs.dudy/self.params['oversample'],
 +                #                          dvdx=wcs.dvdx/self.params['oversample'],
 +                #                          dvdy=wcs.dvdy/self.params['oversample'])
 +                # psf = galsim.Image(gals[gal]['psf'].reshape(object_data['psf_box_size'][i],object_data['psf_box_size'][i]), copy=True, wcs=psf_wcs)
 +                # pixel = psf_wcs.toWorld(galsim.Pixel(scale=1))
 +                # ii = galsim.InterpolatedImage(psf)
 +                # psf = ii.drawImage(nx=object_data['box_size'][i], ny=object_data['box_size'][i], wcs=wcs)
++=======
+                 psf_wcs = galsim.JacobianWCS(dudx=wcs.dudx/self.params['oversample'],
+                                          dudy=wcs.dudy/self.params['oversample'],
+                                          dvdx=wcs.dvdx/self.params['oversample'],
+                                          dvdy=wcs.dvdy/self.params['oversample'])
+                 psf = galsim.Image(gals[gal]['psf'].reshape(object_data['psf_box_size'][i],object_data['psf_box_size'][i]), copy=True, wcs=psf_wcs)
+                 pixel = psf_wcs.toWorld(galsim.Pixel(scale=1))
+                 ii = galsim.InterpolatedImage(psf)
+                 psf = ii.drawImage(nx=object_data['box_size'][i], ny=object_data['box_size'][i], wcs=wcs)
++>>>>>>> 787d4285d6d8374d330e185f16ab9f33589245c7
                  self.dump_meds_pix_info(m,
                                          object_data,
                                          i,
                                          j,
                                          gal_,
                                          weight_,
++<<<<<<< HEAD
 +                                        gals[gal]['psf'].array)
 +
 +        # object_data['psf_box_size'] = object_data['box_size']
++=======
+                                         psf.array.flatten())
+ 
+         object_data['psf_box_size'] = object_data['box_size']
++>>>>>>> 787d4285d6d8374d330e185f16ab9f33589245c7
          print 'Writing meds pixel',self.pix
          m['object_data'].write(object_data)
          m.close()
@@@ -2424,6 -2417,6 +2449,59 @@@
  
              flag = runner.fitter.get_result()['flags']
              gmix = runner.fitter.get_gmix()
++<<<<<<< HEAD
++
++            # except Exception as e:
++            #     print 'exception'
++            #     cnt+=1
++            #     continue
++
++            if flag != 0:
++                print 'flag',flag
++                cnt+=1
++                continue
++
++            e1_, e2_, T_ = gmix.get_g1g2T()
++            dx_, dy_ = gmix.get_cen()
++            if (np.abs(e1_) > 0.5) or (np.abs(e2_) > 0.5) or (dx_**2 + dy_**2 > MAX_CENTROID_SHIFT**2):
++                print 'g,xy',e1_,e2_,dx_,dy_
++                cnt+=1
++                continue
++
++            flux_ = gmix.get_flux() / wcs.pixelArea()
++
++            dx   += dx_
++            dy   += dy_
++            e1   += e1_
++            e2   += e2_
++            T    += T_
++            flux += flux_
++
++        if cnt == len(obs_list):
++            return None
++
++        return cnt, dx/(len(obs_list)-cnt), dy/(len(obs_list)-cnt), e1/(len(obs_list)-cnt), e2/(len(obs_list)-cnt), T/(len(obs_list)-cnt), flux/(len(obs_list)-cnt)
++
++    def measure_psf_shape_moments(self,obs_list):
++
++        def make_psf_image(self,obs):
++
++            wcs = self.make_jacobian(obs.psf.jacobian.dudcol,
++                                    obs.psf.jacobian.dudrow,
++                                    obs.psf.jacobian.dvdcol,
++                                    obs.psf.jacobian.dvdrow,
++                                    obs.psf.jacobian.col0,
++                                    obs.psf.jacobian.row0)
++
++            return galsim.Image(obs.psf.image, xmin=1, ymin=1, wcs=wcs)
++
++        out = np.zeros(len(obs_list),dtype=[('e1','f4')]+[('e2','f4')]+[('T','f4')]+[('dx','f4')]+[('dy','f4')]+[('flag','i2')])
++        for iobs,obs in enumerate(obs_list):
++
++            M = e1 = e2= 0
++            im = make_psf_image(self,obs)
++
++=======
  
              # except Exception as e:
              #     print 'exception'
@@@ -2475,6 -2468,6 +2553,7 @@@
              M = e1 = e2= 0
              im = make_psf_image(self,obs)
  
++>>>>>>> 787d4285d6d8374d330e185f16ab9f33589245c7
              try:
                  shape_data = im.FindAdaptiveMom(weight=None, strict=False)
              except:
