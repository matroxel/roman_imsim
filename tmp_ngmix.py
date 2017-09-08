    def get_ngmix_shape(self,filter):
        """
        Takes in the image and psf data and returns g1 and g2
        Currently these are read in from MEDS files, but obviously it would be
        preferred to bypass the extra I/O and take in the existing structures i\
n
        memory...but this requires a bit more modification of the ngmixer scrip\
ts.
        Config file is currently hard-coded, but should have option for read in
        """
        from __future__ import print_function
        from ngmixer.ngmixing import NGMixer
        from ngmixer.mofngmixing import MOFNGMixer
        import ngmixer

        config_file = 'ngmix-wfirst-imsim-config.yaml'
        ngconfig = ngmixer.files.read_yaml(config_file)

        # A lot of options skipped from ngmixit, revisit later whether
        # these are necessary, also currently skipped MOFNGMixer, but we
        # will want to include this option later
        filename = self.params['output_meds']+'_'+filter+'fits.gz'
        outfile = self.params['output_meds']+'_'+filter+'ngmixed.fits'
        NGMixer(
            config_file, filename, output_file=outfile,config=ngconfig)

    def get_range(rng_string):
        """ Lifted from the ngmixit python file """
        if rng_string is not None:
            rng = rng_string.split(',')
            rng = [int(i) for i in rng]
        else:
            rng=None

        return rng
