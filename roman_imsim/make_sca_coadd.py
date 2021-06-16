import wfirst_imsim

param_file = '/hpc/group/cosmology/masaya/roman_imsim/dc2_H158_g1002.yaml'
filter_ = None
post = wfirst_imsim.postprocessing(param_file)
print(post.params.keys())
post.setup_pointing()
#post.merge_fits_old()
# post.accumulate_index()
# post.get_psf_fits()
post.load_index()
post.get_coadd()