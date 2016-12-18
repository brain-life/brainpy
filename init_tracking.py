# The code below is an example of how to track using Dipy on data on Karst
#
# The script assumes that your environment is set up correctly. 
# 
# This script will require the following set of commands:
#
# (1) module unload python
# (2) module load anaconda2
# (3) git clone git@github.com:nipy/dipy.git
# (4) git clone git@github.com:nipy/nibabel.git 
# (5) Make sure that Dipy is on the PYTHONPATH:
#     (5.1) add Dipy to the shell environment
#           Pythonpath is an environmental 
#           variable that keeps tracks of 
#           what Python can access to.
#     export PYTHONPATH = 
#            "/N/dc2/projects/lifebid/code/franpest/dipy/:
#            /N/dc2/projects/lifebid/code/franpest/nibabel:
#            $PYTHONPATH"
#     (5.2) cd dipy/
#           make clean # makes a
#           make ext # compiles all the files in dipy that 
#                    - needs compilations, compiles them and
#                    - saves the compiled copies loacally under
#                    - dipy (not on the system level).
#                    - This is the command being called:
#                    - python setup.py built_ext -inplace)
#
#      (5.3) open IPython; in the shell type ipython
#
#
# Franco Pestilli 

# Import tools to environment
import numpy as np
import nibabel as nib 

# Load methods from Dipy for tracking and signal voxel reconstruction
from dipy.reconst.dti   import TensorModel, fractional_anisotropy
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.direction import peaks_from_model
from dipy.tracking.eudx import EuDX
from dipy.data import get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table 

# A few requirements for visualizing the outputs (VTK)
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

# We initialize pointers to file paths
data_path = '/N/dc2/projects/lifebid/2t1/HCP/105115/diffusion_data/'
data_file = data_path + 'dwi_data_b1000_aligned_trilin.nii.gz'
data_bvec = data_path + 'dwi_data_b1000_aligned_trilin.bvecs'
data_bval = data_path + 'dwi_data_b1000_aligned_trilin.bvals'
data_brainmask = data_path + '../anatomy/' + 'wm_mask_dwi_res.nii.gz'

# Load the data
nifti_image = nib.load(data_file) 
dmri = nifti_image.get_data() 
affine = nifti_image.affine
brainmask = nib.load(data_brainmask).get_data()


# Load the bvals and bvecs from disk
# ideally we should use the following:
#   bvals, bvecs = read_bvals_bvecs(data_bval, data_bvec)
# in practice this data set does not conform to a standard (FSL)
bvals = np.loadtxt(data_bval)
bvecs = np.loadtxt(data_bvec)
gtab = gradient_table(bvals, bvecs)

# After loading all files we can start voxel reconstruction

# Estimate impulse response for deconvolution (check that the estimation is good)
# http://nipy.org/dipy/examples_built/reconst_csd.html
response, ratio = auto_response(gtab, dmri, roi_radius=10, fa_thr=0.6)

# Build a CSD model kernek
csd_model = ConstrainedSphericalDeconvModel(gtab, response);

# Build the Diffusion Tensor Model.
dt_model = TensorModel(gtab)

# Initialize a sphere 
sphere = get_sphere('repulsion724')
# The following line estimates a series of coefficients and values of the 
# ODF. 
#
# OUTPUT: 
#   csd_peaks is an object containing 'peaks_directions', 'peaks_values'
#   'sph_coeff'
#
# INPUTs:
#   data = diffusion weighted signal volume
#   sphere = the sphere object will indicate a high-resolution 
#            sphereical set of points where ODF and all functions 
#            are evaluated
#   mask   = brain mask or white matter mask
#   relative_peak_threshold = the cutoff (proportion max amplitude) 
#                             of peaks accepted as peak
#   min_angle_separation    = 
#   parallel = set processes to use OpenMP
#   nbr_processes = max number of processes
csd_peaks = peaks_from_model(model=csd_model,
                             data=dmri,
                             sphere=sphere,
                             mask=brainmask,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             parallel=True, 
                             nbr_processes=4)
print('done estimating CSD_peaks')

# Fit the diffusion tensor model to the data
dt_fit = dt_model.fit(data=dmri, mask=brainmask)
print('done estimating DTI')

# Next we visualize the peaks in a single brain slice
ren            = fvtk.ren()
slice = 70
fodf_peaks = fvtk.peaks(csd_peaks.peak_dirs[:,:,slice], csd_peaks.peak_values[:,:,slice], scale=1.3)
fvtk.add(ren, fodf_peaks)
fvtk.show(ren)
ren.azimuth(90)
fvtk.show(ren)

# We now classify the tissue to decide where to stop tracking.
from dipy.tracking.local import ThresholdTissueClassifier
classifier = ThresholdTissueClassifier(dt_fit.fa, .25)

# We will need some seeds for tracking. There are many different functions for seeds.
# Hereafter we will use seeds from a precomputed mask, for us that mask will be the WM mask
from dipy.tracking import utils
seeds = utils.seeds_from_mask(brainmask,density=1)
seeds_im = fvtk.dots(seeds, (1, 0.5, 0))
fvtk.add(ren,seeds_im)

# Now we have prepared all we need and we can track
from dipy.tracking.local import LocalTracking

# Creating the tracking model
tractogram = LocalTracking(csd_peaks, classifier, seeds, affine=np.eye(4), step_size=.5)

# Tracking
tractogram = list(tractogram)

# visualzie the streamlines
tractogram_actor = fvtk.line(tractogram)
fvtk.rm(ren,fodf_peaks)
fvtk.rm(ren, seeds_im)
fvtk.add(ren,tractogram_actor)
fvtk.show(ren)

# For the next time we will save to disk.

# After that we will do Anatomically Constrained Tracking.

# END 



