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
data_path = '/N/dc2/projects/lifebid/HCP7/109123/diffusion_data/'
data_file = data_path + 'data_b1000.nii.gz'
data_bvec = data_path + 'data_b1000.bvecs'
data_bval = data_path + 'data_b1000.bvals'
data_brainmask = data_path + 'nodif_brain_mask.nii.gz'

# Load the data
nifti_image = nib.load(data_file) 
dmri = nifti_image.get_data() 
affine = nifti_image.affine
brainmask = nib.load(data_brainmask).get_data()


# Load the bvals and bvecs from disk
# ideally we should use the following:
#   bvals, bvecs = read_bvals_bvecs(data_bval, data_bvec)
# in practice this data set does not conform to a standard (FSL)
bvals = np.loadtxt(data_bval, delimiter=',')
bvecs = np.loadtxt(data_bvec, delimiter=',')
gtab = gradient_table(bvals, bvecs)

# After loading all files we can start voxel reconstruction

# Estimate impulse response for deconvolution (check that the estimation is good)
# http://nipy.org/dipy/examples_built/reconst_csd.html
response, ratio = auto_response(gtab, dmri, roi_radius=10, fa_thr=0.6)

# Build a CSD model kernek
csd_model = ConstrainedSphericalDeconvModel(gtab, response);
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


