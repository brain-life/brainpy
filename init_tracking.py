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
import dipy as dipy

# Load methods from Dipy for tracking and signal voxel reconstruction
from dipy.reconst.dti   import TensorModel, fractional_anisotropy
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response, response_from_mask)
from dipy.direction import peaks_from_model
from dipy.tracking.eudx import EuDX
from dipy.data import get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table 
from dipy.align.reslice import reslice
# A few requirements for visualizing the outputs (VTK)
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

from dipy.viz import actor, window


# We initialize pointers to file paths
data_path = '/N/dc2/projects/lifebid/HCP7/108323/'
data_file = data_path + 'diffusion_data/'+'data.nii.gz'
data_bvec = data_path + 'diffusion_data/'+'data.bvec'
data_bval = data_path + 'diffusion_data/'+'data.bval'
data_brainmask = data_path + 'diffusion_data/'+'nodif_brain_mask.nii.gz'
data_fs_seg    = data_path + 'anatomy/freesurfer/mri/aparc+aseg.nii.gz'

# Load the data
dmri_image   = nib.load(data_file) 
dmri         = dmri_image.get_data() 
affine       = dmri_image.affine
brainmask_im = nib.load(data_brainmask)
brainmask    = brainmask_im.get_data()
bm_affine    = brainmask_im.affine
aparc_im     = nib.load(data_fs_seg)
aparc        = aparc_im.get_data()
aparc_affine = brainmask_im.affine

# Freesurfer parecellation ROIs
wm_regions = [2, 41, 16, 17, 28, 60, 51, 53, 12, 52, 12, 52, 13, 18,
     54, 50, 11, 251, 252, 253, 254, 255, 10, 49, 46, 7]
     
wm_mask = np.zeros(aparc.shape)
for l in wm_regions:
    wm_mask[aparc==l] = 1

callosal_regions = [255]
callosum = np.zeros(aparc.shape)
for c in callosal_regions:
    callosum[aparc==c] = 1

# reslice the masks to dmri space 
current_zooms = aparc_im.header.get_zooms()[:3]
new_zooms     = dmri_image.header.get_zooms()[:3]
callosum_r, call_r_affine  = reslice(callosum, aparc_affine, 
                                     current_zooms, 
                                     new_zooms)

wm_mask_r, wm_mask_r_affine = reslice(wm_mask, aparc_affine,
                                      current_zooms, 
                                      new_zooms)
show_two_slices(dmri_image, callosum_r, affine, call_r_affine):
    
# Load the bvals and bvecs from disk
# ideally we should use the following:
#   bvals, bvecs = read_bvals_bvecs(data_bval, data_bvec)
# in practice this data set does not conform to a standard (FSL)
bvals = np.loadtxt(data_bval,delimiter=',')
bvecs = np.loadtxt(data_bvec).T
gtab = gradient_table(bvals, bvecs, b0_threshold=100)

# After loading all files we can start voxel reconstruction

# Estimate impulse response for deconvolution (check that the estimation is good)
# http://nipy.org/dipy/examples_built/reconst_csd.html
# response, ratio = response_from_mask(gtab, dmri, callosum)
res, ratio = response_from_mask (gtab, dmri, callosum_r)

# Build a CSD model kernek
csd_model = ConstrainedSphericalDeconvModel(gtab, res);

# Build the Diffusion Tensor Model.
dt_model = TensorModel(gtab)

# Initialize a sphere 
sphere = get_sphere('repulsion724')

# Find fiber-peaks from the CSD fit
csd_peaks = peaks_from_model(model=csd_model,
                             data=dmri,
                             sphere=sphere,
                             mask=brainmask,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             parallel=True, 
                             nbr_processes=10)
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
classifier = ThresholdTissueClassifier(dt_fit.fa, .1)

# We will need some seeds for tracking. There are many different functions for seeds.
# Hereafter we will use seeds from a precomputed mask, for us that mask will be the WM mask
from dipy.tracking import utils
seeds = utils.seeds_from_mask(wm_mask_r, density=1)
seeds_im = fvtk.dots(seeds, (1, 0.5, 0))
#fvtk.add(ren,seeds_im)

# Now we have prepared all we need and we can track
from dipy.tracking.local import LocalTracking

# Creating the tracking model
streamlines = LocalTracking(csd_peaks, classifier, seeds, affine=np.eye(4), step_size=.5)

# Tracking
streamlines = list(streamlines)

# visualzie the streamlines
streamlines_actor = fvtk.line(streamlines[:100000])
#fvtk.rm(ren,fodf_peaks)
#fvtk.rm(ren, seeds_im)
fvtk.add(ren,streamlines_actor)
fvtk.show(ren)

from nibabel.streamlines import Tractogram, save

tractogram = Tractogram(streamlines, affine_to_rasmm=affine)

save(tractogram, 'test.trk')

# For the next time we will save to disk.

# After that we will do Anatomically Constrained Tracking.


def show_slice(volume, affine=None, show_axes=False, k=None):
    ren = window.Renderer()
    slicer_actor = actor.slicer(volume, affine)
    slicer_actor.display(None, None, k)
    ren.add(slicer_actor)
    if show_axes:
        ren.add(actor.axes((100, 100, 100)))
    window.show(ren)

    
def show_two_slices(volume1, affine1, volume2, affine2=None,
                    show_axes=False, k=None):
    ren = window.Renderer()
    slicer_actor = actor.slicer(volume1, affine1)
    slicer_actor.display(None, None, k)
    ren.add(slicer_actor)

    slicer_actor2 = actor.slicer(volume2, affine2)
    slicer_actor2.SetPosition(200, 0, 0)
    slicer_actor2.display(None, None, k)
    ren.add(slicer_actor2)

    
    if show_axes:
        ren.add(actor.axes((100, 100, 100)))
    window.show(ren) 

# END 

