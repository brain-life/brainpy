import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.viz import fvtk, actor, window
from dipy.viz.colormap import line_colors
from dipy.tracking import utils
import time


def show_slice(volume, affine=None, show_axes=False, k=None):
    ren = window.Renderer()
    slicer_actor = actor.slicer(volume, affine)
    slicer_actor.display(None, None, k)
    ren.add(slicer_actor)
    if show_axes:
        ren.add(actor.axes((100, 100, 100)))
    window.show(ren)


def show_two_slices(volume1, affine1, volume2, affine2=None,
                    show_axes=False, k=None, shift=None, opacity=[0.8, 0.4]):
    ren = window.Renderer()
    slicer_actor = actor.slicer(volume1, affine1)
    slicer_actor.display(None, None, k)
    ren.add(slicer_actor)

    slicer_actor2 = actor.slicer(volume2, affine2)
    if shift is not None:
        slicer_actor2.SetPosition(shift, 0, 0)
    slicer_actor2.display(None, None, k)
    ren.add(slicer_actor2)

    if opacity is not None:
        slicer_actor.opacity(opacity[0])
        slicer_actor2.opacity(opacity[1])

    if show_axes:
        ren.add(actor.axes((100, 100, 100)))
    window.show(ren)


def show_streamlines(streamlines, affine):
    ren = window.Renderer()
    line_actor = actor.line(streamlines)
    ren.add(line_actor)
    window.show(ren)


start = time.time()

data_path = '/N/dc2/projects/lifebid/HCP7/108323/'
data_file = data_path + 'original_hcp_data/Diffusion_7T/' + 'data.nii.gz'
data_bvec = data_path + 'original_hcp_data/Diffusion_7T/' + 'bvecs'
data_bval = data_path + 'original_hcp_data/Diffusion_7T/' + 'bvals'
data_brainmask = data_path + 'original_hcp_data/Diffusion_7T/' + 'nodif_brain_mask.nii.gz'

fs_path = '/N/dc2/projects/lifebid/HCP/Brent/7t_rerun/freesurfer/7t_108323'
data_fs_seg = fs_path + '/mri/aparc+aseg.nii.gz'

# Load the data
dmri_image = nib.load(data_file)
dmri = dmri_image.get_data()
affine = dmri_image.affine
brainmask_im = nib.load(data_brainmask)
brainmask = brainmask_im.get_data()
bm_affine = brainmask_im.affine
aparc_im = nib.load(data_fs_seg)
aparc = aparc_im.get_data()
aparc_affine = brainmask_im.affine
end = time.time()
print('Loaded Files1:' + str((end - start)))

# Create the white matter and callosal masks
start = time.time()
wm_regions = [2, 41, 16, 17, 28, 60, 51, 53, 12, 52, 12, 52, 13, 18,
              54, 50, 11, 251, 252, 253, 254, 255, 10, 49, 46, 7]

wm_mask = np.zeros(aparc.shape)
for l in wm_regions:
    wm_mask[aparc == l] = 1

callosal_regions = [255, 254, 253]
callosum = np.zeros(aparc.shape)
for c in callosal_regions:
    callosum[aparc == c] = 1

# bvals = np.loadtxt(data_bval,delimiter=',')
# bvecs = np.loadtxt(data_bvec).T

# Create the gradient table from the bvals and bvecs
from dipy.io.gradients import read_bvals_bvecs

bvals, bvecs = read_bvals_bvecs(data_bval, data_bvec)

gtab = gradient_table(bvals, bvecs, b0_threshold=100)
end = time.time()
print('Created Gradient Table: ' + str((end - start)))

##The deterministic model##

from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model

# Use the Constant Solid Angle (CSA) to find the Orientation Dist. Function
# Helps orient the wm tracts
start = time.time()
csa_model = CsaOdfModel(gtab, sh_order=6)
csa_peaks = peaks_from_model(csa_model, dmri, default_sphere,
                             relative_peak_threshold=.8,
                             min_separation_angle=45,
                             mask=wm_mask)
print('Creating CSA Model ' + str(time.time() - start))
# Restricts the fiber tracking to the restricted diffusion
from dipy.tracking.local import ThresholdTissueClassifier

classifier = ThresholdTissueClassifier(csa_peaks.gfa, .1)

print('Created Tissue classifer ' + str(time.time() - start))

# Begins the seed in the wm tracts
seeds = utils.seeds_from_mask(wm_mask, density=[1, 1, 1], affine=affine)
print('Created White Matter seeds ' + str(time.time() - start))

from dipy.tracking.local import LocalTracking

# Initialization of LocalTracking. The computation happens in the next step.
streamlines = LocalTracking(csa_peaks, classifier, seeds, affine, step_size=.5)

# Compute streamlines and store as a list.
streamlines = list(streamlines)
print(len(streamlines))

"""
# Form an image with the streamlines
color = line_colors(streamlines)
print("Making pretty pictures")
if fvtk.have_vtk:
    streamlines_actor = fvtk.line(streamlines, line_colors(streamlines))
    # Create the 3d display.
    r = fvtk.ren()
    fvtk.add(r, streamlines_actor)
    # Save still images for this static example.
    fvtk.record(r, n_frames=1, out_path='deterministic.png',
                size=(800, 800))
print ('Made pretty pictures')
"""

# Save it as a trk file for vis

# from dipy.io.* import save, Tractogram
# save(Tractogram(streamlines, affine), 'csa_detr.trk')
# print('End the deterministic model')
from nibabel.streamlines import Tractogram, save

tractogram = Tractogram(streamlines, affine_to_rasmm=affine)
save(tractogram, 'csa_detr.trk')
end = time.time()
print("Created the trk file: " + str((end - start)))

##The probabilistic model##

start = time.time()
# Create a CSD model to measure Fiber Orientation Dist
print('Begin the probabilistic model')
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)

response, ratio = auto_response(gtab, dmri, roi_radius=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
csd_fit = csd_model.fit(dmri, mask=wm_mask)
print('CSD_fit is ' + str(type(csd_fit)))
print ('Created the CSD model ' + str(time.time() - start))

# Set the Direction Getter to randomly choose directions
from dipy.direction import ProbabilisticDirectionGetter

prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                    max_angle=30.,
                                                    sphere=default_sphere)
print('Created the Direction Getter ' + str(time.time() - start))
# Restrict the white matter tracking
classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)

print('Created the Tissue Classifier ' + str(time.time() - start))

# Create the probabilistic model
streamlines = LocalTracking(prob_dg, classifier, seeds, affine,
                            step_size=.5, max_cross=1)
print('Created the probabilistic model ' + str(time.time() - start))

# Compute streamlines and store as a list.
print('Computing streamlines ' + str(time.time() - start))
streamlines = list(streamlines)

"""
# Prepare the display objects.
color = line_colors(streamlines)
print("Making pretty pictures")
if fvtk.have_vtk:
    streamlines_actor = fvtk.line(streamlines, line_colors(streamlines))
    # Create the 3d display.
    r = fvtk.ren()
    fvtk.add(r, streamlines_actor)
    # Save still images for this static example.
    fvtk.record(r, n_frames=1, out_path='probabilistic.png',
                size=(800, 800))
print ('Made pretty pictures')
"""

tractogram = Tractogram(streamlines, affine_to_rasmm=affine)
save(tractogram, 'csa_prob.trk')
end = time.time()
print("Created the trk file: " + str((end - start)))