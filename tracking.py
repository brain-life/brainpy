import time
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.viz import fvtk, actor, window
from dipy.viz.colormap import line_colors
from dipy.tracking import utils
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.tracking.local import ThresholdTissueClassifier
from dipy.tracking.local import LocalTracking
from nibabel.streamlines import Tractogram, save
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.direction import ProbabilisticDirectionGetter


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

"""
data_path = '/N/dc2/projects/lifebid/HCP7/108323/'
data_file = data_path + 'original_hcp_data/Diffusion_7T/' + 'data.nii.gz'
data_bvec = data_path + 'original_hcp_data/Diffusion_7T/' + 'bvecs'
data_bval = data_path + 'original_hcp_data/Diffusion_7T/' + 'bvals'
data_brainmask = data_path + 'original_hcp_data/Diffusion_7T/' + 'nodif_brain_mask.nii.gz'

fs_path = '/N/dc2/projects/lifebid/HCP/Brent/7t_rerun/freesurfer/7t_108323'
data_fs_seg = fs_path + '/mri/aparc+aseg.nii.gz'


# Load the data
start = time.time()
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
print('Loaded Files:' + str((end-start)))
"""

def load_dmri(data_file):
    # Load the dmri
    start = time.time()
    dmri_image = nib.load(data_file)
    print("Loaded dmri: " + str(time.time() - start))
    return dmri_image.get_data()


def load_affine(data_file):
    # Load affine
    start = time.time()
    dmri_image = nib.load(data_file)
    print("Loaded affine: " + str(time.time() - start))
    return dmri_image.affine


def load_aparc(data_fs_seg):
    # Loads the segmentation file
    start = time.time()
    aparc_im = nib.load(data_fs_seg)
    print("Loaded aparc: " + str(time.time() - start))
    return aparc_im.get_data()


def create_wm_mask(aparc):
    # Create the white matter
    start = time.time()
    wm_regions = [2, 41, 16, 17, 28, 60, 51, 53, 12, 52, 12, 52, 13, 18,
                  54, 50, 11, 251, 252, 253, 254, 255, 10, 49, 46, 7]

    wm_mask = np.zeros(aparc.shape)
    for l in wm_regions:
        wm_mask[aparc == l] = 1
    end = time.time()
    print("Created wm mask: " + str((end - start)))
    return wm_mask


def create_callosal_mask(data_fs_seg):
    # Create the callosal mask
    start = time.time()
    aparc_im = nib.load(data_fs_seg)
    aparc = aparc_im.get_data()
    callosal_regions = [255, 254, 253]
    callosum = np.zeros(aparc.shape)
    for c in callosal_regions:
        callosum[aparc == c] = 1
    end = time.time()
    print("Created masks: " + str((end - start)))
    return callosum


# bvals = np.loadtxt(data_bval,delimiter=',')
# bvecs = np.loadtxt(data_bvec).T

def create_gradient_table(data_bval, data_bvec):
    # Create the gradient table from the bvals and bvecs
    start = time.time()
    bvals, bvecs = read_bvals_bvecs(data_bval, data_bvec)
    gtab = gradient_table(bvals, bvecs, b0_threshold=100)
    end = time.time()
    print('Created Gradient Table: ' + str((end - start)))
    return gtab


##The deterministic model##

def create_CSA_model(gtab):
    # Use the Constant Solid Angle (CSA) to find the Orientation Dist. Function
    # Helps orient the wm tracts
    start = time.time()
    csa_model = CsaOdfModel(gtab, sh_order=6)
    end = time.time()
    print('Creating CSA Model: ' + str((end - start)))
    return csa_model


def create_peaks(model, dmri, wm_mask):
    # Find the peaks from the CSA model
    peaks = peaks_from_model(model, dmri, default_sphere,
                             relative_peak_threshold=.8,
                             min_separation_angle=45,
                             mask=wm_mask)
    return peaks


def create_classifier(peaks):
    # Restricts the fiber tracking to the restricted diffusion
    start = time.time()
    classifier = ThresholdTissueClassifier(peaks.gfa, .1)
    end = time.time()
    print('Created Tissue classifer: ' + str((end - start)))
    return classifier


def create_wm_seeds(wm_mask, affine):
    # Begins the seed in the wm tracts
    start = time.time()
    seeds = utils.seeds_from_mask(wm_mask, density=[1, 1, 1], affine=affine)
    end = time.time()
    print('Created White Matter seeds: ' + str((end - start)))
    return seeds


def compute_streamlines(peaks, classifier, seeds, affine):
    # Initialization of LocalTracking. The computation happens in the next step.
    start = time.time()
    streamlines = LocalTracking(peaks, classifier, seeds, affine, step_size=.5)

    # Compute streamlines and store as a list.
    streamlines = list(streamlines)
    print("Number of streamlines " + str(len(streamlines)))
    end = time.time()
    print("Computed streamlines " + str((end - start)))
    return streamlines


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

# Save it as a trk file for vis
from dipy.io.* import save, Tractogram
save(Tractogram(streamlines, affine), 'csa_detr.trk')
save(Tractogram(streamlines, affine), 'csa_detr.trk')
print('End the deterministic model')
"""


def create_trk(streamlines, affine, name):
    start = time.time()
    tractogram = Tractogram(streamlines, affine_to_rasmm=affine)
    save(tractogram, name + '.trk')
    end = time.time()
    print("Created the trk file: " + str((end - start)))


##The probabilistic model##

def create_CSD_model(gtab, dmri):
    # Create a CSD model to measure Fiber Orientation Dist, returns CSD model fit
    start = time.time()
    response, ratio = auto_response(gtab, dmri, roi_radius=10, fa_thr=0.7)
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
    end = time.time()
    print ('Created the CSD model: ' + str((end - start)))
    return csd_model


# def create_fit(model, dmri):
#     start = time.time()
#     csd_fit = model.fit(dmri, mask=wm_mask)
#     end = time.time()
#     print('Created the CSD model fit: ' (end-start))
#     return csd_fit

def prob_direction_getter(csd_model, dmri, wm_mask):
    # Set the Direction Getter to randomly choose directions
    start = time.time()
    csd_fit = csd_model.fit(dmri, mask=wm_mask)
    prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                        max_angle=30.,
                                                        sphere=default_sphere)
    end = time.time()
    print('Created the Direction Getter: ' + str((end - start)))
    return prob_dg


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


def make_picture(name, streamlines):
    # Prepare the display objects.
    start = time.time()
    # color = line_colors(streamlines)
    print("Making pretty pictures")
    if fvtk.have_vtk:
        streamlines_actor = fvtk.line(streamlines, line_colors(streamlines))

        # Create the 3d display.
        r = fvtk.ren()
        fvtk.add(r, streamlines_actor)

        # Save still images for this static example.
        fvtk.record(r, n_frames=1, out_path=name + '.png',
                    size=(800, 800))
    end = time.time()
    print ('Made pretty pictures: ' + str((end - start)))


def main(det):
    start = time.time()
    data_path = '/N/dc2/projects/lifebid/HCP7/108323/'
    data_file = data_path + 'original_hcp_data/Diffusion_7T/' + 'data.nii.gz'
    data_bvec = data_path + 'original_hcp_data/Diffusion_7T/' + 'bvecs'
    data_bval = data_path + 'original_hcp_data/Diffusion_7T/' + 'bvals'
    #data_brainmask = data_path + 'original_hcp_data/Diffusion_7T/' + 'nodif_brain_mask.nii.gz'
    fs_path = '/N/dc2/projects/lifebid/HCP/Brent/7t_rerun/freesurfer/7t_108323'
    data_fs_seg = fs_path + '/mri/aparc+aseg.nii.gz'
    print('Set paths: ' + str(time.time() - start))

    # Load the data
    dmri = load_dmri(data_file)
    affine = load_affine(data_file)
    aparc = load_aparc(data_fs_seg)

    # Create masks
    wm_mask = create_wm_mask(aparc)
    # callosal_mask = create_callosal_mask(aparc, [253, 254, 255])

    # Create gradient table
    gtab = create_gradient_table(data_bval, data_bvec)

    # Create CSA model & peaks
    csa_model = create_CSA_model(gtab)
    csa_peaks = create_peaks(csa_model, dmri, wm_mask)

    # Create classifier
    classifier = create_classifier(csa_peaks)

    if det:

        # Create seeds
        seeds = create_wm_seeds(wm_mask, affine)

        # Compute streamlines
        streamlines = compute_streamlines(peaks=csa_peaks, classifier=classifier, seeds=seeds, affine=affine)

        # Save trk
        create_trk(streamlines, affine, 'csa_det')

        # Save png file
        make_picture('deterministic', streamlines)

    else:
        # Create CSD model
        csd_model = create_CSD_model(gtab, dmri)

        # Create the probabilistic direction getter
        prob_dg = prob_direction_getter(csd_model, dmri, wm_mask)

        # Create seeds
        seeds = create_wm_seeds(wm_mask, affine)

        # Compute streamlines
        streamlines = compute_streamlines(peaks=prob_dg, classifier=classifier, seeds=seeds, affine=affine)

        # Save trk
        create_trk(streamlines, affine, 'csa_prob')

        # Save png file
        make_picture('probabilistic', streamlines)

    end = time.time()
    print('Finished!' + (end - start))


main(det=False)
