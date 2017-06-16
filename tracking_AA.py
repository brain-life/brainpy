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

data_path = '/N/dc2/projects/lifebid/HCP7/108323/'
data_file = data_path + 'original_hcp_data/Diffusion_7T/' + 'data.nii.gz'
data_bvec = data_path + 'original_hcp_data/Diffusion_7T/' + 'bvecs'
data_bval = data_path + 'original_hcp_data/Diffusion_7T/' + 'bvals'
data_brainmask = data_path + 'original_hcp_data/Diffusion_7T/' + 'nodif_brain_mask.nii.gz'

fs_path = '/N/dc2/projects/lifebid/HCP/Brent/7t_rerun/freesurfer/7t_108323'
data_fs_seg = fs_path + '/mri/aparc+aseg.nii.gz'

def prob_model(data_file, data_bvec, data_bval, data_fs_seg, data_brainmask):
    start = time.time()

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
    print('Loaded Files: ' + str((end - start)))

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

    bvals, bvecs = read_bvals_bvecs(data_bval, data_bvec)

    gtab = gradient_table(bvals, bvecs, b0_threshold=100)
    end = time.time()
    print('Created Gradient Table: ' + str((end - start)))

    ##The probabilistic model##

    # Use the Constant Solid Angle (CSA) to find the Orientation Dist. Function
    # Helps orient the wm tracts
    start = time.time()
    csa_model = CsaOdfModel(gtab, sh_order=6)
    csa_peaks = peaks_from_model(csa_model, dmri, default_sphere,
                                 relative_peak_threshold=.8,
                                 min_separation_angle=45,
                                 mask=wm_mask)
    print('Creating CSA Model: ' + str(time.time() - start))

    # Begins the seed in the wm tracts
    seeds = utils.seeds_from_mask(wm_mask, density=[1, 1, 1], affine=affine)
    print('Created White Matter seeds: ' + str(time.time() - start))

    # Create a CSD model to measure Fiber Orientation Dist
    print('Begin the probabilistic model')

    response, ratio = auto_response(gtab, dmri, roi_radius=10, fa_thr=0.7)
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
    csd_fit = csd_model.fit(dmri, mask=wm_mask)
    print ('Created the CSD model: ' + str(time.time() - start))

    # Set the Direction Getter to randomly choose directions

    prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                        max_angle=30.,
                                                        sphere=default_sphere)
    print('Created the Direction Getter: ' + str(time.time() - start))

    # Restrict the white matter tracking
    classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)

    print('Created the Tissue Classifier: ' + str(time.time() - start))

    # Create the probabilistic model
    streamlines = LocalTracking(prob_dg, tissue_classifier=classifier, seeds=seeds, affine=affine,
                                step_size=.5, max_cross=1)
    print('Created the probabilistic model: ' + str(time.time() - start))

    # Compute streamlines and store as a list.
    streamlines = list(streamlines)
    print('Computed streamlines: ' + str(time.time() - start))

    tractogram = Tractogram(streamlines, affine_to_rasmm=affine)
    save(tractogram, 'csa_prob.trk')
    end = time.time()
    print("Created the trk file: " + str((end - start)))

    # Prepare the display objects.
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




prob_model(data_file, data_bvec, data_bval, data_fs_seg, data_brainmask)