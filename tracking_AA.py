"""
==============================
Introduction to Basic Tracking
==============================

Local fiber tracking is an approach used to model white matter fibers by
creating streamlines from local directional information. The idea is as
follows: if the local directionality of a tract/pathway segment is known, one
can integrate along those directions to build a complete representation of that
structure. Local fiber tracking is widely used in the field of diffusion MRI
because it is simple and robust.

In order to perform local fiber tracking, three things are needed: 1) A method
for getting directions from a diffusion data set. 2) A method for identifying
different tissue types within the data set. 3) A set of seeds from which to
begin tracking.  This example shows how to combine the 3 parts described above
to create a tractography reconstruction from a diffusion data set.
"""

"""
To begin, let's load an example HARDI data set from Stanford. If you have
not already downloaded this data set, the first time you run this example you
will need to be connected to the internet and this dataset will be downloaded
to your computer.
"""
import numpy as np
import nibabel as nib
import dipy
from dipy.core.gradients import gradient_table
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors


data_path = '/N/dc2/projects/lifebid/HCP7/108323/'
data_file = data_path + 'diffusion_data/' + 'data.nii.gz'
data_bvec = data_path + 'diffusion_data/' + 'data.bvec'
data_bval = data_path + 'diffusion_data/' + 'data.bval'
data_brainmask = data_path + 'diffusion_data/' + 'nodif_brain_mask.nii.gz'

fs_path = '/N/dc2/projects/lifebid/HCP/Brent/7t_rerun/freesurfer/7t_108323'
data_fs_seg = fs_path + '/mri/aparc+aseg+LAS.nii.gz'

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

print 'Loaded Files'

# from dipy.data import read_stanford_labels
#
# hardi_img, gtab, labels_img = read_stanford_labels()
# data = hardi_img.get_data()
labels = labels_img.get_data()
# affine = hardi_img.get_affine()


"""
This dataset provides a label map in which all white matter tissues are
labeled either 1 or 2. Lets create a white matter mask to restrict tracking to
the white matter.
"""

wm_regions = [2, 41, 16, 17, 28, 60, 51, 53, 12, 52, 12, 52, 13, 18,
              54, 50, 11, 251, 252, 253, 254, 255, 10, 49, 46, 7]

wm_mask = np.zeros(aparc.shape)
for l in wm_regions:
    wm_mask[aparc == l] = 1

callosal_regions = [255, 254, 253]
callosum = np.zeros(aparc.shape)
for c in callosal_regions:
    callosum[aparc==c] = 1

print 'Created Masks'

bvals = np.loadtxt(data_bval,delimiter=',')
bvecs = np.loadtxt(data_bvec).T
gtab = gradient_table(bvals, bvecs, b0_threshold=100)

print 'Created Gradient Table'
"""
1. The first thing we need to begin fiber tracking is a way of getting
directions from this diffusion data set. In order to do that, we can fit the
data to a Constant Solid Angle ODF Model. This model will estimate the
orientation distribution function (ODF) at each voxel. The ODF is the
distribution of water diffusion as a function of direction. The peaks of an ODF
are good estimates for the orientation of tract segments at a point in the
image.
"""

from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model

#change model
csa_model = CsaOdfModel(gtab, sh_order=6)
csa_peaks = peaks_from_model(csa_model, dmri, default_sphere,
                             relative_peak_threshold=.8,
                             min_separation_angle=45,
                             mask=wm_mask)
print 'Creating CSA Model'

"""
2. Next we need some way of restricting the fiber tracking to areas with good
directionality information. We've already created the white matter mask,
but we can go a step further and restrict fiber tracking to those areas where
the ODF shows significant restricted diffusion by thresholding on
the general fractional anisotropy (GFA).
"""

from dipy.tracking.local import ThresholdTissueClassifier

classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)

print 'Created Threshold Tissue classifer'

"""
3. Before we can begin tracking is to specify where to "seed" (begin) the fiber
tracking. Generally, the seeds chosen will depend on the pathways one is
interested in modeling. In this example, we'll use a 2x2x2 grid of seeds per
voxel, in a sagittal slice of the Corpus Callosum.  Tracking from this region
will give us a model of the Corpus Callosum tract.  This slice has label value
2 in the labels image.
"""

from dipy.tracking import utils

seeds = utils.seeds_from_mask(callosum, density=[2, 2, 2], affine=affine)

print 'Created Callosum seeds'

"""
Finally, we can bring it all together using ``LocalTracking``. We will then
display the resulting streamlines using the fvtk module.
"""

from dipy.tracking.local import LocalTracking
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

# Initialization of LocalTracking. The computation happens in the next step.
streamlines = LocalTracking(csa_peaks, classifier, seeds, affine, step_size=.5)

# Compute streamlines and store as a list.
streamlines = list(streamlines)

# Prepare the display objects.
color = line_colors(streamlines)

if fvtk.have_vtk:
    streamlines_actor = fvtk.line(streamlines, line_colors(streamlines))

    # Create the 3d display.
    r = fvtk.ren()
    fvtk.add(r, streamlines_actor)

    # Save still images for this static example. Or for interactivity use
    # fvtk.show
    fvtk.record(r, n_frames=1, out_path='deterministic.png',
                size=(800, 800))

"""
.. figure:: deterministic.png
   :align: center

   **Corpus Callosum Deterministic**

We've created a deterministic set of streamlines, so called because if you
repeat the fiber tracking (keeping all the inputs the same) you will get
exactly the same set of streamlines. We can save the streamlines as a Trackvis
file so it can be loaded into other software for visualization or further
analysis.
"""

from dipy.io.trackvis import save_trk
#save_trk("CSA_detr.trk", streamlines, affine, labels.size)
save_trk("CSA_detr.trk", streamlines, affine)

"""
Next let's try some probabilistic fiber tracking. For this, we'll be using the
Constrained Spherical Deconvolution (CSD) Model. This model represents each
voxel in the data set as a collection of small white matter fibers with
different orientations. The density of fibers along each orientation is known
as the Fiber Orientation Distribution (FOD). In order to perform probabilistic
fiber tracking, we pick a fiber from the FOD at random at each new location
along the streamline. Note: one could use this model to perform deterministic
fiber tracking by always tracking along the directions that have the most
fibers.

Let's begin probabilistic fiber tracking by fitting the data to the CSD model.
"""

from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)

response, ratio = auto_response(gtab, dmri, roi_radius=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
csd_fit = csd_model.fit(dmri, mask=wm_mask)

"""
Next we'll need to make a ``ProbabilisticDirectionGetter``. Because the CSD
model represents the FOD using the spherical harmonic basis, we can use the
``from_shcoeff`` method to create the direction getter. This direction getter
will randomly sample directions from the FOD each time the tracking algorithm
needs to take another step.
"""

from dipy.direction import ProbabilisticDirectionGetter

prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                    max_angle=30.,
                                                    sphere=default_sphere)

"""
As with deterministic tracking, we'll need to use a tissue classifier to
restrict the tracking to the white matter of the brain. One might be tempted
to use the GFA of the CSD FODs to build a tissue classifier, however the GFA
values of these FODs don't classify gray matter and white matter well. We will
therefore use the GFA from the CSA model which we fit for the first section of
this example. Alternatively, one could fit a ``TensorModel`` to the data and use
the fractional anisotropy (FA) to build a tissue classifier.
"""

classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)

"""
Next we can pass this direction getter, along with the ``classifier`` and
``seeds``, to ``LocalTracking`` to get a probabilistic model of the corpus
callosum.
"""

streamlines = LocalTracking(prob_dg, classifier, seeds, affine,
                            step_size=.5, max_cross=1)

# Compute streamlines and store as a list.
streamlines = list(streamlines)

# Prepare the display objects.
color = line_colors(streamlines)

if fvtk.have_vtk:
    streamlines_actor = fvtk.line(streamlines, line_colors(streamlines))

    # Create the 3d display.
    r = fvtk.ren()
    fvtk.add(r, streamlines_actor)

    # Save still images for this static example.
    fvtk.record(r, n_frames=1, out_path='probabilistic.png',
                size=(800, 800))

"""
.. figure:: probabilistic.png
   :align: center

   **Corpus Callosum Probabilistic**
"""

save_trk("CSD_prob.trk", streamlines, affine)
