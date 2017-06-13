import time
from dipy.tracking.local import LocalTracking
import csd_peaks, create_classifier, create_seeds, load_file

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

def streamlines():
    peaks = csd_peaks.csd_peaks()
    classifier = create_classifier.classifier()
    seeds = create_seeds.seeds()
    d = load_file.load_files()
    return compute_streamlines(peaks, classifier, seeds, load_file.load_affine(d.data_file))
