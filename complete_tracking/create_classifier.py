import time
from dipy.tracking.local import ThresholdTissueClassifier
import csd_peaks

def create_classifier(peaks):
    # Restricts the fiber tracking to the restricted diffusion
    start = time.time()
    classifier = ThresholdTissueClassifier(peaks.gfa, .1)
    end = time.time()
    print('Created Tissue classifer: ' + str((end - start)))
    return classifier

def classifier():
    return create_classifier(csd_peaks.csd_peaks())
