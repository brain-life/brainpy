import time
from dipy.direction import peaks_from_model

def create_peaks(model, dmri, wm_mask):
    # Find the peaks from the CSA model
    peaks = peaks_from_model(model, dmri, default_sphere,
                             relative_peak_threshold=.8,
                             min_separation_angle=45,
                             mask=wm_mask)
    return peaks