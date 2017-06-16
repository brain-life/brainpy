import time
from dipy.direction import peaks_from_model
from dipy.data import default_sphere
import create_csd
import create_mask
import numpy as np
from dipy.io.peaks import save_peaks

def create_peaks(model, dmri, wm_mask):
    # Find the peaks from the CSA model
    start = time.time()
    peaks = peaks_from_model(model, dmri, default_sphere,
                             relative_peak_threshold=.8,
                             min_separation_angle=45,
                             mask=wm_mask)
    end = time.time()
    save_peaks(peaks)
    print('Created peaks: ' + str(end-start))
    return peaks

def csd_peaks():
    model = create_csd.create_csd()
    wm_mask = create_mask.mask()
    files = np.load('files.npz')

    return create_peaks(model, files['dmri'], wm_mask)


