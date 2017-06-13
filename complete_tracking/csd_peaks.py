import time
from dipy.direction import peaks_from_model
from dipy.data import default_sphere
import load_file
import create_csd
import create_mask

def create_peaks(model, dmri, wm_mask):
    # Find the peaks from the CSA model
    start = time.time()
    peaks = peaks_from_model(model, dmri, default_sphere,
                             relative_peak_threshold=.8,
                             min_separation_angle=45,
                             mask=wm_mask)
    end = time.time()
    print('Created peaks: ' + str(end-start))
    return peaks

def csd_peaks():
    d = load_file.load_files()
    model = create_csd.create_csd()
    wm_mask = create_mask.create_wm_mask()

    return create_peaks(model, load_file.load_dmri(d.data_file), wm_mask)