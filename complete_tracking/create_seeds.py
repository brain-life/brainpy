import time
from dipy.tracking import utils
import create_mask
import numpy as np

def create_wm_seeds(wm_mask, affine):
    # Begins the seed in the wm tracts
    start = time.time()
    seeds = utils.seeds_from_mask(wm_mask, density=[1, 1, 1], affine=affine)
    end = time.time()
    print('Created White Matter seeds: ' + str((end - start)))
    return seeds

def seeds():
    wm_mask = create_mask.mask()
    files = np.load('files')
    return create_wm_seeds(wm_mask, files['affine'])