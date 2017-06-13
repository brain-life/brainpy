import time
import numpy as np
import load_file

def create_wm_mask(aparc):
    # Create the white matter mask
    start = time.time()
    wm_regions = [2, 41, 16, 17, 28, 60, 51, 53, 12, 52, 12, 52, 13, 18,
                  54, 50, 11, 251, 252, 253, 254, 255, 10, 49, 46, 7]

    wm_mask = np.zeros(aparc.shape)
    for l in wm_regions:
        wm_mask[aparc == l] = 1
    end = time.time()
    print("Created wm mask: " + str((end - start)))
    return wm_mask

def mask():
    d = load_file.load_files()
    return create_wm_mask(load_file.load_aparc(d.data_fs_seg))