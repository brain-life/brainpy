import time
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
import numpy as np
import grad_table

def create_CSD_model(gtab, dmri):
    # Create a CSD model to measure Fiber Orientation Dist, returns CSD model fit
    start = time.time()
    response, ratio = auto_response(gtab, dmri, roi_radius=10, fa_thr=0.7)
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
    end = time.time()
    print ('Created the CSD model: ' + str((end - start)))
    return csd_model

def create_csd():
    files = np.load('files.npz')
    return create_CSD_model(grad_table.grad_table(), files['dmri'])


