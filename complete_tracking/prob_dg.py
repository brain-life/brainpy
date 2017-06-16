import time
from dipy.data import default_sphere
from dipy.direction import ProbabilisticDirectionGetter
import create_csd, create_mask
import numpy as np

def prob_direction_getter(csd_model, dmri, wm_mask):
    # Set the Direction Getter to randomly choose directions
    start = time.time()
    csd_fit = csd_model.fit(dmri, mask=wm_mask)
    prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                        max_angle=30.,
                                                        sphere=default_sphere)
    end = time.time()
    print('Created the Direction Getter: ' + str((end - start)))
    return prob_dg

def prob_dg():
    model = create_csd.create_csd()
    wm_mask = create_mask.mask()
    files = np.load('files')
    prob_dg = prob_direction_getter(model, files['dmri'], wm_mask)
    return prob_dg
