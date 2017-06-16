import time
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
import json

def create_gradient_table(data_bval, data_bvec):
    # Create the gradient table from the bvals and bvecs
    start = time.time()
    bvals, bvecs = read_bvals_bvecs(data_bval, data_bvec)
    gtab = gradient_table(bvals, bvecs, b0_threshold=100)
    end = time.time()
    print('Created Gradient Table: ' + str((end - start)))
    return gtab

def grad_table():
    with open('config.json') as config_json:
        config = json.load(config_json)
    return create_gradient_table(config['data_bval'], config['data_bvec'])

