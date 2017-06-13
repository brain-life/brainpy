import time
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
import load_file

def create_gradient_table(data_bval, data_bvec):
    # Create the gradient table from the bvals and bvecs
    start = time.time()
    bvals, bvecs = read_bvals_bvecs(data_bval, data_bvec)
    gtab = gradient_table(bvals, bvecs, b0_threshold=100)
    end = time.time()
    print('Created Gradient Table: ' + str((end - start)))
    return gtab

def grad_table():
    d = load_file.load_files()
    return create_gradient_table(d.data_bval, d.data_bvec)