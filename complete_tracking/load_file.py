import time
import nibabel as nib
import json
import numpy as np

def load_dmri(data_file):
    # Load the dmri
    start = time.time()
    dmri_image = nib.load(data_file)
    print("Loaded dmri: " + str(time.time() - start))
    return dmri_image

def load_aparc(data_fs_seg):
    # Loads the segmentation file
    start = time.time()
    aparc_im = nib.load(data_fs_seg)
    print("Loaded aparc: " + str(time.time() - start))
    return aparc_im.get_data()

def load_files():
    start = time.time()
    # data_path = '/N/dc2/projects/lifebid/HCP7/108323/'
    # data_file = data_path + 'original_hcp_data/Diffusion_7T/' + 'data.nii.gz'
    #
    # fs_path = '/N/dc2/projects/lifebid/HCP/Brent/7t_rerun/freesurfer/7t_108323'
    # data_fs_seg = fs_path + '/mri/aparc+aseg.nii.gz'
    #
    # data_bvec = data_path + 'original_hcp_data/Diffusion_7T/' + 'bvecs'
    # data_bval = data_path + 'original_hcp_data/Diffusion_7T/' + 'bvals'

    with open('config.json') as config_json:
        config = json.load(config_json)

    #Data_path = collections.namedtuple("Data_path", ['data_file', 'data_fs_seg', 'data_bval', 'data_bvec'])
    #d = Data_path(data_file=config['data_file'], data_fs_seg=config['data_fs_seg'], data_bval=config['data_bval'],
    #                 data_bvec=config['data_bvec'])

    #Data_path = collections.namedtuple("Data_path", ['data_path', 'data_file', 'fs_path', 'data_fs_seg', 'data_bval', 'data_bvec'])
    #d = Data_path(data_path=data_path, data_file=data_file, fs_path=fs_path, data_fs_seg=data_fs_seg, data_bval=data_bval, data_bvec=data_bvec)

    dmri_image = load_dmri(config['data_file'])
    dmri = dmri_image.get_data()
    affine = dmri_image.affine
    aparc = load_aparc(config['data_fs_seg'])

    np.savez_compressed('files', dmri=dmri, affine=affine, aparc=aparc)

    end = time.time()
    print('Loaded Data: ' + str(end-start))
