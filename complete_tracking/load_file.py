import time
import nibabel as nib
import collections

def load_dmri(data_file):
    # Load the dmri
    dmri_image = nib.load(data_file)
    dmri = dmri_image.get_data()
    return dmri

def load_affine(data_file):
    # Load the affine file
    dmri_image = nib.load(data_file)
    affine = dmri_image.affine
    return affine

def load_aparc(data_fs_seg):
    # Loads the segmentation file
    aparc_im = nib.load(data_fs_seg)
    aparc = aparc_im.get_data()
    return aparc

def load_files():
    start = time.time()
    data_path = '/N/dc2/projects/lifebid/HCP7/108323/'
    data_file = data_path + 'original_hcp_data/Diffusion_7T/' + 'data.nii.gz'

    fs_path = '/N/dc2/projects/lifebid/HCP/Brent/7t_rerun/freesurfer/7t_108323'
    data_fs_seg = fs_path + '/mri/aparc+aseg.nii.gz'

    data_bvec = data_path + 'original_hcp_data/Diffusion_7T/' + 'bvecs'
    data_bval = data_path + 'original_hcp_data/Diffusion_7T/' + 'bvals'

    Data_path = collections.namedtuple("Data_path", ['data_path', 'data_file', 'fs_path', 'data_fs_seg', 'data_bval', 'data_bvec'])
    d = Data_path(data_path=data_path, data_file=data_file, fs_path=fs_path, data_fs_seg=data_fs_seg, data_bval=data_bval, data_bvec=data_bvec)
    # dmri = load_dmri(data_file)
    # affine = load_affine(data_file)
    # aparc = load_aparc(data_fs_seg)
    #
    # Data = collections.namedtuple("Data", ['dmri', 'affine', 'aparc', 'data_bvec', 'data_bval'])
    # d = Data(dmri=dmri, affine=affine, aparc=aparc, data_bvec=data_bvec, data_bval=data_bval)
    end = time.time()
    print('Loaded Data: ' + str(end-start))
    return d

