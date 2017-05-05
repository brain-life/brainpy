# -*- coding: utf-8 -*-
"""
Franco 

This loads a nifti and checks the image orientation.
"""
import dipy as dp
import numpy as np
import nibabel as nib

im = nib.load('data.nii.gz')
nib.aff2axcodes(im.affine)
