import numpy as np
from scipy.ndimage.interpolation import shift, zoom
from scipy.ndimage import rotate

def rotation(data,label,rot_angle):
    # rot_angle -> size(1,3) e.g. (15,10,20)
    # Rotate around x-axis
    rot_data = rotate(data, axes=(1,2), angle=rot_angle[0], cval=0.0, reshape=False)
    rot_label = rotate(label, axes=(1,2), angle=rot_angle[0], cval=0.0, reshape=False)

    # Rotate around y-axis
    rot_data = rotate(rot_data, axes=(0,2), angle=rot_angle[1], cval=0.0, reshape=False)
    rot_label = rotate(rot_label, axes=(0,2), angle=rot_angle[1], cval=0.0, reshape=False)
    
    # Rotate around z-axis
    rot_data = rotate(rot_data, axes=(0,1), angle=rot_angle[2], cval=0.0, reshape=False)
    rot_label = rotate(rot_label, axes=(0,1), angle=rot_angle[2], cval=0.0, reshape=False)
    
    return rot_data, rot_label
    
def flip(data,label,axis):
    flip_data = np.flip(data, axis)
    flip_label = np.flip(label, axis)
    
    return flip_data, flip_label

def rot90(data,label, axis):
    # Rotate about a specific axis by 90 degree
    # axis = 0, 1, or 2
    
    if axis == 0: axes = (1,2)
    elif axis == 1: axes = (0,2)
    elif axis == 2: axes = (0,1)
    else: raise ValueError('Accepted values are 0, 1, and 2')
    
    rot_data = rotate(data, axes=axes, angle=90, cval=0.0, reshape=False)
    rot_label = rotate(label, axes=axes, angle=90, cval=0.0, reshape=False)
    
    return rot_data, rot_label

def rot180(data,label, axis):
    # Rotate about a specific axis by 180 degree
    # axis = 0, 1, or 2
    
    if axis == 0: axes = (1,2)
    elif axis == 1: axes = (0,2)
    elif axis == 2: axes = (0,1)
    else: raise ValueError('Accepted values are 0, 1, and 2')
    
    rot_data = rotate(data, axes=axes, angle=180, cval=0.0, reshape=False)
    rot_label = rotate(label, axes=axes, angle=180, cval=0.0, reshape=False)
    
    return rot_data, rot_label