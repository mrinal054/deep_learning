# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 08:54:11 2022

@author: mrinal

Reference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
"""
Note:
Copy aug2d file from https://github.com/mrinal054/my_utils
Keep aug2d and dataloader in the same directory
"""
import tensorflow
from tensorflow.keras.utils import Sequence
import numpy as np
from PIL import Image
from aug2d import Augmentor2d
import os
import random

class DataGenerator(Sequence):
    def __init__(self,
                 list_IDs,
                 dir_image,
                 dir_mask,
                 batch_size=1,
                 dim=(512, 512),
                 n_channels_image=3,
                 n_channels_mask=1,
                 n_classes=2,
                 shuffle=True,
                 extensions:tuple=(str, str), # Extensions of data and label. e.g. ('.png', '.png')
                 to_categorical=False,
                 do_normalization:tuple = (False, False),
                 do_augmentation = False,
                 aug_list=None,
                 aug_per_call:int=None, # no. of augmentations needed to perform per call. Must no exceed length of aug_list
                 flip_axis=None,
                 rotate_range=None,
                 shift_range:tuple=(None, None),
                 zoom_range = None,
                 shear_range = None,   
                 verbose = False,
                 ):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.dir_image = dir_image
        self.dir_mask = dir_mask
        self.n_channels_image = n_channels_image
        self.n_channels_mask = n_channels_mask
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.extensions = extensions
        self.to_categorical = to_categorical
        self.do_normalization = do_normalization
        self.do_augmentation = do_augmentation
        self.aug_list = aug_list
        self.aug_per_call = aug_per_call
        self.flip_axis = flip_axis
        self.rotate_range = rotate_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range 
        self.shear_range = shear_range
        self.verbose = verbose
        self.on_epoch_end()

    def __len__(self):

        # Counts the number of possible batches that can be made from the total available datasets in list_IDs
        # Rule of thumb, num_datasets % batch_size = 0, so every sample is seen
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):

        # Gets the indexes of batch_size number of data from list_IDs for one epoch
        # If batch_size = 8, 8 files/indexes from list_ID are selected
        # Makes sure that on next epoch, the batch does not come from same indexes as the previous batch
        # The same batch is not seen again until __len()__ - 1 batches are done

        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

            X = np.empty((self.batch_size, *self.dim, self.n_channels_image)) # 3 for color image
            y = np.empty((self.batch_size, *self.dim, self.n_channels_mask)) # 1 for binary/grayscale image

            for i, ID in enumerate(list_IDs_temp):
                
                # Get data and label
                ID_only = os.path.splitext(ID)[0] # remove extension
                xt = np.array(Image.open(os.path.join(self.dir_image, ID_only + self.extensions[0])))                
                yt = np.array(Image.open(os.path.join(self.dir_mask, ID_only + self.extensions[1])))
                
                # If no channel dimension, then add it.
                if xt.ndim == 2: xt = np.expand_dims(xt, axis=-1)
                if yt.ndim == 2: yt = np.expand_dims(yt, axis=-1)
                
                # Do augmentation
                if self.do_augmentation:
                    
                    # Ensure no. of unique augmentations per call does not exceeds the length of aug_list
                    if self.aug_per_call==None: n_aug = 1
                    else: n_aug = self.aug_per_call
                    
                    assert n_aug <= len(self.aug_list), 'aug_per_call cannot be higher than the lenght of aug_list'
                    
                    # Randomly choose which augmentations will be performed in aug_list
                    aug_idx = random.sample(range(len(self.aug_list)), n_aug)
                    
                    # Get aug_list for this call
                    final_aug_list = [self.aug_list[idx] for idx in aug_idx]
                    
                    # Start doing augmentation
                    for aug in final_aug_list:
                        # Rotation
                        if aug == 'rotate':
                            
                            # Create augmentor instances for data and label
                            xt_aug = Augmentor2d(xt)
                            yt_aug = Augmentor2d(yt)
                            
                            if self.rotate_range is not None:
                                rotate_angle = random.randint(-self.rotate_range, self.rotate_range)
                                
                                if self.verbose: print(f'Performing rotation with angle {rotate_angle}')
                                
                                xt = xt_aug.rotate(rotate_angle, mode='constant')
                                yt = yt_aug.rotate(rotate_angle, mode='constant')
                        
                        # Flip
                        elif aug == 'flip':
                            
                            # Create augmentor instances for data and label
                            xt_aug = Augmentor2d(xt)
                            yt_aug = Augmentor2d(yt)
                            
                            if self.verbose: print('Performing flip')
                            if self.flip_axis is not None:
                                if self.flip_axis == 'random':
                                    curr_flip_axis = random.randint(0,1)
                                    if self.verbose: print(f'Randomly chosen flip axis is {curr_flip_axis}')
                                    xt = xt_aug.flip(curr_flip_axis)
                                    yt = yt_aug.flip(curr_flip_axis)
                                else: 
                                    xt = xt_aug.flip(self.flip_axis)
                                    yt = yt_aug.flip(self.flip_axis)
                                    
                        # Shift
                        elif aug == 'shift':
                            
                            # Create augmentor instances for data and label
                            xt_aug = Augmentor2d(xt)
                            yt_aug = Augmentor2d(yt)
                            
                            assert len(self.shift_range) == 2, 'length of shift_range should be 2'
                            
                            # Determine shifts across height and width
                            if self.shift_range[0] is None: curr_shift_h = 0
                            else: curr_shift_h = random.randint(-self.shift_range[0], self.shift_range[0])
                            
                            if self.shift_range[1] is None: curr_shift_w = 0
                            else: curr_shift_w = random.randint(-self.shift_range[1], self.shift_range[1])
                            
                            if self.verbose: print(f'Performing shift with shift amount of ({curr_shift_h},{curr_shift_w})')
                            
                            xt = xt_aug.shift((curr_shift_h, curr_shift_w), mode='constant')
                            yt = yt_aug.shift((curr_shift_h, curr_shift_w), mode='constant')
                            
                            
                        # Zoom
                        elif aug == 'zoom':
                            
                            # Create augmentor instances for data and label
                            xt_aug = Augmentor2d(xt)
                            yt_aug = Augmentor2d(yt)
                            
                            # Assert max zoom-out of 0.9 and min zoom-in of 1.1
                            assert self.zoom_range[0] <= 0.9, 'zoom_range[0] should be <= 0.9'
                            assert self.zoom_range[1] >= 1.1, 'zoom_range[0] should be >= 1.1'
                            
                            # Decide zoom-in or zoom-out. 1 for zoom-in, 0 for zoom-out                            
                            if random.randint(0, 1) == 0: zoom_amount = random.uniform(self.zoom_range[0], 0.9)
                            else: zoom_amount = random.uniform(1.1, self.zoom_range[1])
                            
                            if self.verbose: print(f'Performing zoom with factor {zoom_amount}')
                            
                            xt = xt_aug.zoom(zoom_amount)
                            yt = yt_aug.zoom(zoom_amount)
                                
                        # Shear
                        elif aug == 'shear':
                            
                            # Create augmentor instances for data and label
                            xt_aug = Augmentor2d(xt)
                            yt_aug = Augmentor2d(yt)
                            
                            curr_shear_factor = random.uniform(-self.shear_range, self.shear_range)
                            
                            if self.verbose: print(f'Performing shear with {curr_shear_factor}')
                            
                            xt = xt_aug.shear(curr_shear_factor)
                            yt = yt_aug.shear(curr_shear_factor)
                            
                        # Unaugmented
                        elif aug == None:
                            if self.verbose: print('No augmentation chosen')
                            pass
                        
                        # Skip augmentation if augmentation keyword is invalid
                        else:
                            if self.verbose: print('Invalid augmentation keyword. Skipping augmentation')
                            pass   
                
                # Normalization
                if self.do_normalization[0]: xt = xt/255.0
                if self.do_normalization[1]: yt = yt/255.0
                            
                X[i,] = xt
                y[i,] = yt

            # Convert to categorical
            if self.to_categorical: y = tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes, dtype ="int8") # num_classes may vary 
            
            return X, y
        

# =============================================================================
# # Example
# import matplotlib.pyplot as plt
# 
# img_dir_train = './dataset/images'
# mask_dir_train = './dataset/labels_1ch'
# 
# list_IDs_train = os.listdir(img_dir_train)
# 
# # Parameters
# params = {'dim': (512,512), # (height, width)
#           'batch_size': 1,
#           'n_classes': 2,
#           'n_channels_image': 3,
#           'n_channels_mask': 1,
#           'shuffle': False,
#           'extensions': ('.png', '.png'),
#           'to_categorical': False, # set it to true if you want categorical format, otherwise false.
#           'do_normalization': (True,True),
#           'do_augmentation': True, # to do on-the-fly augmentation, set it to true. 
#           'aug_list': [None, 'rotate', 'flip', 'shift', 'zoom', 'shear'], # add types of augmenation you want. 
#           'aug_per_call': 1, # no. of unique augmentations in each call
#           'rotate_range': 30, # set rotation angle between +rotate_range to -rotate_range
#           'flip_axis': 'random', # set 0, 1 or 'random'
#           'shift_range': (75, 75), # shift between +shift_range to -shift_range
#           'zoom_range': (0.5, 1.8), # Ensure zoom_range[0]<=0.9 and zoom_range[1]>=1.1
#           'shear_range': 0.25, # shear between +shear_range and -shear_range
#           'verbose': True, # set to False for training. Otherwise, it will print status. 
#           }
# 
# train_gen = DataGenerator(list_IDs=list_IDs_train,
#                           dir_image=img_dir_train,
#                           dir_mask=mask_dir_train,
#                           **params)
# 
# iters = iter(train_gen)
# 
# a,b = next(iters)
# 
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(np.squeeze(a)) # uncomment if normalization is True
# # ax[0].imshow(np.squeeze(a.astype(np.uint8))) # uncomment if normalization is False
# ax[1].imshow(np.squeeze(b), cmap='gray')
# =============================================================================
