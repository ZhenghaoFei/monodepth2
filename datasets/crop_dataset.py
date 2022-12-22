# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class CropDataset(MonoDataset):
    """Superclass for different types of Crop dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(CropDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.

        self.K = np.array([[0.53398137, 0.        , 0.48660831, 0],
                            [0.        , 0.72386856, 0.33451852, 0],
                            [0.        , 0.        , 1.        ,0 ],
                            [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (416, 128)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        return color

    def check_depth(self):
        return False


class CropRAWDataset(CropDataset):
    """Crop dataset
    """
    def __init__(self, *args, **kwargs):
        super(CropRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        frame_index = min(frame_index, len(self.filenames) - 1)
        frame_index = max(frame_index, 0)
        # print("get_image_path folder %s, frame_index %d, side %s" % (folder, frame_index, side))
        image_path = self.filenames[frame_index]
        # f_str = "{:010d}{}".format(frame_index, self.img_ext)
        # image_path = os.path.join(
        #     self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        # print("image_path: ", image_path)

        return image_path
