# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import numpy as np
import torch

from PIL import Image
import io
import os
import sys
import h5py
import errno

sys.path.insert(1, '../')

from src.utils.geom_utils import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Hdf5Cacher:
    r"""**Save and load data from disk using** `h5py` **module.**
    Data will be saved as `.h5` in specified path. If path does not exist,
    it will be created.
    Tips for using HDF5 with multiprocessing (e.g. pytorch)
    https://github.com/pytorch/pytorch/issues/11929
    Attributes
    ----------
    path: str
            Path to the file where samples will be saved and loaded from.
    """

    def __init__(self, path: str, access='a'):
        self.path = path
        try:
            os.makedirs(os.path.dirname(self.path))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self.hdf5 = h5py.File(self.path, access)

    def _hdf5(self):
        return self.hdf5

    def __contains__(self, key: str) -> bool:
        """**Check whether file exists on disk.**
        If file is available it is considered cached, hence you can cache data
        between multiple runs (if you ensure repeatable sampling).
        """
        contains = key in self._hdf5()
        return contains

    def __setitem__(self, key: str, data: dict):
        """**Save** `data` **in specified folder.**
        Name of the item will be equal to `{self.path}/{key}{extension}`.
        """
        if key in self._hdf5().keys():
            del self._hdf5()[key]
        grp = self._hdf5().create_group(key)
        if data is not None:
            for key, val in data.items():
                if val is not None:
                    grp.create_dataset(key, data=val)

    def __getitem__(self, key: str):
        """**Retrieve** `data` **specified by** `key`.
        """
        data = dict()
        grp = self._hdf5()[key]
        for key in grp.keys():
            data[key] = grp[key][()]
        return data

    def clean(self) -> None:
        """**Remove file** `self.path`.
        Behaves just like `os.remove`, but won't act if file does not exist.
        """
        if os.path.isfile(self.path):
            os.remove(self.path)

    def close(self) -> None:
        """
        Close file
        """
        self.hdf5.close()

# Data saving and loading utils.
def get_png(img):
    '''
    Save a numpy array as a PNG, then get it out as a binary blob
    '''
    im = Image.fromarray(img)
    output = io.BytesIO()
    im.save(output, format="PNG")
    return output.getvalue()

def png_to_numpy(png):
    stream = io.BytesIO(png)
    im = Image.open(stream)
    return np.array(im)

def transform_input(x, concept):
    # Ask the person questions about whether the concept cares about the moving/anchor absolute pose.
    single_object_matters = False
    absolute_poses_matter = False
    obj_bbs_matter = False
    if concept in ["above180", "above45", "abovebb"]:
        absolute_poses_matter = True
    if concept in ["upright", "upright45"]:
        single_object_matters = True
    if concept in ["abovebb"]:
        obj_bbs_matter = True
    rel_pose = relative_pose(x[:, :7], x[:, 7:14])
    rel_diff = relative_difference(x[:, :7], x[:, 7:14])

    # Based on the answer to the questions, construct input space.
    if single_object_matters:
        # Concept doesn't have an anchor.
        x_transform = x[:, :7]
    else:
        if absolute_poses_matter:
            # Concept is not relative, rather it is wrt the world frame.
            x_transform = rel_diff
        else:
            # Concept is relative: consider moving wrt to anchor.
            x_transform = rel_pose
    if obj_bbs_matter:
        # Concept cares about the bounding boxes of the objects, not just the poses.
        x_transform = torch.cat((x_transform, x[:,17:20] - x[:,14:17], x[:,23:] - x[:,20:23]), axis=1)
    return x_transform

def make_weights_for_balanced_classes(inputs, nclasses):
    count = [0] * nclasses
    for item in inputs:
        count[int(item[1].item())] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(inputs)
    for idx, val in enumerate(inputs):
        weight[idx] = weight_per_class[int(val[1].item())]
    return weight