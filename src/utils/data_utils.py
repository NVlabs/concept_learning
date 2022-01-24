# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import h5py

from src.utils.geom_utils import *
from src.utils.input_utils import png_to_numpy, Hdf5Cacher


# Dataset utils.
class RGBDataset(Dataset):
    """
    Define a dataset class that reads in data that is organized by image view:
        {
         'color':color_image, 
         'depth':depth_image,
         'segmentation':segmentation, 
         'concept':label
         }
    Params:
        data_dir -- root directory where the data is stored.
    """
    def __init__(self, path, label_type="label", split_path=None, transform=False):
        self.data_path = data_path
        self.label_path = label_path
        self.hdf5cacher_data = None
        self.hdf5cacher_label = None
        self.transform = transform

        if split_path is None:
            with h5py.File(self.data_path, 'r') as file:
                examples = list(file.keys())
        else:
            examples = []
            with open(split_path, "r") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    examples.append(line[:-1])

        self.examples = examples
        print("Loading dataset with {} examples".format(len(self.examples)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        if self.hdf5cacher_data is None:
            self.hdf5cacher_data = Hdf5Cacher(self.data_path, 'r')
            self.hdf5cacher_label = Hdf5Cacher(self.label_path, 'r')
        example = self.examples[index]
        data = self.hdf5cacher_data.__getitem__(example)
        label = self.hdf5cacher_label.__getitem__(example)
        img = png_to_numpy(data["rgb"]).astype(np.uint8)
        label = label["label"]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        y_label = torch.tensor(float(label)).unsqueeze(-1)

        return (img, y_label)


class PointDataset(Dataset):
    """
    Define a dataset class that reads in data that is organized by image view:
        {
         'color':color_image, 
         'depth':depth_image,
         'segmentation':segmentation, 
         'concept':label
         }
    Params:
        data_dir -- root directory where the data is stored.
    """
    def __init__(self, data_path, label_path, split_path=None):
        self.data_path = data_path
        self.label_path = label_path
        self.hdf5cacher_data = None
        self.hdf5cacher_label = None

        if split_path is None:
            with h5py.File(self.data_path, 'r') as file:
                examples = list(file.keys())
        else:
            examples = []
            with open(split_path, "r") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    examples.append(line[:-1])

        self.examples = examples
        print("Loading dataset with {} examples".format(len(self.examples)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        if self.hdf5cacher_data is None:
            self.hdf5cacher_data = Hdf5Cacher(self.data_path, 'r')
            self.hdf5cacher_label = Hdf5Cacher(self.label_path, 'r')
        example = self.examples[index]
        data = self.hdf5cacher_data.__getitem__(example)
        label = self.hdf5cacher_label.__getitem__(example)
        depth = png_to_numpy(data["depth"]).astype(np.float32)
        mask = png_to_numpy(data["mask"]).astype(np.uint16)
        proj_matrix = data["proj_matrix"]
        view_matrix = data["view_matrix"]
        depth_max = data["depth_max"]
        depth_min = data["depth_min"]
        label = label["label"]

        depth = depth * (depth_max - depth_min) / 65535.0 + depth_min
        camera_data = {'depth':depth, 'mask':mask, 'proj_matrix':proj_matrix, 'view_matrix':view_matrix}
        camera_data = get_pointcloud_from_depth(camera_data)

        # Center around the anchor.
        anchor_pts = camera_data["pc"][camera_data["pc_seg"]==SegLabel.ANCHOR.value]
        if anchor_pts.shape[0] > 0:
            camera_data["pc"] -= np.mean(anchor_pts, axis=0)

        # Add one-hot segmentation.
        anchor_hot = (camera_data["pc_seg"]==SegLabel.ANCHOR.value).astype(int).reshape((-1,1))
        moving_hot = (camera_data["pc_seg"]==SegLabel.MOVING.value).astype(int).reshape((-1,1))
        points = np.hstack((camera_data["pc"], moving_hot, anchor_hot))

        # Sample to fit in the network.
        samples = 1024
        anchor_idx = np.where(camera_data["pc_seg"]==SegLabel.ANCHOR.value)[0]
        moving_idx = np.where(camera_data["pc_seg"]==SegLabel.MOVING.value)[0]
        idxes = np.random.choice(np.hstack((anchor_idx, moving_idx)), size=samples, replace=True)
        np.random.shuffle(idxes)
        points = points[idxes]
        points = torch.tensor(points).float()

        y_label = torch.tensor(float(label)).unsqueeze(-1)
        y_label = torch.round(y_label)
        return (points, y_label)


class RawStateDataset(Dataset):
    """
    Define a dataset class that reads in raw state data and concept labels.
    Params:
        data_dir -- root directory where the data is stored.
    """

    def __init__(self, data_path, label_path, split_path=None, balanced=False, noise=0.0):
        self.data_path = data_path
        self.label_path = label_path
        self.hdf5cacher_data = None
        self.hdf5cacher_label = None

        if split_path is None:
            with h5py.File(self.data_path, 'r') as file1:
                with h5py.File(self.label_path, 'r') as file2:
                    examples = list(file1.keys())
                    # Balance the dataset if necessary.
                    if balanced:
                        zero_examples = []
                        one_examples = []
                        for example in examples:
                            data = file1.__getitem__(example)
                            label = file2.__getitem__(example)
                            raw_state = data["raw_state"]
                            label = label["label"]
                            if label == 1.0:
                                one_examples.append(example)
                            else:
                                zero_examples.append(example)
                        if min(len(zero_examples), len(one_examples)) > 0:
                            num_per_class = max(len(zero_examples), len(one_examples))
                            examples = random.choices(zero_examples, k=num_per_class) + random.choices(one_examples, k=num_per_class)
        else:
            examples = []
            with open(split_path, "r") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    examples.append(line[:-1])

        if noise > 0.0:
            epsilon = np.random.uniform(0, 1, size=len(examples))
            self.flip = epsilon < noise

        self.examples = examples
        print("Loading dataset with {} examples".format(len(self.examples)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        if self.hdf5cacher_data is None:
            self.hdf5cacher_data = Hdf5Cacher(self.data_path, 'r')
            self.hdf5cacher_label = Hdf5Cacher(self.label_path, 'r')
        example = self.examples[index]
        data = self.hdf5cacher_data.__getitem__(example)
        label = self.hdf5cacher_label.__getitem__(example)
        raw_state = data["raw_state"]
        label = label["label"]

        if hasattr(self, 'flip'):
            if self.flip[index]:
                label = 1.0 - label

        raw_state = torch.tensor(raw_state).float()
        y_label = torch.tensor(float(label)).unsqueeze(-1)
        y_label = torch.round(y_label)

        return (raw_state, y_label)

class OptimizationDataset(Dataset):
    """
    Define a dataset class that reads in data that is organized by image view:
        {
         'color':color_image, 
         'depth':depth_image,
         'segmentation':segmentation, 
         'concept':label
         }
    Params:
        data_dir -- root directory where the data is stored.
    """
    def __init__(self, data_path, split_path=None, sample=True):
        self.data_path = data_path
        self.hdf5cacher_data = None
        self.sample = sample

        if split_path is None:
            with h5py.File(self.data_path, 'r') as file:
                examples = list(file.keys())
        else:
            examples = []
            with open(split_path, "r") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    examples.append(line[:-1])

        self.examples = examples
        print("Loading dataset with {} examples".format(len(self.examples)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        if self.hdf5cacher_data is None:
            self.hdf5cacher_data = Hdf5Cacher(self.data_path, 'r')
        example = self.examples[index]
        data = self.hdf5cacher_data.__getitem__(example)
        depth = png_to_numpy(data["depth"]).astype(np.float32)
        mask = png_to_numpy(data["mask"]).astype(np.uint16)
        proj_matrix = data["proj_matrix"]
        view_matrix = data["view_matrix"]
        depth_max = data["depth_max"]
        depth_min = data["depth_min"]
        raw_state = data["raw_state"]

        depth = depth * (depth_max - depth_min) / 65535.0 + depth_min
        camera_data = {'depth':depth, 'mask':mask, 'proj_matrix':proj_matrix, 'view_matrix':view_matrix}
        camera_data = get_pointcloud_from_depth(camera_data)

        # Add one-hot segmentation.
        anchor_hot = (camera_data["pc_seg"]==SegLabel.ANCHOR.value).astype(int).reshape((-1,1))
        moving_hot = (camera_data["pc_seg"]==SegLabel.MOVING.value).astype(int).reshape((-1,1))
        points = np.hstack((camera_data["pc"], moving_hot, anchor_hot))

        if self.sample:
            # Sample to fit in the network.
            samples = 1024
            anchor_idx = np.where(camera_data["pc_seg"]==SegLabel.ANCHOR.value)[0]
            moving_idx = np.where(camera_data["pc_seg"]==SegLabel.MOVING.value)[0]
            idxes = np.random.choice(np.hstack((anchor_idx, moving_idx)), size=samples, replace=True)
            np.random.shuffle(idxes)
            points = points[idxes]
        points = torch.tensor(points).float()
        raw_state = torch.tensor(raw_state).float()

        return (points, raw_state)