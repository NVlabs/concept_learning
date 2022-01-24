# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

""" 
Generate concept data on the cpu or the gpu.
"""

import numpy as np
import random
import argparse
import os, sys

sys.path.insert(1, '../')

from isaacgym import gymtorch
import torch

from src.utils.concept_utils import *
from src.utils.input_utils import Hdf5Cacher, transform_input

from src.models.MLP import MLP

torch.set_default_tensor_type(torch.FloatTensor)
np.set_printoptions(precision=2)


class MultiConceptLabeler(object):
    def __init__(self, args):
        path = args.data_dir + "/data.hdf5"
        self.concept = args.concept
        self.hdf5cacher = Hdf5Cacher(path, "a")
        self.examples = list(self.hdf5cacher._hdf5().keys())
        print("Loading dataset with {} examples".format(len(self.examples)))
  
    def label_data(self):
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        sample = self.hdf5cacher.__getitem__(self.examples[0])
        raw_state = sample["raw_state"].astype(np.float32)
        input_dim = transform_input(torch.tensor(raw_state).unsqueeze(0), self.concept).shape[1]
        train_amts = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

        for train_amt in train_amts:
            model_path = os.path.abspath(parent_dir + "/data/models/{}/".format(args.concept) + "rawstate_classification_{}.pt".format(train_amt))
            model = MLP(2, 64, input_dim, self.concept).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            label_type = "label_{}".format(train_amt)
            for example in self.examples:
                sample = self.hdf5cacher.__getitem__(example)
                raw_state = sample["raw_state"]
                raw_state = torch.tensor(raw_state, device=device, dtype=torch.float32)
                label = torch.sigmoid(model(raw_state)).cpu().detach().numpy()
                sample[label_type] = label
                self.hdf5cacher.__setitem__(example, sample)

if __name__ == '__main__':
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--concept', type=str, default='above180', help='concept')

    args = parser.parse_args()

    parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    args.data_dir = os.path.abspath(parent_dir + "/data/concept_shapenet/")

    generator = MultiConceptLabeler(args)
    generator.label_data()
    generator.hdf5cacher.close()