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


class ConceptLabeler(object):
    def __init__(self, args):
        # Open data hdf5 cacher for read.
        data_path = args.data_dir + "/data.hdf5"
        self.hdf5cacher_read = Hdf5Cacher(data_path, "r")
        self.examples = list(self.hdf5cacher_read._hdf5().keys())

        print("Loading dataset with {} examples".format(len(self.examples)))
        self.concept = args.concept
        self.model = None
        self.label_type = "label"
        if args.concept_model is not None:
            device = ("cuda" if torch.cuda.is_available() else "cpu")
            sample = self.hdf5cacher_read.__getitem__(self.examples[0])
            raw_state = sample["raw_state"].astype(np.float32)
            input_dim = transform_input(torch.tensor(raw_state).unsqueeze(0), self.concept).shape[1]
            self.model = MLP(3, 256, input_dim, self.concept).to(device)
            self.model.load_state_dict(torch.load(args.concept_model))
            self.model.eval()
            model_name = args.concept_model.split("/")[-1]
            model_params = model_name.split("_")
            strat = model_params[0]
            strat_str = "" if strat == "oracle" else strat
            self.label_type = "label_{}{}".format(strat_str, model_params[-2])

        # Open label hdf5 cacher for write.
        label_dir = args.data_dir + "/" + args.concept
        if not os.path.isdir(label_dir):
            os.mkdir(label_dir)
        label_path = label_dir + "/{}.hdf5".format(self.label_type)
        self.hdf5cacher_write = Hdf5Cacher(label_path, "w")

    def label_data(self):
        for example in self.examples:
            sample = self.hdf5cacher_read.__getitem__(example)
            label_dict = {}

            raw_state = sample["raw_state"]

            if self.model is None:
                label = concept_value(raw_state, self.concept)
            else:
                device = ("cuda" if torch.cuda.is_available() else "cpu")
                raw_state = torch.tensor(raw_state, device=device, dtype=torch.float32)
                label = torch.sigmoid(self.model(raw_state)).cpu().detach().numpy()
            label_dict["label"] = label
            self.hdf5cacher_write.__setitem__(example, label_dict)


if __name__ == '__main__':
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--concept', type=str, default='above180', help='concept')
    parser.add_argument('--concept_model', type=str, default=None, help='concept model')

    args = parser.parse_args()

    parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    args.data_dir = os.path.abspath(parent_dir + "/data/concept_shapenet/")
    if args.concept_model is not None:
        args.concept_model = os.path.abspath(parent_dir + "/data/models/{}/".format(args.concept) + args.concept_model)

    generator = ConceptLabeler(args)
    generator.label_data()
    generator.hdf5cacher_read.close()
    generator.hdf5cacher_write.close()