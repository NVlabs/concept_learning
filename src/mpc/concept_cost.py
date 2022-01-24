# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import torch
import os
import torch.nn as nn

from storm_kit.util_file import get_assets_path, join_path, get_weights_path
from src.models.pointnet import PointNetEncoder

class ConceptCost(nn.Module):
    def __init__(self, weight=None, nn_weight_file='',
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):
        super(ConceptCost, self).__init__()
        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight, **self.tensor_args)
                
        self.load_nn(nn_weight_file)

    def load_nn(self, weight_file_name):
        parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
        checkpoints_dir = os.path.abspath(parent_dir + "/data/models/")
        model_data = torch.load(join_path(checkpoints_dir, weight_file_name))
        pointnet_radius = 0.5
        pointnet_nclusters = 512
        scale = 1
        in_features = 2

        model = PointNetEncoder(pointnet_radius, pointnet_nclusters, scale, in_features)
        model = model.to(**self.tensor_args)
        model.load_state_dict(model_data)
        self.model = model.to(**self.tensor_args)
        self.model.eval()

    def forward(self, points):
        def l2(x, y):
            return (x-y)**2
        outputs = torch.sigmoid(self.model(points)).squeeze()
        labels = torch.full((outputs.shape[0], ), 0.0, requires_grad=True, device=self.tensor_args['device'])
        cost = self.weight * self.model(points).squeeze()
        #return cost
        return self.weight * l2(outputs, labels)

