# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNetEncoder(nn.Module):
    def __init__(self, pointnet_radius, pointnet_nclusters, scale, in_features):
        super().__init__()

        self._build_model(pointnet_radius, pointnet_nclusters, scale, in_features)

    def _build_model(self, pointnet_radius, pointnet_nclusters, scale, in_features):
        # The number of input features is 3+additional input info where 3
        # represents the x, y, z position of the point-cloud

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=pointnet_nclusters,
                radius=pointnet_radius,
                nsample=64,
                mlp=[in_features, 64 * scale, 64 * scale, 128 * scale]
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128 * scale, 128 * scale, 128 * scale, 256 * scale]
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256 * scale, 256 * scale, 512 * scale, 1024 * scale]
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024 * scale, 512 * scale),
            nn.BatchNorm1d(512 * scale),
            nn.ReLU(True),
            nn.Linear(512 * scale, 256 * scale),
            nn.BatchNorm1d(256 * scale),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        """
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))
        

if __name__ == '__main__':
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    pointnet_radius = 0.02
    pointnet_nclusters = 128
    scale = 1
    in_features = 1
    
    points = torch.rand(2, 1024, 3 + in_features).to(device)
    model = PointNetEncoder(pointnet_radius, pointnet_nclusters, scale, in_features).to(device)
    
    output = model(points)
    print("Output: ", output.shape)
    print(output)
    