# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import torch
from torch import nn
from torchvision import models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
"""
 import torchvision.models
    ...: from torchvision.models.vgg import model_urls
    ...:
    ...: model_urls['vgg16'] = model_urls['vgg16'].replace('https://', 'http://')
    ...: vgg16 = torchvision.models.vgg16(pretrained=True)
"""

class PretrainedImageEncoder(nn.Module):
    """
    Compute resnet features for the image
    maxpool over object mask or bounding box
    """

    def __init__(self, input_dim, frozen=True):
        super(PretrainedImageEncoder, self).__init__()
        self.dim = input_dim
        model_conv = models.resnet18(pretrained=True)
        if frozen:
            for param in model_conv.parameters():
                param.requires_grad = False
        num_features = model_conv.fc.in_features
        last_layer = "fc"
        modules = []
        print("--- LOADING WEIGHTS ---")
        for n, c in model_conv.named_children():
            print(n, last_layer, n == last_layer)
            if n == last_layer:
                print("--- END MODEL HERE ---")
                break
            modules.append(c)
        print("NUM FEATURES =", num_features)
        self.num_features = num_features
        self.extractor = nn.Sequential(*modules)
        self.linear = nn.Linear(num_features, self.dim)
        self.norm = nn.LayerNorm(self.dim)

    def encode(self, inp):
        return self.forward(inp)

    def forward(self, inp):
        x = self.extractor(inp)
        x = x.view(inp.shape[0], self.num_features)
        x = self.linear(x)
        return self.norm(x)
