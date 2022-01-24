# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.insert(1, '../')

from src.utils.input_utils import transform_input

class MLP(nn.Module):
	"""
	Creates a NN with leaky ReLu non-linearity.
	---
	input nb_layers, nb_units, input_dim
	output scalar
	"""
	def __init__(self, nb_layers, nb_units, input_dim, concept):
		super(MLP, self).__init__()
		self.concept = concept

		layers = []
		dim_list = [input_dim] + [nb_units] * nb_layers + [1]

		for i in range(len(dim_list) - 1):
			layers.append(nn.Linear(dim_list[i], dim_list[i+1]))

		self.fc = nn.ModuleList(layers)

		# initialize weights
		def weights_init(m):
			if isinstance(m, nn.Linear):
				torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
				torch.nn.init.zeros_(m.bias)

		self.apply(weights_init)

		print("Initializing MLP with input dimensionality ", input_dim)

	def forward(self, x):
		x = self.input_torchify(x)
		x = transform_input(x, self.concept)
		for layer in self.fc[:-1]:
			x = F.leaky_relu(layer(x))
		return self.fc[-1](x)

	def input_torchify(self, x):
		"""
			Transforms numpy input to torch tensors.
		"""
		if not torch.is_tensor(x):
			x = torch.Tensor(x)
		if len(x.shape) == 1:
			x = torch.unsqueeze(x, axis=0)
		return x