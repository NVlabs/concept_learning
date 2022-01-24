# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import os, sys
import yaml
import argparse
import numpy as np
import random

sys.path.insert(1, '../')

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.utils.data_utils import RawStateDataset
from src.utils.geom_utils import show_pcs_with_frame

if __name__ == "__main__":
	# Parse args.
	parser = argparse.ArgumentParser(description='pass args')
	parser.add_argument('--concept', type=str, default='above180', help='data directory')
	parser.add_argument('--strategy', type=str, default='random', help='model file')
	args = parser.parse_args()

	# Load in data folder and labels.
	here = os.path.dirname(os.path.abspath(__file__))
	data_filename = here+"/../../data/g_shapenet/{}/{}gt_data.hdf5".format(args.concept, args.strategy)
	label_filename = here+"/../../data/g_shapenet/{}/{}gt_label.hdf5".format(args.concept, args.strategy)
	split_filename = None

	#data_filename = here+"/../../data/concept_shapenet/data.hdf5"
	#label_filename = here+"/../../data/concept_shapenet/{}/label.hdf5".format(args.concept)
	#split_filename = here+"/../../data/concept_shapenet/{}/train.txt".format(args.concept)

	torch.manual_seed(0)
	random.seed(0)
	np.random.seed(0)

	samples = 500
	train_set = RawStateDataset(data_filename, label_filename, split_filename)
	indices = random.sample(range(len(train_set)), samples)
	#train_set = Subset(train_set, indices)

	# Sort the training set based on uid.
	idxes = []
	for i in range(len(train_set)):
		data, label = train_set[i]
		idx = train_set.examples[i].split("_")[0]
		idxes.append(int(idx))
	idxes = np.argsort(idxes)

	pts = []
	# Shift all points so that they're anchor-centered
	for i in range(samples):
		idx = idxes[i]
		data, label = train_set[idx]
		pt = (data[:3] - data[7:10])
		color = np.zeros((3))
		color[0] = label.item()
		pts.append(np.hstack((pt, color)))
	pts = np.array(pts)

	# Plot the pcs with label as color.
	show_pcs_with_frame(pts, [0,0,0])