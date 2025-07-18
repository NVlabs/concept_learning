# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import os, sys
import yaml
import argparse
import numpy as np
import random

sys.path.insert(1, '../')

from src.utils.geom_utils import *
from src.utils.data_utils import OptimizationDataset as ObjectDataset

import torch
from torch import nn
from torch.utils.data import DataLoader


def sanity_check(old_state, old_rawstate):
	# Define input points.
	moving_idx = np.where(old_state[:,3]==1)[0]
	anchor_idx = np.where(old_state[:,4]==1)[0]
	notmoving_idx = np.where(old_state[:,3]==0)[0]

	# Extract the relevant points.
	idxes = np.hstack((moving_idx, anchor_idx))
	pc_old = old_state[idxes].unsqueeze(0)

	# Initialize transform around anchor.
	init_anchor_pts = pc_old[0, pc_old[0, :, 4]==1,:3].unsqueeze(0).detach().numpy()
	init_moving_pts = pc_old[0, pc_old[0, :, 3]==1,:3].unsqueeze(0).detach().numpy()
	#T = torch.tensor(initialize_T_around_pts(init_moving_pts, init_anchor_pts), dtype=torch.float32)

	xyz = np.random.randn(1,3) * 0.1
	quat = np.random.randn(1, 4)
	quat = quat / np.linalg.norm(quat, axis=1)[:, None]
	T = torch.tensor(np.hstack((xyz, quat)), dtype=torch.float32)

	# Move points.
	pc_old_shifted = copy.deepcopy(pc_old)
	moving_center = torch.mean(pc_old[:,pc_old[0, :, 3]==1,:3],axis=1).unsqueeze(1)
	pc_old_shifted[:,pc_old[0, :, 3]==1,:3] -= moving_center
	pc_new = move_points(pc_old_shifted, T)
	pc_new[:,pc_new[0, :, 3]==1,:3] += moving_center
	movingpc_new = pc_new[:, pc_new[0, :, 3]==1, :]
	new_state = torch.cat((movingpc_new[0], old_state[notmoving_idx]), dim=0)

	# Move rawstate.
	moving_center = torch.mean(pc_old[0,pc_old[0, :, 3]==1,:3],axis=0)
	rawstate = copy.deepcopy(old_rawstate).unsqueeze(0)
	rawstate[:,:3] -= moving_center
	new_rawstate = transform_rawstate(rawstate, T).float()
	new_rawstate[:,:3] += moving_center
	new_rawstate = new_rawstate.squeeze().detach().numpy()

	# Show old state and new state.
	show_pcs_with_frame(old_state.cpu().detach().numpy(), [0,0,0]) # center at anchor in pre-moving state
	show_pcs_with_frame(old_state.cpu().detach().numpy(), old_rawstate[:3]) # center at moving object
	show_pcs_with_frame(new_state.cpu().detach().numpy(), [0,0,0]) # center at anchor in post-moving state
	show_pcs_with_frame(new_state.cpu().detach().numpy(), new_rawstate[:3]) # center at moved object

if __name__ == "__main__":
	# Parse args.
	parser = argparse.ArgumentParser(description='pass args')
	parser.add_argument('--config', type=str, default="/../../configs/rawstate.yaml", help='config file')
	parser.add_argument('--concept', type=str, default='above180', help='data directory')
	args = parser.parse_args()

	# Load yaml parameters.
	here = os.path.dirname(os.path.abspath(__file__))
	with open(here+args.config, 'r') as stream:
		params = yaml.safe_load(stream)

	# Set random seed if it exists.
	if "seed" in params["data"].keys():
		torch.manual_seed(params["data"]["seed"])
		random.seed(params["data"]["seed"])
		np.random.seed(params["data"]["seed"])

	# Set up data and model parameters.
	data_path = here + params["data"]["data_dir"]
	concept = args.concept

	dataset = ObjectDataset(data_path+"/data.hdf5",\
							split_path=data_path+concept+'/test.txt',
							sample=False)

	# Sanity check transform.
	tests = 1000
	i=0
	while i < tests:
		idx = np.random.choice(range(len(dataset)))
		sample = dataset[idx]
		old_state = torch.cat((sample[0], torch.zeros((sample[0].shape[0],1))), dim=1)
		old_rawstate = sample[1]

		# Recenter around anchor.
		anchor_center = torch.mean(old_state[old_state[:, 4]==1,:3], axis=0)
		old_state[:, :3] -= anchor_center
		old_rawstate[:3] -= anchor_center # center around anchor
		old_rawstate[7:10] -= anchor_center # center around anchor

		sanity_check(old_state, old_rawstate)