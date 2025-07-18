# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import os, sys
import yaml
import argparse
import numpy as np
import random

sys.path.insert(1, '../')

from src.utils.concept_utils import concept_value
from src.utils.train_utils import *
from src.utils.input_utils import transform_input
from src.utils.geom_utils import *

import torch
from torch import nn
from torch.utils.data import DataLoader


if __name__ == "__main__":
	# Parse args.
	parser = argparse.ArgumentParser(description='pass args')
	parser.add_argument('--config', type=str, default="/../../configs/rawstate.yaml", help='config file')
	parser.add_argument('--concept_dir', type=str, default='above180', help='data directory')
	parser.add_argument('--concept_model', type=str, default=None, help='model file')
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
	device = ("cuda" if torch.cuda.is_available() else "cpu")
	data_path = here + params["data"]["data_dir"]
	model_path = here + params["data"]["save_dir"] + args.concept_dir + "/" + args.concept_model
	concept = args.concept_dir.split("_")[0]

	dataset_params = {}
	model_params = {}
	if params["data"]["type"] == "rawstate":
		from src.utils.data_utils import OptimizationDataset as ObjectDataset
		from src.models.MLP import MLP as Network
		model_params["nb_layers"] = params["train"]["network"]["nb_layers"]
		model_params["nb_units"] = params["train"]["network"]["nb_units"]
		model_params["concept"] = args.concept_dir.split("_")[0]
	elif params["data"]["type"] == "pointcloud":
		from src.utils.data_utils import OptimizationDataset as ObjectDataset
		from src.models.pointnet import PointNetEncoder as Network
		model_params["pointnet_radius"] = params["train"]["network"]["pointnet_radius"]
		model_params["pointnet_nclusters"] = params["train"]["network"]["pointnet_nclusters"]
		model_params["scale"] = params["train"]["network"]["scale"]
		model_params["in_features"] = params["train"]["network"]["in_features"]
	elif params["data"]["type"] == "rgb":
		from src.utils.data_utils import RGBDataset as ObjectDataset
		from src.models.pointnet import PretrainedImageEncoder as Network
		transform = transforms.Compose(
			[
				transforms.Resize((224, 224)),
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
			]
		)
		dataset_params["transform"] = transform
		model_params["input_dim"] = params["train"]["network"]["input_dim"]
		model_params["frozen"] = params["train"]["network"]["frozen"]

	dataset = ObjectDataset(data_path+"/data.hdf5",\
							split_path=data_path+concept+'/test.txt',
							sample=False, **dataset_params)
	if params["data"]["type"] == "rawstate":
		input_dim = transform_input(dataset[0][1].unsqueeze(0), concept).shape[1]
		model_params["input_dim"] = input_dim

	# Define model, optimization, and loss.
	model = Network(**model_params).to(device)
	model.load_state_dict(torch.load(model_path))
	model.eval()

	# Evaluate concept network.
	visualize = True
	results = []
	tests = 1000
	i=0
	while i < tests:
		idx = np.random.choice(range(len(dataset)))
		sample = dataset[idx]
		old_state = torch.cat((sample[0], torch.zeros((sample[0].shape[0],1))), dim=1)
		old_rawstate = sample[1]

		# Skip if either object isn't here.
		old_concept_val = concept_value(old_rawstate.cpu().detach().numpy(), concept)
		if sum(old_state[:, 3]) == 0 or sum(old_state[:, 4]) == 0 or old_concept_val > 0.5:
			continue

		# Recenter around anchor.
		anchor_center = torch.mean(old_state[old_state[:, 4]==1,:3], axis=0)
		old_state[:, :3] -= anchor_center
		old_rawstate[:3] -= anchor_center # center around anchor
		old_rawstate[7:10] -= anchor_center # center around anchor

		# Get new pose.
		T, new_state = evaluate_concept(model, (old_state[:, :5], old_rawstate, 1.0), dtype=params["data"]["type"], \
										opt="CEM", batch_size=100, epochs=10, lr=1e-2, device=device, visualize=visualize)
		if params["data"]["type"] == "pointcloud":
			# Apply transform T to raw state pose.
			moving_center = torch.mean(old_state[old_state[:, 3]==1,:3],axis=0).unsqueeze(0).to(device)
			rawstate = copy.deepcopy(old_rawstate).unsqueeze(0).to(device)
			rawstate[:,:3] -= moving_center
			new_rawstate = transform_rawstate(rawstate, T.unsqueeze(0)).float()
			new_rawstate[:,:3] += moving_center
			new_rawstate = new_rawstate.squeeze()
		else:
			new_rawstate = new_state
		new_rawstate = new_rawstate.cpu().detach().numpy()
		new_state = new_state.cpu().detach().numpy()
		new_concept_val = concept_value(new_rawstate, concept)
		print("old concept: ", old_concept_val)
		print("new concept: ", new_concept_val)

		# Record result.
		results.append(new_concept_val)
		print("Finished test {}.".format(i))
		i += 1
	print("Got {}/{}".format(sum(np.round(results)), tests))

	# Save test error.
	results_str = "/opt_{}.txt".format(args.concept_model[:-3])
	results_dir = here + params["data"]["results_dir"] + args.concept_dir.split("_")[0]
	if not os.path.isdir(results_dir):
		os.mkdir(results_dir)
	results_path = results_dir + results_str
	with open(results_path, 'w') as f:
		f.write('%.3f' % (sum(np.round(results))/tests))