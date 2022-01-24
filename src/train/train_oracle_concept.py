# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import os, sys
import glob
import yaml
import random
import argparse
import numpy as np

sys.path.insert(1, '../')

from src.utils.train_utils import *
from src.utils.input_utils import transform_input

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
	# Parse args.
	parser = argparse.ArgumentParser(description='pass args')
	parser.add_argument('--config', type=str, default="/../../configs/rawstate_oracle.yaml", help='config file')
	parser.add_argument('--concept_dir', type=str, default='above180', help='data directory')
	parser.add_argument('--test', action='store_true', default=False, help='training mode')
	parser.add_argument('--train_amt', type=int, default=None, help='data amount used for training')
	args = parser.parse_args()

	# Load yaml parameters.
	here = os.path.dirname(os.path.abspath(__file__))
	with open(here+args.config, 'r') as stream:
		params = yaml.load(stream)

	# Set random seed if it exists.
	if "seed" in params["data"].keys():
		torch.manual_seed(params["data"]["seed"])
		random.seed(params["data"]["seed"])
		np.random.seed(params["data"]["seed"])

	# Set up data and model parameters.
	device = ("cuda" if torch.cuda.is_available() else "cpu")
	concept = args.concept_dir.split("_")[0]

	dataset_params = {}
	model_params = {}
	if params["data"]["type"] == "rawstate":
		from src.utils.data_utils import RawStateDataset as ObjectDataset
		from src.models.MLP import MLP as Network
		model_params["nb_layers"] = params["train"]["network"]["nb_layers"]
		model_params["nb_units"] = params["train"]["network"]["nb_units"]
		model_params["concept"] = concept
	elif params["data"]["type"] == "pointcloud":
		from src.utils.data_utils import PointDataset as ObjectDataset
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

	training_params = {}
	training_params["num_epochs"] = params["train"]["num_epochs"]
	training_params["learning_rate"] = params["train"]["learning_rate"]
	training_params["batch_size"] = params["train"]["batch_size"]
	training_params["num_workers"] = params["train"]["num_workers"]

	# Get dataset and dataloaders.
	data_path = here + params["data"]["data_dir"]
	label_folder = "label" if len(args.concept_dir.split("_")) == 1 else "label_{}".format(args.concept_dir.split("_")[1])
	train_set = ObjectDataset(data_path+"/data.hdf5", \
							  data_path+concept+"/{}.hdf5".format(label_folder), \
							  data_path+concept+'/train.txt', **dataset_params)
	test_set = ObjectDataset(data_path+"/data.hdf5",\
							 data_path+concept+"/{}.hdf5".format("label"),\
							 data_path+concept+'/test.txt', **dataset_params)

	if args.train_amt is not None:
		indices = random.sample(range(len(train_set)), args.train_amt)
		train_set = Subset(train_set, indices)

	train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=training_params["batch_size"], 
							  num_workers=training_params["num_workers"], pin_memory=True)
	test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=training_params["batch_size"],
							 num_workers=training_params["num_workers"], pin_memory=True)

	if params["data"]["type"] == "rawstate":
		input_dim = transform_input(train_set[0][0].unsqueeze(0), concept).shape[1]
		model_params["input_dim"] = input_dim

	# Define model, optimization, and loss.
	model = Network(**model_params).to(device)

	# Train and evaluate.
	if args.train_amt is not None:
		query_str = "_{}".format(args.train_amt)
	elif len(args.concept_dir.split("_")) > 1:
		query_str = "_g{}".format(args.concept_dir.split("_")[1])
	else:
		query_str = ""
	model_str = "/oracle_{}{}_{}_{}.pt".format(params["data"]["type"], query_str, noise, params["data"]["seed"])
	save_dir = here + params["data"]["save_dir"] + concept
	if not os.path.isdir(save_dir):
		os.mkdir(save_dir)
	model_path = save_dir + model_str

	if args.test:
		model.load_state_dict(torch.load(model_path))
	else:
		train(model, train_loader, test_loader, epochs=training_params["num_epochs"], 
			  lr=training_params["learning_rate"], device=device)
		torch.save(model.state_dict(), model_path)
		print("Saved in ", model_path)   
	test_acc = check_accuracy(model, test_loader, device=device)

	# Save test error.
	results_str = "/oracle_{}{}_{}.txt".format(params["data"]["type"], query_str, params["data"]["seed"])
	results_dir = here + params["data"]["results_dir"] + concept
	if not os.path.isdir(results_dir):
		os.mkdir(results_dir)
	results_path = results_dir + results_str
	with open(results_path, 'w') as f:
		f.write('%.3f' % test_acc)