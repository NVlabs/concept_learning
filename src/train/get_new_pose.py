# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import torch
import sys, os
import numpy as np

sys.path.insert(1, '../')

from src.utils.train_utils import evaluate_concept
from src.models.pointnet import PointNetEncoder

def get_new_pose(input_moving_object_pointcloud, input_static_object_pointcloud):
	"""
		input_moving_object_pointcloud is N_pts1 x 3
		input_static_object_pointcloud is N_pts2 x 3
	"""

	# Set up data and model parameters.
	device = ("cuda" if torch.cuda.is_available() else "cpu")
	model_path=here+"/../../data/models/above180/pointcloud_classification.pt"

	model_params = {}
	model_params["pointnet_radius"] = 0.5
	model_params["pointnet_nclusters"] = 512
	model_params["scale"] = 1
	model_params["in_features"] = 2

	# Define model, optimization, and loss.
	model = PointNetEncoder(**model_params).to(device)
	model.load_state_dict(torch.load(model_path))
	model.eval()

	# Process input.
	pc = np.vstack((input_moving_object_pointcloud, input_static_object_pointcloud))
	anchor_center = np.mean(input_static_object_pointcloud, axis=0)
	pc -= anchor_center
	seg1 = np.vstack((np.ones((input_moving_object_pointcloud.shape[0],1)), np.zeros((input_static_object_pointcloud.shape[0],1))))
	seg2 = np.vstack((np.zeros((input_moving_object_pointcloud.shape[0],1)), np.ones((input_static_object_pointcloud.shape[0],1))))
	old_state = torch.tensor(np.hstack((pc, seg1, seg2))).float()
	T, _ = evaluate_concept(model, (old_state, [], 1.0), dtype="pointcloud", \
									opt="CEM", batch_size=100, epochs=10, device=device)
	return T # this is xyz, wijk

if __name__ == '__main__':
	# Test the function.
	from src.utils.data_utils import OptimizationDataset
	here = os.path.dirname(os.path.abspath(__file__))
	data_path = here + "/../../data/test_shapenet/"
	concept = "above180"
	dataset = OptimizationDataset(data_path+"/data.hdf5", split_path=data_path+concept+'/test.txt', sample=False)

	idx = np.random.choice(range(len(dataset)))
	sample = dataset[idx]
	state = sample[0]

	moving_pts = state[state[:, 3]==1,:3]
	anchor_pts = state[state[:, 4]==1,:3]

	T = get_new_pose(moving_pts.cpu().detach().numpy(), anchor_pts.cpu().detach().numpy())
	print(T)