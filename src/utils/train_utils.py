# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import torch
from torch import nn
from tqdm import tqdm
import copy

from src.utils.geom_utils import *


# Train and validate concept network.

def check_accuracy(model, loader, device="cpu"):
	error = 0
	num_samples = 0
	model.eval()

	num_wrong_zeros = 0
	num_wrong_ones = 0
	num_correct_zeros = 0
	num_correct_ones = 0
	num_zeros = 0
	num_nonzeros = 0
	num_correct = 0
	num_samples = 0

	with torch.no_grad():
		for x, y in loader:
			x = x.to(device=device)
			y = y.to(device=device)
			predictions = model(x)
			scores = torch.sigmoid(predictions)
			predictions = torch.tensor([1.0 if i >= 0.5 else 0.0 for i in scores]).to(device)

			# Compute accuracy, precision, recall.
			num_wrong_zeros += predictions[y.squeeze()==0].sum()
			num_wrong_ones += (y[y==1].squeeze() - predictions[y.squeeze()==1]).sum()
			num_correct_zeros += (y==0.0).sum() - predictions[y.squeeze()==0].sum()
			num_correct_ones += predictions[y.squeeze()==1].sum()
			num_zeros += (y==0.0).sum()
			num_nonzeros += (y>0.0).sum()

	error = float(num_correct_zeros + num_correct_ones) / float(num_zeros+num_nonzeros)
	precision = num_correct_ones / (num_correct_ones + num_wrong_zeros)
	recall = num_correct_ones / (num_correct_ones + num_wrong_ones)
	f1 = 2 * (precision * recall) / (precision + recall)
	print(f"Got {num_correct_zeros + num_correct_ones} / {num_zeros+num_nonzeros} \
		with accuracy {error*100:.2f}, precision {precision*100:.2f}, recall {recall*100:.2f}, f1 {f1}.")

	zero_label_error = num_wrong_zeros / num_zeros
	print("Zero Error {} for {} samples.".format(zero_label_error, num_zeros))
	nonzero_label_error = num_wrong_ones / num_nonzeros
	print("Nonzero Error {} for {} samples.".format(nonzero_label_error, num_nonzeros))
	return error

def train(model, train_loader, validation_loader, epochs=10, lr=1e-4, device="cpu"):
	criterion = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	model.train()

	for epoch in range(epochs):
		loop = tqdm(train_loader, total = len(train_loader), leave = True)
		if epoch % 5 == 0:
			loop.set_postfix(val_acc = check_accuracy(model, validation_loader, device))
			model.train()
		epoch_loss = []
		for pts, labels in loop:
			pts = pts.to(device)
			labels = labels.to(device)
			outputs = model(pts)
			loss = criterion(outputs, labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			epoch_loss.append(loss.item())
			loop.set_description(f"Epoch [{epoch}/{epochs}]")
			loop.set_postfix(loss = np.mean(epoch_loss))

# Optimize concept network.
def evaluate_concept(model, sample, dtype="pointcloud", opt="sanity", batch_size=100, epochs=10, lr=1e-2, device="cpu", visualize=False):
	# For CEM if necessary.
	def sample_dx(mu, sigma):
		mu = mu[None].repeat(batch_size, 1)
		sigma = sigma[None].repeat(batch_size, 1)
		eps = torch.randn(batch_size, mu.shape[1]).to(device)
		dx = mu + (eps * sigma)
		dx[:, 3:] = dx[:, 3:] / torch.norm(dx[:, 3:], dim=1)[:, None]
		return dx.to(device)

	def l2(x, y):
		return torch.mean((x-y)**2, dim=-1)

	def l2_pose_dist(pose_old, pose_new):
		return torch.norm(pose_old[:,:3] - pose_new[:,:3], dim=1)

	def anchor_pose_dist(pose):
		dists = torch.norm(pose[:,7:10] - pose[:,:3], dim=1) - 0.3
		return dists * (dists > 0)

	def l2_pc_dist(pc_old, pc_new):
		moving_center = torch.mean(pc_old[:,pc_old[0, :, 3]==1,:3], axis=1)
		moved_center = torch.mean(pc_new[:,pc_new[0, :, 3]==1,:3], axis=1)
		dists = torch.norm(moving_center - anchor_center, dim=1) - 0.3
		return dists * (dists > 0)

	def anchor_pc_dist(pc):
		moving_center = torch.mean(pc[:,pc[0, :, 3]==1,:3],axis=1)
		anchor_center = torch.mean(pc[:,pc[0, :, 4]==1,:3],axis=1)
		dists = torch.norm(moving_center - anchor_center, dim=1) - 0.3
		return dists * (dists > 0)

	# Set optimizer variables.
	requires_grad = True if opt=="gradient" else False

	# Unpack input.
	state = sample[0]
	rawstate = sample[1]
	labels = torch.full((batch_size, 1), sample[2], requires_grad=requires_grad, device=device)

	# Define input points.
	moving_idx = np.where(state[:,3]==1)[0]
	anchor_idx = np.where(state[:,4]==1)[0]
	notmoving_idx = np.where(state[:,3]==0)[0]
	if dtype == "pointcloud":
		# Extract the relevant points.
		idxes = np.hstack((moving_idx, anchor_idx))
		pts = state[idxes].to(device).repeat(batch_size, 1, 1)
	else:
		pts = rawstate.to(device).repeat(batch_size, 1)
	pts.requires_grad = requires_grad

	# Initialize transform.
	if dtype == "pointcloud":
		# Initialize pose around anchor.
		init_anchor_pts = pts[:,pts[0, :, 4]==1,:3].cpu().detach().numpy()
		init_moving_pts = pts[:,pts[0, :, 3]==1,:3].cpu().detach().numpy()

		# Sample to fit in the network.
		samples = 1024
		idxes = np.random.choice(np.arange(pts.shape[1]), size=samples, replace=True)
	else:
		init_moving_pts = pts[:,:3].unsqueeze(1).cpu().detach().numpy()
		init_anchor_pts = pts[:,7:10].unsqueeze(1).cpu().detach().numpy()
	T = torch.tensor(initialize_T_around_pts(init_moving_pts, init_anchor_pts),\
					 requires_grad=requires_grad, device=device, dtype=torch.float32)

	if opt == "sanity":
		epochs = 1
	elif opt == "gradient":
		optimizer = torch.optim.Adam([T], lr=lr)

	if visualize:
		# Save gif pcs.
		pc_obj_gif = [[torch.cat((state, torch.zeros((state.shape[0],1))),dim=1).cpu().numpy()]*batch_size]
		pc_gif = []

	for epoch in range(epochs):
		if dtype == "pointcloud":
			# Transform points (NOT in place) then pass them to the concept network.
			pc_old = copy.deepcopy(pts)
			pc_old_shifted = copy.deepcopy(pc_old)
			moving_center = torch.mean(pc_old[:,pc_old[0, :, 3]==1,:3],axis=1).unsqueeze(1)
			pc_old_shifted[:,pc_old[0, :, 3]==1,:3] -= moving_center
			pc_new = move_points(pc_old_shifted, T)
			pc_new[:,pc_new[0, :, 3]==1,:3] += moving_center
			net_input = pc_new[:,idxes,:]
		else:
			net_input = transform_rawstate(pts, T).float()

			# Use the pointcloud too.
			idxes = np.hstack((moving_idx, anchor_idx))
			pc_old = copy.deepcopy(state[idxes].to(device).repeat(batch_size, 1, 1))
			pc_new = move_points(pc_old, T)
		outputs = model(net_input)
		outputs = torch.sigmoid(outputs)

		if dtype == "pointcloud":
			res1 = l2(outputs, labels)
			res2 = anchor_pc_dist(pc_new)
			res = res1 + res2
		else:
			res1 = l2(outputs, labels)
			res2 = anchor_pose_dist(net_input)
			res = res1 + res2

		if opt == "gradient":
			loss = torch.mean(res)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			#print("LOSS: ", loss.item())
		elif opt == "CEM":
			# Score and pick top 4.
			_, best_idxes = torch.topk(res, 4, largest=False)
			mu = torch.mean(T[best_idxes], dim=0)
			sigma = torch.std(T[best_idxes], dim=0) + (0.01 * torch.ones(T.shape[1], device=device))
			if epoch < epochs-1:
				T = sample_dx(mu, sigma)
			#print(torch.norm(sigma))

		if visualize:
			# Add gif frames.
			movingpc_new = pc_new[:, pc_new[0, :, 3]==1, :]
			pc_obj_frame = []
			for idx in range(batch_size):
				new_pc = torch.cat((movingpc_new[idx], state[notmoving_idx].to(device)), dim=0)
				new_pc = torch.cat((new_pc, torch.zeros((new_pc.shape[0],1)).to(device)), dim=1)
				pc_obj_frame.append(new_pc.cpu().detach().numpy())
			pc_obj_gif.append(pc_obj_frame)

			# Show objects as dots.
			background_idx = np.where((state[:,3]==0)*(state[:,4]==0)==1)[0]
			background_pts = state[background_idx].to(device)
			background_pts = torch.cat((background_pts, torch.zeros((background_pts.shape[0],1)).to(device)), dim=1)
			cloud = torch.mean(pc_old[0,pc_old[0, :, 4]==1,:], axis=0).unsqueeze(0)
			cloud = torch.cat((cloud, torch.zeros((cloud.shape[0],1)).to(device)), dim=1)
			for idx in range(batch_size):
				moved_pts = movingpc_new[idx]
				moving_center = torch.mean(moved_pts, axis=0).unsqueeze(0)
				moving_center[0,3] *= outputs[idx,0]
				if opt == "CEM" and idx in best_idxes:
					moving_center = torch.cat((moving_center, torch.ones((moving_center.shape[0],1)).to(device)), dim=1)
				else:
					moving_center = torch.cat((moving_center, torch.zeros((moving_center.shape[0],1)).to(device)), dim=1)
				cloud = torch.cat((cloud, moving_center), dim=0)
			cloud = torch.cat((cloud, background_pts))
			pc_gif.append(cloud.cpu().detach().numpy())

	# Save results.
	best_idx = torch.argmin(res1)
	T_final = T[best_idx]

	if dtype == "pointcloud":
		best_pts = torch.cat((pc_new[:, pc_new[0, :, 3]==1, :][best_idx], state[notmoving_idx].to(device)), dim=0)
		best_pts = torch.cat((best_pts, torch.zeros((best_pts.shape[0],1)).to(device)),dim=1)
	else:
		best_pts = net_input[best_idx]

	# Visualize if necessary.
	if visualize:
		# View object center batches optimization evolution.
		pc_gif_final = [pc_obj_gif[i][best_idx] for i in range(len(pc_obj_gif))]
		show_pcs_gif(pc_gif, shadow=True)

		# View first and last state.
		viz_old = torch.cat((state, torch.zeros((state.shape[0],1))),dim=1)
		show_pcs(viz_old.cpu().detach().numpy())
		show_pcs(pc_gif_final[-1])

		# View object pc optimization evolution.
		show_pcs_gif(pc_gif_final)

	return T_final, best_pts