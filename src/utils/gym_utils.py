# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import numpy as np
import open3d
import random

from isaacgym import gymapi
from isaacgym import gymtorch

import torch

from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix
from quaternion import from_euler_angles, as_float_array


class WorldObject(object):
    def __init__(self, agent_handle, body_handle, idx):
        self.agent_handle = agent_handle
        self.body_handle = body_handle
        self.idx = idx


def gymT_to_quatT(gymT):
    rotT = torch.eye(4)
    quat = torch.tensor([gymT.r.w, gymT.r.x, gymT.r.y, gymT.r.z]).unsqueeze(0)
    rot = quaternion_to_matrix(quat)
    rotT[0,3] = gymT.p.x
    rotT[1,3] = gymT.p.y
    rotT[2,3] = gymT.p.z
    rotT[:3,:3] = rot[0]
    return rotT

def get_raw_state(object_world, world):
    if object_world._root_tensor is not None:
        root_tensor = gymtorch.wrap_tensor(object_world._root_tensor)
        root_positions = root_tensor[:, 0:3]
        root_orientations = root_tensor[:, 3:7]
        anchor_idx = world.gym.get_actor_index(world.env_ptr, world.anchor.agent_handle, gymapi.DOMAIN_SIM)
        moving_idx = world.gym.get_actor_index(world.env_ptr, world.moving.agent_handle, gymapi.DOMAIN_SIM)

        ap = root_positions[anchor_idx].cpu().numpy()
        ao = root_orientations[anchor_idx].cpu().numpy()
        mp = root_positions[moving_idx].cpu().numpy()
        mo = root_orientations[moving_idx].cpu().numpy()
        poses = np.hstack((mp, mo, ap, ao))
    else:
        ap = world.get_pose(world.anchor.agent_handle)
        mp = world.get_pose(world.moving.agent_handle)
        poses = np.array([mp.p.x, mp.p.y, mp.p.z, mp.r.x, mp.r.y, mp.r.z, mp.r.w,
                         ap.p.x, ap.p.y, ap.p.z, ap.r.x, ap.r.y, ap.r.z, ap.r.w])

    obj1_mesh, obj2_mesh = object_world.obj_meshes[world.moving.idx], object_world.obj_meshes[world.anchor.idx]
    obj1_bb, obj2_bb = obj1_mesh.bounds, obj2_mesh.bounds
    return np.hstack((poses, np.ravel(obj1_bb), np.ravel(obj2_bb)))

def move_agent_to_random_pose(world, agent_handle, _root_tensor=None):
    # Set data generation boundary.
    limits = [(-0.15, 0.15), (-0.15, 0.15), (0.2, 0.5)]

    pos = [random.uniform(l[0], l[1]) for l in limits]
    ang = [random.uniform(-np.pi, np.pi) for l in range(3)]
    q = as_float_array(from_euler_angles(ang[0], ang[1], ang[2]))
    target_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]),
                                   gymapi.Quat(q[1], q[2], q[3], q[0]))
    move_agent_to_pose(world, agent_handle, target_pose, _root_tensor=_root_tensor)

def move_all_active_agents_to_random_pose(object_world):
    actor_idxes = []
    for idx, world in enumerate(object_world.worlds):
        move_agent_to_random_pose(world, world.anchor.agent_handle, object_world._root_tensor)
        move_agent_to_random_pose(world, world.moving.agent_handle, object_world._root_tensor)
        actor_idxes.append(world.gym.get_actor_index(world.env_ptr, world.anchor.agent_handle, gymapi.DOMAIN_SIM))
        actor_idxes.append(world.gym.get_actor_index(world.env_ptr, world.moving.agent_handle, gymapi.DOMAIN_SIM))
    if object_world._root_tensor is not None:
        actor_idxes = torch.tensor(actor_idxes, dtype=torch.int32, device="cuda:0")
        object_world.gym.set_actor_root_state_tensor_indexed(object_world.sim, object_world._root_tensor,
                                                             gymtorch.unwrap_tensor(actor_idxes), len(actor_idxes))

def move_agent_to_pose(world, agent_handle, target_pose, _root_tensor=None):
    if _root_tensor is not None:
        obj_idx = world.gym.get_actor_index(world.env_ptr, agent_handle, gymapi.DOMAIN_SIM)

        root_tensor = gymtorch.wrap_tensor(_root_tensor)
        root_positions = root_tensor[:, 0:3]
        root_orientations = root_tensor[:, 3:7]

        root_positions[obj_idx] = torch.tensor([target_pose.p.x, target_pose.p.y, target_pose.p.z])
        root_orientations[obj_idx] = torch.tensor([target_pose.r.x, target_pose.r.y, target_pose.r.z, target_pose.r.w])
    else:
        world.gym.set_rigid_transform(world.env_ptr, agent_handle, world.robot_pose * target_pose)

def move_all_active_agents_to_pose(object_world, target_pose):
    actor_idxes = []
    for idx, world in enumerate(object_world.worlds):
        move_agent_to_pose(world, world.anchor.agent_handle, target_pose, _root_tensor=object_world._root_tensor)
        move_agent_to_pose(world, world.moving.agent_handle, target_pose, _root_tensor=object_world._root_tensor)
        actor_idxes.append(world.gym.get_actor_index(world.env_ptr, world.anchor.agent_handle, gymapi.DOMAIN_SIM))
        actor_idxes.append(world.gym.get_actor_index(world.env_ptr, world.moving.agent_handle, gymapi.DOMAIN_SIM))
    if object_world._root_tensor is not None:
        actor_idxes = torch.tensor(actor_idxes, dtype=torch.int32, device="cuda:0")
        object_world.gym.set_actor_root_state_tensor_indexed(object_world.sim, object_world._root_tensor,
                                                             gymtorch.unwrap_tensor(actor_idxes), len(actor_idxes))