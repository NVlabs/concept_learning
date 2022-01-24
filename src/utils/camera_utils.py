# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

""" 
Code adapted from https://gitlab-master.nvidia.com/srl/stochastic-control/-/tree/main

"""
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import numpy as np
import os
import random

from quaternion import *
from src.utils.input_utils import get_png
from src.utils.gym_utils import gymT_to_quatT

def move_camera(world, camera_pose=None):
    """
    Move camera to a random hozirontal view.
    """
    gym = world.gym
    sim = world.sim
    ch = world.camera_handle
    ep = world.env_ptr
    
    if camera_pose is None:
        # Pick random location.
        while True:
            p = np.array([random.uniform(-3.5, 3.5), random.uniform(-3.5, 3.5), 0.3])
            if np.linalg.norm(p) < 3.5 and np.linalg.norm(p) > 2.7:
                break
        
        # Look at target.
        cam_loc = gymapi.Vec3(p[0], p[1], 0.3)
        target_loc = gymapi.Vec3(0, 0, 0)
        gym.set_camera_location(ch, ep, cam_loc, target_loc)

        camera_pose = gym.get_camera_transform(sim, ep, ch)
        e = as_euler_angles(quaternion(*[getattr(camera_pose.r, k) for k in 'wxyz']))
        q = as_float_array(from_euler_angles(e[0], e[1], 90 * 0.01745))
        camera_pose = np.hstack((p, np.array([q[1], q[2], q[3], q[0]])))

    camera_pose = gymapi.Transform(
                                   gymapi.Vec3(camera_pose[0], camera_pose[1], camera_pose[2]),
                                   gymapi.Quat(camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6])
                                  )
    world_camera_pose = world.robot_pose * camera_pose

    gym.set_camera_transform(ch, ep, world_camera_pose)


def spawn_camera(world, fov, width, height, camera_pose=None, cuda=False):
    """
    Spawn a camera in the environment.
    Args:
    fov, width, height: camera params
    robot_camera_pose: Camera pose w.r.t robot_body_handle [x, y, z, qx, qy, qz, qw]
    """
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = fov
    camera_props.height = height
    camera_props.width = width
    camera_props.use_collision_geometry = False
    if cuda:
        camera_props.enable_tensors = True

    camera_handle = world.gym.create_camera_sensor(world.env_ptr, camera_props)
    world.camera_handle = camera_handle
    move_camera(world, camera_pose)

    if cuda:
        world.cam_tensors = get_camera_tensors(world)
    
def get_camera_tensors(world):
    gym = world.gym
    sim = world.sim
    ch = world.camera_handle
    ep = world.env_ptr

    color_image = gym.get_camera_image_gpu_tensor(sim, ep, ch, gymapi.IMAGE_COLOR)
    depth_image = gym.get_camera_image_gpu_tensor(sim, ep, ch, gymapi.IMAGE_DEPTH)
    segmentation = gym.get_camera_image_gpu_tensor(sim, ep, ch, gymapi.IMAGE_SEGMENTATION)
    
    return (gymtorch.wrap_tensor(color_image), gymtorch.wrap_tensor(depth_image), gymtorch.wrap_tensor(segmentation))

def observe_camera(world, cuda=False):
    gym = world.gym
    sim = world.sim
    ch = world.camera_handle
    ep = world.env_ptr

    if not cuda:
        gym.render_all_camera_sensors(sim)

    proj_matrix = gym.get_camera_proj_matrix(sim, ep, ch)
    camera_pose = gym.get_camera_transform(sim, ep, ch)
    view_matrix = gymT_to_quatT(camera_pose).cpu().numpy()
    if cuda:
        color_image = world.cam_tensors[0].cpu().numpy()
        depth_image = world.cam_tensors[1].cpu().numpy()
        segmentation = world.cam_tensors[2].cpu().numpy()
    else:
        color_image = gym.get_camera_image(sim, ep, ch, gymapi.IMAGE_COLOR)
        depth_image = gym.get_camera_image(sim, ep, ch, gymapi.IMAGE_DEPTH)
        segmentation = gym.get_camera_image(sim, ep, ch, gymapi.IMAGE_SEGMENTATION)
                                            
    color_image = np.reshape(color_image, [480, 640, 4])[:, :, :3]
    depth_image[np.abs(depth_image) == np.inf] = 0
    depth_min = depth_image.min()
    depth_max = depth_image.max()
    depth_image = (depth_image - depth_min) * 65535.0 / (depth_max - depth_min)
    
    camera_data = {'color':color_image, 'depth':depth_image, 'mask':segmentation, 'proj_matrix':proj_matrix, 
                   'view_matrix':view_matrix, "depth_min": depth_min, "depth_max":depth_max}
    return camera_data

def get_camera_dict(camera_data):
    camera_dict = {}
    camera_dict["rgb"] = [get_png(camera_data["color"].astype(np.uint8))]
    camera_dict["depth"] = [get_png(camera_data["depth"].astype(np.uint16))]
    camera_dict["mask"] = [get_png(camera_data["mask"].astype(np.uint16))]
    camera_dict["proj_matrix"] = camera_data["proj_matrix"]
    camera_dict["view_matrix"] = camera_data["view_matrix"]
    camera_dict["depth_min"] = camera_data["depth_min"]
    camera_dict["depth_max"] = camera_data["depth_max"]
    return camera_dict