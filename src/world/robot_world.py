# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

""" 
Generate concept data on the cpu or the gpu.
"""
from isaacgym import gymapi

import numpy as np
import random
import yaml
import argparse
import glob
import os, sys
import copy

sys.path.insert(1, '../')

from storm_kit.gym.core import World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from quaternion import from_euler_angles, as_float_array

from src.utils.camera_utils import spawn_camera
from src.utils.geom_utils import SegLabel
from src.utils.gym_utils import WorldObject
from src.world.basic_world import BasicWorld

np.set_printoptions(precision=3)


class RobotWorld(BasicWorld):
    def __init__(self, args, asset_root):
        super(RobotWorld, self).__init__(args, asset_root)

        robot_file = args.robot + '.yml'
        robot_yml = join_path(get_gym_configs_path(), robot_file)
        with open(robot_yml) as file:
            robot_params = yaml.safe_load(file)

        # Define robot parameters and create robot simulation.
        sim_params = robot_params['sim_params']
        sim_params['asset_root'] = get_assets_path()
        sim_params['collision_model'] = None
        #sim_params["init_state"] = [0.8, 0.3, 0.0, -1.57, 0.0, 1.86, 0.]
        device = 'cuda' if self.cuda else 'cpu'
        
        # Create gym environment.
        spacing = 5.0
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.env_ptr = self.gym.create_env(self.sim, lower, upper, 1)
        self.gym_instance.env_list.append(self.env_ptr)

        self.robot_sim = RobotSim(gym_instance=self.gym, sim_instance=self.sim, **sim_params, device=device)
        self.robot_ptr = self.robot_sim.spawn_robot(self.env_ptr, sim_params['robot_pose'], coll_id=2)

        # Create world.
        self.w_T_r = copy.deepcopy(self.robot_sim.spawn_robot_pose)
        self.world = World(self.gym, self.sim, self.env_ptr, self.world_params, w_T_r=self.w_T_r)

        # Add objects.
        obj1_idx, obj2_idx = random.sample(range(self.num_objs), 2)
        obj1_idx = 42 #18
        obj2_idx = 6
        obj_handle = self.gym.create_actor(self.env_ptr, self.obj_assets[obj1_idx], gymapi.Transform() * self.w_T_r, 
                                           'moving', 2, 2, SegLabel.MOVING.value)
        obj_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, obj_handle, 0)
        self.gym.set_rigid_body_color(self.env_ptr, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.1, 0.8, 0.1))
        self.world.moving = WorldObject(obj_handle, obj_body_handle, obj1_idx)

        anc_pose = gymapi.Transform(gymapi.Vec3(-0.9, 0.1, -0.0), gymapi.Quat(0, 0, 0, 1)) * self.w_T_r
        obj_handle = self.gym.create_actor(self.env_ptr, self.obj_assets[obj2_idx], anc_pose,
                                           'anchor', 2, 2, SegLabel.ANCHOR.value)
        obj_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, obj_handle, 0)
        self.gym.set_rigid_body_color(self.env_ptr, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.8, 0.1, 0.1))
        self.world.anchor = WorldObject(obj_handle, obj_body_handle, obj2_idx)

        # Spawn camera.
        camera_pose = np.array([0.0,-2.8, 0.3,0.707,0.0,0.0,0.707])
        q = as_float_array(from_euler_angles(-90.0 * 0.01745, 90.0 * 0.01745, 90 * 0.01745))
        camera_pose[3:] = np.array([q[1], q[2], q[3], q[0]])
        spawn_camera(self.world, 60, 640, 480, cuda=self.cuda, camera_pose=camera_pose)

        if self.cuda:
            self.gym.prepare_sim(self.sim)
            self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        else:
            self._root_tensor = None
        print("Worlds initialized.")

if __name__ == '__main__':
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--cuda', action='store_true', default=False, help='cuda')
    args = parser.parse_args()
    args.envs = 1
    
    parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    asset_root = os.path.abspath(parent_dir + "/data/shapenet_objects/")
    object_world = RobotWorld(args, asset_root)