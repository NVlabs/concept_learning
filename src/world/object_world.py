# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

""" 
Generate concept data on the cpu or the gpu.
"""
from isaacgym import gymapi

import numpy as np
import yaml
import argparse
import glob
import os, sys

sys.path.insert(1, '../')

from storm_kit.gym.core import World

from src.utils.camera_utils import *
from src.world.basic_world import BasicWorld
from src.utils.geom_utils import SegLabel
from src.utils.gym_utils import WorldObject

np.set_printoptions(precision=3)


class ObjectWorld(BasicWorld):
    def __init__(self, args, asset_root):
        super(ObjectWorld, self).__init__(args, asset_root)

        # Create gym environments one by one.
        self.worlds = []
        self.storage_pose = gymapi.Transform(gymapi.Vec3(100, 100, 100), gymapi.Quat(0, 0, 0, 1))
        spacing = 5.0
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        for env_idx in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, int(np.sqrt(self.num_envs)))
            self.gym_instance.env_list.append(env_ptr)
            world = World(self.gym, self.sim, env_ptr, self.world_params, w_T_r=gymapi.Transform())
            world.anchors = []
            world.movings = []

            for obj_idx in range(self.num_objs):
                # Anchor.
                obj_handle = self.gym.create_actor(env_ptr, self.obj_assets[obj_idx], self.storage_pose, 
                                                   'anc{}_env{}'.format(obj_idx, env_idx), 2, 2, SegLabel.ANCHOR.value)                
                obj_body_handle = self.gym.get_actor_rigid_body_handle(env_ptr, obj_handle, 0)  
                world.anchors.append(WorldObject(obj_handle, obj_body_handle, obj_idx))
                # Moving.
                obj_handle = self.gym.create_actor(env_ptr, self.obj_assets[obj_idx], self.storage_pose, 
                                                   'mov{}_env{}'.format(obj_idx, env_idx), 2, 2, SegLabel.MOVING.value)                
                obj_body_handle = self.gym.get_actor_rigid_body_handle(env_ptr, obj_handle, 0)  
                world.movings.append(WorldObject(obj_handle, obj_body_handle, obj_idx))
            
            # Spawn camera.
            spawn_camera(world, 60, 640, 480, cuda=self.cuda)

            # Save world and mesh files.
            self.worlds.append(world)

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
    parser.add_argument('--envs', type=int, default=1, help='number of parallel environments')
    args = parser.parse_args()
    
    parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    asset_root = os.path.abspath(parent_dir + "/data/shapenet_objects/")
    object_world = ObjectWorld(args, asset_root)