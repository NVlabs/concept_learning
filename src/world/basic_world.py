# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

""" 
Empty world on the cpu or the gpu.
"""
from isaacgym import gymapi

import numpy as np
import yaml
import argparse
import glob
import os, sys

sys.path.insert(1, '../')

import trimesh

from storm_kit.gym.core import Gym
from storm_kit.util_file import get_gym_configs_path, join_path, load_yaml, get_assets_path

np.set_printoptions(precision=3)


class BasicWorld(object):
    def __init__(self, args, asset_root):
        # Create gym instance.
        self.cuda = args.cuda
        sim_params = load_yaml(join_path(get_gym_configs_path(), 'physx.yml'))
        sim_params['headless'] = args.headless
        sim_params['sim_params']['use_gpu_pipeline'] = True if self.cuda else False

        self.gym_instance = Gym(**sim_params)
        self.gym = self.gym_instance.gym
        self.sim = self.gym_instance.sim

        # Object folders and options.
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.thickness = 0.002
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False

        if "ycb" in asset_root:
            obj_urdf_files = sorted(glob.glob("{}/**/textured.urdf".format(asset_root)))
            obj_mesh_files = sorted(glob.glob("{}/**/textured.obj".format(asset_root)))
        elif "shapenet" in asset_root:
            obj_urdf_files = sorted(glob.glob("{}/urdf/*.urdf".format(asset_root)))
            obj_mesh_files = sorted(glob.glob("{}/meshes/*.obj".format(asset_root)))
        obj_urdf_files = [os.path.relpath(i, asset_root) for i in obj_urdf_files]
        self.obj_assets = [self.gym.load_asset(self.sim, asset_root, urdf, asset_options) for urdf in obj_urdf_files]
        self.obj_meshes= [trimesh.load(obj_mesh_file) for obj_mesh_file in obj_mesh_files]
        self.num_objs = len(obj_urdf_files)
        print("Loaded object assets.")

        # Some world parameters.
        world_file = 'collision_table.yml'
        world_yml = join_path(get_gym_configs_path(), world_file)
        with open(world_yml) as file:
            self.world_params = yaml.load(file, Loader=yaml.FullLoader)
        
        self.num_envs = args.envs


if __name__ == '__main__':
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--cuda', action='store_true', default=False, help='cuda')
    parser.add_argument('--envs', type=int, default=1, help='number of parallel environments')
    args = parser.parse_args()
    
    parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    asset_root = os.path.abspath(parent_dir + "/data/shapenet_objects/")
    object_world = BasicWorld(args, asset_root)