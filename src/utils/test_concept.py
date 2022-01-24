# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

""" 
Generate concept data on the cpu or the gpu.
"""

import numpy as np
import random
import argparse
import os, sys

sys.path.insert(1, '../')

from src.utils.gym_utils import *
from src.world.object_world import ObjectWorld
from src.utils.concept_utils import *

np.set_printoptions(precision=2)


class ConceptTester(object):
    def __init__(self, args, asset_root):
        self.object_world = ObjectWorld(args, asset_root)

    def test_concept(self, concept):
        from storm_kit.util_file import get_assets_path

        gym = self.object_world.gym
        sim = self.object_world.sim
        gym_instance = self.object_world.gym_instance
        world = self.object_world.worlds[0]
        num_actors = gym.get_sim_actor_count(sim)

        obj_asset_file = "urdf/mug/moving.urdf"
        obj_asset_root = get_assets_path()
        moving_color = gymapi.Vec3(0.8, 0.1, 0.1)
        anchor_color = gymapi.Vec3(0.1, 0.8, 0.1)
        # Spawn moving.
        moving_object = world.spawn_object(obj_asset_file, obj_asset_root, gymapi.Transform(), name='moving_object')
        moving_body_handle = gym.get_actor_rigid_body_handle(world.env_ptr, moving_object, 6)
        gym.set_rigid_body_color(world.env_ptr, moving_object, 0, gymapi.MESH_VISUAL_AND_COLLISION, moving_color)
        gym.set_rigid_body_color(world.env_ptr, moving_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, moving_color)
        # Spawn anchor.
        #obj_asset_root = asset_root
        #obj_asset_file = "urdf/Teapot_7c381f85d3b6e8f46a47bc678e9c8907_L.urdf"
        #obj_asset_file = "urdf/Candle_bf7150430f5b5aa12f841233486fac2b_L.urdf"
        anchor_object = world.spawn_object(obj_asset_file, obj_asset_root, gymapi.Transform(), name='anchor_object')
        anchor_body_handle = gym.get_actor_rigid_body_handle(world.env_ptr, anchor_object, 6)
        gym.set_rigid_body_color(world.env_ptr, anchor_object, 0, gymapi.MESH_VISUAL_AND_COLLISION, anchor_color)
        gym.set_rigid_body_color(world.env_ptr, anchor_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, anchor_color)
        world.anchor = world.anchors[0]
        world.anchor.agent_handle = anchor_body_handle
        world.moving = world.movings[0]
        world.moving.agent_handle = moving_body_handle

        while True:
            try:
                raw_state = get_raw_state(self.object_world, world)
                print(concept_value(raw_state, concept))
                gym_instance.step()
            except KeyboardInterrupt:
                print('Closing')
                done = True
                break
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--concept', type=str, default='above180', help='concept')
    args = parser.parse_args()

    args.headless = False
    args.cuda = False
    args.envs = 1

    parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    asset_root = os.path.abspath(parent_dir + "/data/shapenet_objects/")

    tester = ConceptTester(args, asset_root)
    tester.test_concept(args.concept)