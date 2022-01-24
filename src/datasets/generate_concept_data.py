# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

""" 
Generate data on the cpu or the gpu.
"""

import numpy as np
import random
import copy
import argparse
import os, sys

sys.path.insert(1, '../')

from src.utils.camera_utils import *
from src.utils.gym_utils import *
from src.utils.input_utils import *
from src.world.object_world import ObjectWorld

from src.models.MLP import MLP

np.set_printoptions(precision=2)


class DataGenerator(object):
    def __init__(self, args, asset_root, save_dir):
        self.object_world = ObjectWorld(args, asset_root)
        self.save_dir = save_dir

    def generate_data(self, samples_per_env=1000):
        # Create dataset file to save data to.
        save_filename = self.save_dir+"/data.hdf5"
        hdf5cacher = Hdf5Cacher(save_filename, "w")

        gym = self.object_world.gym
        sim = self.object_world.sim
        gym_instance = self.object_world.gym_instance
        num_actors = gym.get_sim_actor_count(sim)

        for i in range(samples_per_env):
            try:
                # Move all objects randomly.
                anchor_idxes, moving_idxes = np.arange(self.object_world.num_objs), np.arange(self.object_world.num_objs)
                random.shuffle(anchor_idxes)
                random.shuffle(moving_idxes)
                for idx, world in enumerate(self.object_world.worlds):
                    # Pick anchor and moving.
                    world.anchor = world.anchors[anchor_idxes[idx]]
                    world.moving = world.movings[moving_idxes[idx]]
                move_all_active_agents_to_random_pose(self.object_world)
                gym_instance.step()

                if self.object_world.cuda:
                    gym.refresh_actor_root_state_tensor(sim)
                    gym.render_all_camera_sensors(sim)
                    gym.start_access_image_tensors(sim)

                # Save camera data.
                for world in self.object_world.worlds:
                    camera_data = observe_camera(world, cuda=self.object_world.cuda)
                    save_dict = get_camera_dict(camera_data)
                    save_dict["raw_state"] = get_raw_state(self.object_world, world)
                    uid = "{}_{}_{}".format(world.moving.idx, world.anchor.idx, i)
                    hdf5cacher.__setitem__(uid, save_dict)

                # Move anchor and moving back to original position.
                move_all_active_agents_to_pose(self.object_world, self.object_world.storage_pose)
                gym_instance.step()

                if self.object_world.cuda:
                    gym.refresh_actor_root_state_tensor(sim)
                    gym.end_access_image_tensors(sim)
                print("Finished {}".format(i))
            except KeyboardInterrupt:
                print('Closing')
                done = True
                break
        hdf5cacher.close()
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--cuda', action='store_true', default=False, help='cuda')
    parser.add_argument('--samples', type=int, default=1000, help='samples')
    parser.add_argument('--envs', type=int, default=1, help='number of parallel environments')

    args = parser.parse_args()

    parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    asset_root = os.path.abspath(parent_dir + "/data/shapenet_objects/")
    save_dir = os.path.abspath(parent_dir + "/data/concept_shapenet/")

    generator = DataGenerator(args, asset_root, save_dir)
    generator.generate_data(samples_per_env=args.samples)