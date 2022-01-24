# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

""" 
Generate concept data on the cpu or the gpu.
"""
import copy
from isaacgym import gymapi

import numpy as np
import random
import argparse
import os, sys
import h5py

sys.path.insert(1, '../')

from quaternion import from_euler_angles, as_float_array, as_euler_angles, as_quat_array

from src.utils.gym_utils import *
from src.utils.camera_utils import *
from src.utils.concept_utils import *
from src.utils.input_utils import Hdf5Cacher
from src.world.object_world import ObjectWorld

np.set_printoptions(precision=2)


class PassiveQuerier(object):
    def __init__(self, args, asset_root):
        self.object_world = ObjectWorld(args, asset_root)

        color = gymapi.Vec3(0.8, 0.1, 0.1)
        for moving in self.object_world.worlds[0].movings:
            self.object_world.worlds[0].gym.set_rigid_body_color(self.object_world.worlds[0].env_ptr,
                                                                 moving.agent_handle, 0,
                                                                 gymapi.MESH_VISUAL_AND_COLLISION, color)

        self.simulated = args.simulated

    def collect_data(self, concept, N_queries=100, query_type="label"):
        pos_step = 0.05
        rot_step = 0.05

        parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
        save_dir = parent_dir + "/data/g_shapenet/" + "{}/".format(concept)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        labeler_str = "gt" if self.simulated else "human"

        data_filename = save_dir+"/{}_{}_data.hdf5".format(query_type, labeler_str)
        hdf5cacher_data = Hdf5Cacher(data_filename, "w")
        label_filename = save_dir+"/{}_{}_label.hdf5".format(query_type, labeler_str)
        hdf5cacher_label = Hdf5Cacher(label_filename, "w")

        gym = self.object_world.gym
        gym_instance = self.object_world.gym_instance
        world = self.object_world.worlds[0]

        if query_type == "demo":
            # Subscribe to input events.
            gym.subscribe_viewer_keyboard_event(gym_instance.viewer, gymapi.KEY_LEFT, "left")
            gym.subscribe_viewer_keyboard_event(gym_instance.viewer, gymapi.KEY_RIGHT, "right")
            gym.subscribe_viewer_keyboard_event(gym_instance.viewer, gymapi.KEY_UP, "forward")
            gym.subscribe_viewer_keyboard_event(gym_instance.viewer, gymapi.KEY_DOWN, "backward")
            gym.subscribe_viewer_keyboard_event(gym_instance.viewer, gymapi.KEY_U, "up")
            gym.subscribe_viewer_keyboard_event(gym_instance.viewer, gymapi.KEY_J, "down")

            gym.subscribe_viewer_keyboard_event(gym_instance.viewer, gymapi.KEY_A, "roll_cc")
            gym.subscribe_viewer_keyboard_event(gym_instance.viewer, gymapi.KEY_D, "roll_ccw")
            gym.subscribe_viewer_keyboard_event(gym_instance.viewer, gymapi.KEY_W, "pitch_cc")
            gym.subscribe_viewer_keyboard_event(gym_instance.viewer, gymapi.KEY_S, "pitch_ccw")
            gym.subscribe_viewer_keyboard_event(gym_instance.viewer, gymapi.KEY_Q, "yaw_cc")
            gym.subscribe_viewer_keyboard_event(gym_instance.viewer, gymapi.KEY_E, "yaw_ccw")

            gym.subscribe_viewer_keyboard_event(gym_instance.viewer, gymapi.KEY_ENTER, "save")

        for i in range(N_queries):
            try:
                # Pick anchor and moving.
                world.anchor = random.choice(world.anchors)
                world.moving = random.choice(world.movings)

                # Move anchor and moving on the table.
                move_agent_to_random_pose(world, world.anchor.agent_handle)
                move_agent_to_random_pose(world, world.moving.agent_handle)

                gym_instance.step()

                # Query human until keyboard button is pressed.
                if query_type == "demo":
                    saved = False
                    while not saved:
                        new_pose = copy.deepcopy(world.get_pose(world.moving.agent_handle))
                        new_rot = as_euler_angles(as_quat_array((new_pose.r.w, new_pose.r.x, new_pose.r.y, new_pose.r.z)))
                        
                        for evt in gym.query_viewer_action_events(gym_instance.viewer):
                            if evt.action == "left" and evt.value > 0:
                                new_pose.p.x += pos_step
                            if evt.action == "right" and evt.value > 0:
                                new_pose.p.x -= pos_step
                            if evt.action == "forward" and evt.value > 0:
                                new_pose.p.y += pos_step
                            if evt.action == "backward" and evt.value > 0:
                                new_pose.p.y -= pos_step
                            if evt.action == "up" and evt.value > 0:
                                new_pose.p.z += pos_step
                            if evt.action == "down" and evt.value > 0:
                                new_pose.p.z -= pos_step

                            if evt.action == "roll_cc" and evt.value > 0:
                                new_rot[0] += rot_step
                            if evt.action == "roll_ccw" and evt.value > 0:
                                new_rot[0] -= rot_step
                            if evt.action == "pitch_cc" and evt.value > 0:
                                new_rot[1] += rot_step
                            if evt.action == "pitch_ccw" and evt.value > 0:
                                new_rot[1] -= rot_step
                            if evt.action == "yaw_cc" and evt.value > 0:
                                new_rot[2] += rot_step
                            if evt.action == "yaw_ccw" and evt.value > 0:
                                new_rot[2] -= rot_step

                            if evt.action == "save" and evt.value > 0:
                                saved = True

                        if not saved:
                            q = as_float_array(from_euler_angles(new_rot))
                            new_pose.r = gymapi.Quat(q[1], q[2], q[3], q[0])
                            move_agent_to_pose(world, world.moving.agent_handle, new_pose)
                            gym_instance.step()

                raw_state = get_raw_state(self.object_world, world)
                if self.simulated:
                    label = concept_value(raw_state, concept)
                else:
                    # Ask human for label.
                    answer = input("Is this {}? (yes/Y/y)\n".format(concept))
                    label = 1.0 if answer in ["yes", "Y", "y"] else 0.0

                # Save camera data.
                camera_data = observe_camera(world, cuda=self.object_world.cuda)
                data_dict = get_camera_dict(camera_data)
                data_dict["raw_state"] = raw_state
                label_dict = {"label": label}
                uid = "{}_{}_{}".format(world.moving.idx, world.anchor.idx, i)
                hdf5cacher_data.__setitem__(uid, data_dict)
                hdf5cacher_label.__setitem__(uid, label_dict)

                # Move anchor and moving back to original position.
                move_agent_to_pose(world, world.anchor.agent_handle, self.object_world.storage_pose)
                move_agent_to_pose(world, world.moving.agent_handle, self.object_world.storage_pose)

                print("Collected query {}.".format(i))
            except KeyboardInterrupt:
                print('Closing')
                done = True
                break
        hdf5cacher_data.close()
        hdf5cacher_label.close()
        return data_filename, label_filename

    def kill_instance(self):
        self.object_world.gym.destroy_viewer(self.object_world.gym_instance.viewer)


if __name__ == '__main__':
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--concept', type=str, default='above180', help='concept')
    parser.add_argument('--query', type=str, default='label', help='query type')
    parser.add_argument('--simulated', action='store_true', default=False, help='cuda')
    parser.add_argument('--samples', type=int, default=100, help='samples')
    args = parser.parse_args()

    args.headless = False
    args.cuda = False
    args.envs = 1
    parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    asset_root = os.path.abspath(parent_dir + "/data/shapenet_objects/")
    generator = PassiveQuerier(args, asset_root)
    generator.collect_data(args.concept, N_queries=args.samples, query_type=args.query)