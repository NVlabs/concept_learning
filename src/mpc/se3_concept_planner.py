# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

""" Example spawning a robot in gym 

"""
import copy
from isaacgym import gymapi
from isaacgym import gymutil

import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import time
import yaml
import argparse
import numpy as np
import random
import os, glob
from quaternion import quaternion, from_rotation_vector, from_rotation_matrix

from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array

from storm_kit.gym.core import Gym, World
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path

from storm_kit.mpc.control import MPPI

from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform, transform_point

from storm_kit.geom.utils import get_pointcloud_from_depth

from mpc.concept_task import ConceptTask
from src.utils.gym_utils import *
from src.utils.camera_utils import *
from src.utils.geom_utils import *


class SE3ConceptPlanner(object):
    def __init__(self, args, asset_root):
        # Create gym instance.
        sim_params = load_yaml(join_path(get_gym_configs_path(), 'physx.yml'))
        sim_params['headless'] = args.headless
        sim_params['sim_params']['use_gpu_pipeline'] = False
        self.gym_instance = Gym(**sim_params)
        self.gym_instance._create_envs(1)

        self.gym = self.gym_instance.gym
        self.sim = self.gym_instance.sim
        self.env_ptr = self.gym_instance.env_list[0]

        # Some environment files.
        world_file = 'collision_table.yml'
        task_file = 'concept_mbrl.yml'
        robot_file = args.robot + '.yml'

        world_yml = join_path(get_gym_configs_path(), world_file)
        with open(world_yml) as file:
            world_params = yaml.safe_load(file)

        # Create world.
        self.w_T_r = gymapi.Transform()
        self.world_instance = World(self.gym, self.sim, self.env_ptr, world_params, w_T_r=self.w_T_r)

        # Object folders and options. 
        if "ycb" in asset_root:
            obj_urdf_files = sorted(glob.glob("{}/**/textured.urdf".format(asset_root)))
        elif "shapenet" in asset_root:
            obj_urdf_files = sorted(glob.glob("{}/urdf/*.urdf".format(asset_root)))
        obj_urdf_files = [os.path.relpath(i, asset_root) for i in obj_urdf_files]
        obj1_urdf, obj2_urdf = random.sample(obj_urdf_files, 2)

        # Spawn objects and camera in the world instance.
        moving_pose = self.w_T_r * gymapi.Transform(gymapi.Vec3(0, -1.0, 0.5), gymapi.Quat(0, 0, 0, 1))
        obj1_urdf="urdf/CerealBox_a15f43d04b3d5256c9ea91c70932c737_S.urdf"
        obj2_urdf="urdf/Book_59fd296e42d9d65cd889106d819b8d66_L.urdf"
        self.world_instance.moving = self.world_instance.spawn_object(obj1_urdf, asset_root, moving_pose, 
                                                                      seg_label=SegLabel.MOVING.value, name='mov_obj')
        anchor_pose = self.w_T_r * gymapi.Transform(gymapi.Vec3(0, 0.0, 0.1), gymapi.Quat(0, 0, 0, 1))
        self.world_instance.anchor = self.world_instance.spawn_object(obj2_urdf, asset_root, anchor_pose, 
                                                                   seg_label=SegLabel.ANCHOR.value, name='anc_obj')
        spawn_camera(self.world_instance, 60, 640, 480)

        # Create control task.
        self.tensor_args = {'device':torch.device('cuda', 0), 'dtype':torch.float32}
        self.mpc_control = ConceptTask(task_file, robot_file, world_file, self.tensor_args, spawn_process=False)
        print("World initialized.")

    def reach(self, target_pose):
        # Define some initial parameters.
        sim_dt = self.mpc_control.exp_params['control_dt']
        t_step = self.gym_instance.get_sim_time()
        moving_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, self.world_instance.moving, 0)
        moving_pose = self.world_instance.get_pose(moving_body_handle)
        moving_state = self.mpc_control.controller.rollout_fn.get_state(gymT_to_quatT(moving_pose))
        current_robot_state = {'position':moving_state.cpu().numpy(),
                               'velocity':np.zeros(6),
                               'acceleration':np.zeros(6)}

        # Set goal state and pointcloud information.
        camera_data = observe_camera(self.world_instance)
        camera_data = get_pointcloud_from_depth(camera_data)
        
        samples = 1024
        anchor_idx = np.where(camera_data["pc_seg"]==SegLabel.ANCHOR.value)[0]
        moving_idx = np.where(camera_data["pc_seg"]==SegLabel.MOVING.value)[0]
        idxes = np.random.choice(np.hstack((anchor_idx, moving_idx)), size=samples, replace=True)
        np.random.shuffle(idxes)
        camera_data["pc"] -= np.mean(camera_data["pc"][anchor_idx], axis=0)

        self.mpc_control.update_params(goal_ee_pos=target_pose[:3], goal_ee_quat=target_pose[3:], 
                                       pc=camera_data["pc"][idxes], 
                                       pc_seg=np.uint8(camera_data["pc_seg"][idxes]),
                                       pc_pose=gymT_to_quatT(moving_pose))


        w_robot_coord = CoordinateTransform(trans=torch.tensor([0,0,0]).unsqueeze(0),
                                            rot=quaternion_to_matrix(torch.tensor([1,0,0,0])).unsqueeze(0))


        while(True):
            try:
                self.gym_instance.step()
                
                t_step += sim_dt
                command = self.mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)
                
                filtered_state_mpc = current_robot_state
                curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))
                curr_state_tensor = torch.as_tensor(curr_state, **self.tensor_args).unsqueeze(0)
                pose_state = self.mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)

                # get current pose:
                e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
                e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
                ee_pose = gymapi.Transform(copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2])),
                                           gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0]))
                ee_pose = copy.deepcopy(self.w_T_r) * copy.deepcopy(ee_pose)

                self.gym.set_rigid_transform(self.env_ptr, moving_body_handle, copy.deepcopy(ee_pose))

                print("Position: ",ee_pose.p)
                self.gym_instance.clear_lines()
                top_trajs = self.mpc_control.top_trajs.cpu().float()
                n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
                w_pts = w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)

                top_trajs = w_pts.cpu().numpy()
                color = np.array([0.0, 1.0, 0.0])
                for k in range(top_trajs.shape[0]):
                    pts = top_trajs[k,:,:]
                    color[0] = float(k) / float(top_trajs.shape[0])
                    color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
                    self.gym_instance.draw_lines(pts, color=color)

                current_robot_state = command

            except KeyboardInterrupt:
                print('Closing')
                done = True
                break

        self.mpc_control.close()
        return 
            
if __name__ == '__main__':
    
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()
    
    parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/..")
    asset_root = os.path.abspath(parent_dir + "/data/shapenet_objects/")
    planner = SE3ConceptPlanner(args, asset_root)
    planner.reach(np.array([0.0, 1.0, 0.5, 1, 0, 0, 0]))