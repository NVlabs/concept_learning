# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

""" Example spawning a robot in gym 

"""
import copy
from isaacgym import gymapi
from isaacgym import gymutil

import torch

import yaml
import argparse
import numpy as np
import os, glob
from quaternion import quaternion, from_rotation_vector, from_rotation_matrix

from storm_kit.mpc.control import MPPI
from storm_kit.mpc.utils.state_filter import JointStateFilter, RobotStateFilter
from storm_kit.mpc.utils.mpc_process_wrapper import ControlProcess

from storm_kit.differentiable_robot_model.coordinate_transform import CoordinateTransform

import sys
sys.path.insert(1, '../')

from src.utils.gym_utils import *
from src.utils.camera_utils import *
from src.utils.geom_utils import *
from src.world.robot_world import RobotWorld
from storm_kit.mpc.task.reacher_task import ReacherTask
from src.mpc.se3_optimization_task import SE3OptimizationTask


class RobotConceptReacher(object):
    def __init__(self, args, asset_root):
        # Create robot world.
        self.robot_world = RobotWorld(args, asset_root)

        # Create control task.
        task_file = args.robot + '_reacher_srl.yml'
        opt_file = 'optimize_concept.yml'
        robot_file = args.robot + '.yml'
        world_file = 'collision_table.yml'
        self.tensor_args = {'device':torch.device('cuda', 0), 'dtype':torch.float32}
        self.mpc_control = ReacherTask(task_file, robot_file, world_file, self.tensor_args, spawn_process=False)
        self.concept_optimization = SE3OptimizationTask(opt_file, robot_file, world_file, self.tensor_args, spawn_process=False)

    def state_to_EEpose(self, state):
        filtered_state_mpc = state
        curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))
        curr_state_tensor = torch.as_tensor(curr_state, **self.tensor_args).unsqueeze(0)
        pose_state = self.mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)

        # get current pose:
        e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
        e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
        ee_pose = gymapi.Transform(copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2])),
                                   gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0]))
        ee_pose = copy.deepcopy(self.robot_world.w_T_r) * copy.deepcopy(ee_pose)
        return ee_pose

    def compute_goal(self):
        # Define some useful variables.
        sim_dt = self.concept_optimization.exp_params['control_dt']
        t_step = self.robot_world.gym_instance.get_sim_time()
        world = self.robot_world.world
        env_ptr = self.robot_world.env_ptr

        # Move moving where the robot is.
        current_robot_state = copy.deepcopy(self.robot_world.robot_sim.get_state(env_ptr, self.robot_world.robot_ptr))
        self.robot_world.gym.set_rigid_transform(env_ptr, world.moving.body_handle, self.state_to_EEpose(current_robot_state))
        self.robot_world.gym_instance.step()
        moving_pose = world.get_pose(world.moving.body_handle)
        moving_state = self.concept_optimization.controller.rollout_fn.get_state(gymT_to_quatT(moving_pose))
        current_state = {'position':moving_state.cpu().numpy(),
                        'velocity':np.zeros(6),
                        'acceleration':np.zeros(6)}
        print(current_state)

        # Set goal state and pointcloud information.
        goal_state = torch.as_tensor([0,0,0,0,0,0], **self.tensor_args)

        camera_data = observe_camera(world)
        camera_data = get_pointcloud_from_depth(camera_data)

        samples = 1024
        anchor_idx = np.where(camera_data["pc_seg"]==SegLabel.ANCHOR.value)[0]
        moving_idx = np.where(camera_data["pc_seg"]==SegLabel.MOVING.value)[0]
        idxes = np.random.choice(np.hstack((anchor_idx, moving_idx)), size=samples, replace=True)
        np.random.shuffle(idxes)
        camera_data["pc"] -= np.mean(camera_data["pc"][anchor_idx], axis=0)

        self.concept_optimization.update_params(goal_state=goal_state, 
                                                pc=camera_data["pc"][idxes], 
                                                pc_seg=np.uint8(camera_data["pc_seg"][idxes]),
                                                pc_pose=gymT_to_quatT(moving_pose))

        # Optimize concept.
        prev_error = 0.0
        while(True):
            try:
                t_step += sim_dt
                command = self.concept_optimization.get_command(t_step, current_state, control_dt=sim_dt, WAIT=True)
                ee_error = self.concept_optimization.get_current_error(current_state)
                current_state = command
                print(["{:.3f}".format(x) for x in ee_error])
                print(current_state)
                if ee_error[0] - prev_error < 0.00001:
                    break
                prev_error = ee_error[0]
            except KeyboardInterrupt:
                print('Closing')
                done = True
                break
        self.concept_optimization.close()

        # Convert SO3 state into pose.
        curr_state = np.hstack((current_state['position'], current_state['velocity'], current_state['acceleration']))
        curr_state_tensor = torch.as_tensor(curr_state, **self.tensor_args).unsqueeze(0)
        return self.concept_optimization.controller.rollout_fn.get_ee_pose(curr_state_tensor)

    def reach(self, goal_pose):
        # Set goal state.
        self.mpc_control.update_params(goal_ee_pos=goal_pose['ee_pos_seq'], goal_ee_rot=goal_pose['ee_rot_seq'])

        # Define some initial parameters.
        sim_dt = self.mpc_control.exp_params['control_dt']
        gym_instance = self.robot_world.gym_instance
        gym = self.robot_world.gym
        sim = self.robot_world.sim
        robot_sim = self.robot_world.robot_sim
        world = self.robot_world.world
        env_ptr = self.robot_world.env_ptr
        robot_ptr = self.robot_world.robot_ptr
        w_T_r = self.robot_world.w_T_r
        t_step = gym_instance.get_sim_time()
        current_robot_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
        w_robot_coord = CoordinateTransform(trans=gymT_to_quatT(w_T_r)[0:3,3].unsqueeze(0),
                                            rot=gymT_to_quatT(w_T_r)[0:3,0:3].unsqueeze(0))


        while(True):
            try:
                gym_instance.step()

                t_step += sim_dt
                command = self.mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)
                q_des = copy.deepcopy(command['position'])
                qd_des = copy.deepcopy(command['velocity'])

                ee_pose = self.state_to_EEpose(current_robot_state)

                gym.set_rigid_transform(env_ptr, world.moving.body_handle, copy.deepcopy(ee_pose))

                gym_instance.clear_lines()
                top_trajs = self.mpc_control.top_trajs.cpu().float()
                n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
                w_pts = w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)

                top_trajs = w_pts.cpu().numpy()
                color = np.array([0.0, 1.0, 0.0])
                for k in range(top_trajs.shape[0]):
                    pts = top_trajs[k,:,:]
                    color[0] = float(k) / float(top_trajs.shape[0])
                    color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
                    gym_instance.draw_lines(pts, color=color)

                robot_sim.set_robot_state(q_des, qd_des, env_ptr, robot_ptr)
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
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    args = parser.parse_args()
    args.envs = 1
    
    parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    asset_root = os.path.abspath(parent_dir + "/data/shapenet_objects/")

    reacher = RobotConceptReacher(args, asset_root)
    goal_pose = reacher.compute_goal()
    reacher.reach(goal_pose)