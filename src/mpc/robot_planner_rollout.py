# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import torch
import copy
from storm_kit.mpc.rollout.arm_reacher import ArmReacher
from storm_kit.differentiable_robot_model.coordinate_transform import transform_point, CoordinateTransform

from src.mpc.concept_cost import ConceptCost
from src.utils.geom_utils import SegLabel

class RobotPlannerRollout(ArmReacher):
    def __init__(self, exp_params, world_params=None,
                 tensor_args={'device':'cpu','dtype':torch.float32}):
        super().__init__(exp_params=exp_params, world_params=world_params, tensor_args=tensor_args)

        if(exp_params['cost']['concept']['weight'] > 0.0):
            self.concept_cost = ConceptCost(**exp_params['cost']['concept'], tensor_args=self.tensor_args)

    def cost_fn(self, state_dict, action_batch, no_coll=False, horizon_cost=True, return_dist=False):
        cost = super(RobotPlannerRollout, self).cost_fn(state_dict, action_batch, no_coll, horizon_cost, return_dist)

        if(self.exp_params['cost']['concept']['weight'] > 0.0):
            # Get transform from state_batch
            state_batch = state_dict['state_seq']
            ee_pos_batch, ee_rot_batch = state_dict['ee_pos_seq'], state_dict['ee_rot_seq']

            # Separate points based on segmentation.
            pc = copy.deepcopy(self.pc)
            pc_seg = copy.deepcopy(self.pc_seg)
            moving_pc = pc[self.pc_seg==SegLabel.MOVING.value]

            # Get transform.
            pc_pose = CoordinateTransform(trans=self.pc_pose[0:3,3].unsqueeze(0),
                                          rot=self.pc_pose[0:3,0:3].unsqueeze(0), 
                                          tensor_args=self.tensor_args)
            pc_pose_inv = pc_pose.inverse()
            batch_pose = CoordinateTransform(trans=ee_pos_batch, rot=ee_rot_batch, tensor_args=self.tensor_args)
            old_T_new = batch_pose.multiply_transform(pc_pose_inv)

            # Apply transform on the pointcloud to get it in the world state?
            new_moving_pc = torch.stack([transform_point(moving_pc[i], old_T_new._rot, old_T_new._trans) for i in range(moving_pc.shape[0])],dim=2)

            # Concatenate with the non-object points and pass to the network.
            pc = pc.unsqueeze(0).unsqueeze(0).repeat(new_moving_pc.shape[0], new_moving_pc.shape[1], 1, 1)
            pc[:,:,self.pc_seg==SegLabel.MOVING.value,:] = new_moving_pc
            
            # Add one-hot segmentation.
            moving_hot = (pc_seg==SegLabel.MOVING.value).int().unsqueeze(0).unsqueeze(0).repeat(new_moving_pc.shape[0], new_moving_pc.shape[1], 1)
            anchor_hot = (pc_seg==SegLabel.ANCHOR.value).int().unsqueeze(0).unsqueeze(0).repeat(new_moving_pc.shape[0], new_moving_pc.shape[1], 1)
            points = torch.cat((pc, moving_hot.unsqueeze(-1), anchor_hot.unsqueeze(-1)),dim=-1).float()
            points_reshaped = points.reshape((points.shape[0]*points.shape[1],points.shape[2],points.shape[3]))

            with torch.cuda.amp.autocast(enabled=False):
                #c_cost = self.concept_cost.forward(points_reshaped)
                #c_cost = self.concept_cost.forward(points_reshaped[:int(points_reshaped.shape[0]/2)])
                #c_cost = torch.cat((c_cost,self.concept_cost.forward(points_reshaped[int(points_reshaped.shape[0]/2):])))
                c_cost = torch.stack([self.concept_cost.forward(points[:,i,:,:]) for i in range(points.shape[1])], dim=1)
            #c_cost = c_cost.reshape((points.shape[0], points.shape[1]))
            print(torch.sum(c_cost, axis=1))
            cost += c_cost
        if(return_dist):
            return cost, rot_err_norm, goal_dist
        return cost


    def update_params(self, goal_state=None, goal_ee_pos=None, goal_ee_rot=None, goal_ee_quat=None,
                        pc=None, pc_seg=None, pc_pose=None):
        """
        Update params for the cost terms and dynamics model.
        goal_state: n_dofs
        goal_ee_pos: 3
        goal_ee_rot: 3,3
        goal_ee_quat: 4
        pc: N_points, 3
        pc_seg: N_points

        """
        super(RobotPlannerRollout, self).update_params(goal_state=goal_state, goal_ee_pos=goal_ee_pos, 
                                                       goal_ee_rot=goal_ee_rot, goal_ee_quat=goal_ee_quat)
        if (pc is not None):
            self.pc = torch.as_tensor(pc, **self.tensor_args)
        if (pc_seg is not None):
            self.pc_seg = torch.as_tensor(pc_seg, **self.tensor_args)
        if (pc_pose is not None):
            self.pc_pose = torch.as_tensor(pc_pose, **self.tensor_args)
        return True
