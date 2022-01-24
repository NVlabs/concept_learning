# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import torch

from storm_kit.mpc.task.se3_task import SE3Task
from src.mpc.se3_optimization_rollout import SE3OptimizationRollout


class SE3OptimizationTask(SE3Task):
    def __init__(self, task_file='franka.yml', robot_file='ur10_reacher.yml',
                 world_file='collision_env.yml',
                 tensor_args={'device':"cpu", 'dtype':torch.float32},
                 spawn_process=True):

        super().__init__(task_file, robot_file, world_file, tensor_args, spawn_process)

    def get_rollout_fn(self, **kwargs):
        rollout_fn = SE3OptimizationRollout(**kwargs)
        return rollout_fn