# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

""" 
Generate concept data on the cpu or the gpu.
"""

import numpy as np
import argparse
import os, sys

sys.path.insert(1, '../')

from src.datasets.passive_querier import PassiveQuerier
from src.datasets.active_querier import ActiveQuerier

import torch

torch.set_default_tensor_type(torch.FloatTensor)
np.set_printoptions(precision=2)


if __name__ == '__main__':
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--config', type=str, default="/../../configs/rawstate.yaml", help='config file')
    parser.add_argument('--concept', type=str, default='above180', help='concept')
    parser.add_argument('--simulated', action='store_true', default=False, help='cuda')
    parser.add_argument('--active_samples', type=int, default=100, help='samples')
    parser.add_argument('--passive_samples', type=int, default=10, help='samples')
    parser.add_argument('--batch_size', type=int, default=10, help='batch for active learning')
    parser.add_argument('--objective', type=str, default="random", help='type of AL strategy')
    parser.add_argument('--warmstart', type=int, default=0, help='first epochs only random')
    parser.add_argument('--mining', type=int, default=500, help='first epochs mine for positives')
    args = parser.parse_args()

    args.cuda = False
    args.envs = 1
    parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    asset_root = os.path.abspath(parent_dir + "/data/shapenet_objects/")

    # First collect some demonstration data if needed.
    if args.passive_samples > 0:
        args.headless = False
        passive_generator = PassiveQuerier(args, asset_root)
        data_filename, label_filename = passive_generator.collect_data(args.concept, N_queries=args.passive_samples, query_type="demo")
        passive_generator.kill_instance()

    save_dir = parent_dir + "/data/g_shapenet/" + "{}/".format(args.concept)
    data_filename = save_dir+"/demo_gt_data.hdf5"
    label_filename = save_dir+"/demo_gt_label.hdf5"

    if args.active_samples > 0:
        args.headless = True if args.simulated else False

        active_generator = ActiveQuerier(args, asset_root)

        # Collect data.
        active_generator.reset_model()

        if args.passive_samples > 0:
            # Warmstart the model.
            active_generator.retrain(data_filename, label_filename)

        errors = active_generator.collect_data(args.concept, N_queries=args.active_samples, objective=args.objective,\
                                               warmstart=args.warmstart, mining=args.mining)
        print(errors)
        active_generator.kill_instance()