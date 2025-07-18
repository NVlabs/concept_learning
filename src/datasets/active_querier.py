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
import yaml

sys.path.insert(1, '../')

from src.utils.camera_utils import *
from src.utils.gym_utils import *
from src.utils.geom_utils import *
from src.utils.train_utils import *
from src.utils.input_utils import transform_input, Hdf5Cacher, make_weights_for_balanced_classes
from src.utils.concept_utils import concept_value
from src.world.object_world import ObjectWorld

from src.models.MLP import MLP
from src.utils.data_utils import RawStateDataset

import pytorch3d.transforms as tra3d
import torch
from torch.utils.data import DataLoader

torch.set_default_tensor_type(torch.FloatTensor)
np.set_printoptions(precision=2)


class ActiveQuerier(object):
    def __init__(self, args, asset_root):
        self.object_world = ObjectWorld(args, asset_root)

        color = gymapi.Vec3(0.8, 0.1, 0.1)
        for moving in self.object_world.worlds[0].movings:
            self.object_world.worlds[0].gym.set_rigid_body_color(self.object_world.worlds[0].env_ptr,
                                                                 moving.agent_handle, 0,
                                                                 gymapi.MESH_VISUAL_AND_COLLISION, color)

        self.simulated = args.simulated
        self.batch_size = args.batch_size

        # Start empty model.
        here = os.path.dirname(os.path.abspath(__file__))
        with open(here+args.config, 'r') as stream:
            params = yaml.safe_load(stream)

        # Set up model parameters.
        self.seed = params["data"]["seed"]
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = here + params["data"]["data_dir"]
        self.model_path = here + params["data"]["save_dir"]+args.concept
        self.results_path = here + params["data"]["results_dir"]+args.concept
        self.concept = args.concept

        model_params = {}
        model_params["nb_layers"] = params["train"]["network"]["nb_layers"]
        model_params["nb_units"] = params["train"]["network"]["nb_units"]
        world = self.object_world.worlds[0]
        world.anchor = random.choice(world.anchors)
        world.moving = random.choice(world.movings)
        x = torch.tensor(get_raw_state(self.object_world, world)).unsqueeze(0)
        model_params["input_dim"] = transform_input(x, args.concept).shape[1]
        model_params["concept"] = args.concept

        # Setup training parameters.
        training_params = {}
        training_params["num_epochs"] = params["train"]["num_epochs"]
        training_params["learning_rate"] = params["train"]["learning_rate"]
        training_params["batch_size"] = params["train"]["batch_size"]
        training_params["num_workers"] = params["train"]["num_workers"]
        self.training_params = training_params
        self.model_params = model_params

    def reset_model(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.model = MLP(**self.model_params).to(self.device)

    def retrain(self, data_filename, label_filename, model_filename):
        # Make a dataset out of the passive queries.
        train_set = RawStateDataset(data_filename, label_filename, balanced=True)
        test_set = RawStateDataset(self.data_path+"/data.hdf5", self.data_path+self.concept+"/label.hdf5", self.data_path+self.concept+'/test.txt')                                                                  
        train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.training_params["batch_size"],
                                  num_workers=self.training_params["num_workers"], pin_memory=True)
        test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=self.training_params["batch_size"], 
                                  num_workers=self.training_params["num_workers"], pin_memory=True)

        # Train
        train(self.model, train_loader, train_loader, epochs=self.training_params["num_epochs"], 
              lr=self.training_params["learning_rate"], device=self.device)
        torch.save(self.model.state_dict(), model_filename)
        print("Saved in ", model_filename)  
        error = check_accuracy(self.model, test_loader, device=self.device)
        return error

    def collect_data(self, concept, N_queries=100, objective="max", warmstart=0, mining=100):
        # Create dataset file to save data to.
        parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
        save_dir = parent_dir + "/data/g_shapenet/" + "{}/".format(concept)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        labeler_str = "gt" if self.simulated else "human"
        mine_str = "mine" if mining > 0 else ""
        data_filename = save_dir+"/{}{}{}_data.hdf5".format(objective, labeler_str, mine_str)
        label_filename = save_dir+"/{}{}{}_label.hdf5".format(objective, labeler_str, mine_str)
        print("Collecting data in ", data_filename)

        gym = self.object_world.gym
        gym_instance = self.object_world.gym_instance
        world = self.object_world.worlds[0]

        if mining > 0:
            # Ask the person questions about whether the concept cares about the moving/anchor absolute pose.
            single_object_matters = False
            absolute_poses_matter = False
            if concept in ["above180", "above45", "abovebb"]:
                absolute_poses_matter = True
            if concept in ["upright", "upright45"]:
                single_object_matters = True

        test_errors = []

        num_zeros = 0
        num_ones = 0
        positives = []
        gt_labels = []

        for i in range(N_queries):
            try:
                # Pick anchor and moving.
                world.anchor = random.choice(world.anchors)
                world.moving = random.choice(world.movings)

                # Move anchor and moving on the table.
                move_agent_to_random_pose(world, world.anchor.agent_handle)
                move_agent_to_random_pose(world, world.moving.agent_handle)
                gym_instance.step()

                al = np.random.choice([0, 1]) if objective in ["confrand", "allrand"] else int(objective != "random")
                gt_label = "random"
                if objective != "random" and al == 1 and i >= warmstart:
                    # Get current state and pointcloud.
                    state = torch.tensor(get_raw_state(self.object_world, world)).float()
                    camera_data = observe_camera(world, cuda=self.object_world.cuda)
                    depth = camera_data["depth"] * (camera_data["depth_max"] - camera_data["depth_min"]) / 65535.0 + camera_data["depth_min"]
                    camera_data["depth"] = depth
                    camera_data = get_pointcloud_from_depth(camera_data)
                    anchor_hot = (camera_data["pc_seg"]==SegLabel.ANCHOR.value).astype(int).reshape((-1,1))
                    moving_hot = (camera_data["pc_seg"]==SegLabel.MOVING.value).astype(int).reshape((-1,1))
                    points = torch.tensor(np.hstack((camera_data["pc"], moving_hot, anchor_hot))).float()

                    # Evaluate concept network using objective.
                    self.model.eval()

                    if objective == "max":
                        gt_label = 1.0
                    elif objective == "min":
                        gt_label = 0.0
                    elif objective in ["all", "allrand"]:
                        gt_label = np.random.choice([0.0, 0.5, 1.0])
                    elif objective == "allscheduled":
                        if num_zeros > num_ones + 5:
                            gt_label = 1.0
                        elif num_ones > num_zeros + 5:
                            gt_label = 0.0
                        else:
                            gt_label = 0.5
                    elif objective in ["confusion", "confrand"]:
                        gt_label = 0.5

                    # Generate a query.
                    T, new_state = evaluate_concept(self.model, (points, state, gt_label), dtype="rawstate", \
                                                    opt="CEM", batch_size=100, epochs=10, device=self.device)

                    # Move moving to new pose.
                    moving_pose = gymapi.Transform(gymapi.Vec3(new_state[0], new_state[1], new_state[2]),
                                                   gymapi.Quat(new_state[3], new_state[4], new_state[5], new_state[6]))
                    move_agent_to_pose(world, world.moving.agent_handle, moving_pose)
                    gym_instance.step()
                gt_labels.append(gt_label)

                # Mine some more positives.
                if (i<mining+warmstart) and (i>=warmstart) and (len(positives)>0):
                    if random.uniform(0, 1) > 0.5:
                        init_state = random.choice(positives)
                        curr_state = get_raw_state(self.object_world, world)
                        new_state = copy.deepcopy(curr_state)
                        p_sigma = 0.01
                        r_sigma = 0.3

                        # Only moving matters, so preserve the moving absolute pose.
                        if single_object_matters:
                            new_state[:7] = copy.deepcopy(init_state[:7])
                        # Both objects matter, we just need to know if in a relative of absolute way.
                        else:
                            # Preserve position and rotation differential.
                            if absolute_poses_matter:
                                rel_diff = relative_difference(torch.tensor(init_state[:7]).unsqueeze(0), \
                                                               torch.tensor(init_state[7:14]).unsqueeze(0))
                                moving_delta = rel_diff[0,:3].detach().numpy()
                                q_delta = rel_diff[0,3:]
                                # Change relative position and rotation.
                                new_state[:3] = new_state[7:10] + moving_delta
                                R_delta = tra3d.quaternion_to_matrix(q_delta)
                                q = [new_state[13], new_state[10], new_state[11], new_state[12]]
                                R = tra3d.quaternion_to_matrix(torch.tensor(q))
                                R_new = torch.matmul(torch.inverse(R.T), R_delta)
                                q_new = tra3d.matrix_to_quaternion(R_new)
                                new_state[3:7] = [q_new[1], q_new[2], q_new[3], q_new[0]]
                            # Preserve relative position and rotation.
                            else:
                                # Change relative pose.
                                rel_pose = relative_pose(torch.tensor(init_state[:7]).unsqueeze(0), \
                                                         torch.tensor(init_state[7:14]).unsqueeze(0))
                                new_state[:7] = compose_qq(torch.tensor(new_state[7:14]).unsqueeze(0), rel_pose)[0]
                        add_noise_to_pose(new_state[:7], p_sigma=p_sigma, r_sigma=r_sigma)
                        #print(concept_value(init_state, concept), concept_value(new_state, concept))

                        # Move moving to new pose.
                        moving_pose = gymapi.Transform(gymapi.Vec3(new_state[0], new_state[1], new_state[2]),
                                                       gymapi.Quat(new_state[3], new_state[4], new_state[5], new_state[6]))
                        move_agent_to_pose(world, world.moving.agent_handle, moving_pose)
                        anchor_pose = gymapi.Transform(gymapi.Vec3(new_state[7], new_state[8], new_state[9]),
                                                       gymapi.Quat(new_state[10], new_state[11], new_state[12], new_state[13]))
                        move_agent_to_pose(world, world.anchor.agent_handle, anchor_pose)
                        gym_instance.step()
                        gt_labels[-1] = "mine"

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
                uid = "{}_{}_{}".format(i, world.moving.idx, world.anchor.idx)

                hdf5cacher_data = Hdf5Cacher(data_filename, "a")
                hdf5cacher_label = Hdf5Cacher(label_filename, "a")
                hdf5cacher_data.__setitem__(uid, data_dict)
                hdf5cacher_label.__setitem__(uid, label_dict)
                hdf5cacher_data.close()
                hdf5cacher_label.close()

                # Move anchor and moving back to original position.
                move_agent_to_pose(world, world.anchor.agent_handle, self.object_world.storage_pose)
                move_agent_to_pose(world, world.moving.agent_handle, self.object_world.storage_pose)

                print("Collected query {}.".format(i))

                # Retrain model if needed.
                train_now = False
                if i == (warmstart - 1):
                    train_now = True
                elif (i >= warmstart) and ((i - warmstart) % self.batch_size == self.batch_size-1):
                    train_now = True
                if train_now:
                    model_filename = self.model_path + "/{}{}{}_rawstate_{}_{}.pt".format(objective, labeler_str, mine_str, i+1, self.seed)
                    error = self.retrain(data_filename, label_filename, model_filename)
                    test_errors.append(error)

                # Keep track of true positives.
                if label > 0.5:
                    num_ones += 1
                    positives.append(raw_state)
                else:
                    num_zeros += 1

            except KeyboardInterrupt:
                print('Closing')
                done = True
                break
        print(gt_labels)
        # Save test error.
        results_str = "/{}{}{}_rawstate_{}.txt".format(objective, labeler_str, mine_str, self.seed)
        results_path = self.results_path + results_str
        with open(results_path, 'w') as f:
            for element in test_errors:
                f.write('%.5f' % element + " ")
        return test_errors

    def kill_instance(self):
        self.object_world.gym.destroy_viewer(self.object_world.gym_instance.viewer)


if __name__ == '__main__':
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--config', type=str, default="/../../configs/rawstate.yaml", help='config file')
    parser.add_argument('--concept', type=str, default='above180', help='concept')
    parser.add_argument('--simulated', action='store_true', default=False, help='cuda')
    parser.add_argument('--samples', type=int, default=100, help='samples')
    parser.add_argument('--batch_size', type=int, default=10, help='batch for active learning')
    parser.add_argument('--objective', type=str, default="random", help='type of AL strategy')
    parser.add_argument('--warmstart', type=int, default=50, help='first epochs only random')
    parser.add_argument('--mining', type=int, default=500, help='first epochs mine for positives')
    args = parser.parse_args()

    args.headless = True if args.simulated else False
    args.cuda = False
    args.envs = 1
    parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    asset_root = os.path.abspath(parent_dir + "/data/shapenet_objects/")

    generator = ActiveQuerier(args, asset_root)

    objective_errors = []
    for objective in ["random"]: #["random", "max", "min", "confusion", "all"]:
        #args.objective = objective
        generator.reset_model()
        errors = generator.collect_data(args.concept, N_queries=args.samples, objective=args.objective, warmstart=args.warmstart)
        objective_errors.append(errors)
    print(objective_errors)
    generator.kill_instance()