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
import glob

sys.path.insert(1, '../')

from src.utils.input_utils import Hdf5Cacher

np.set_printoptions(precision=2)


if __name__ == '__main__':
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--concept', type=str, default='above180', help='concept')

    args = parser.parse_args()
    parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    args.data_dir = os.path.abspath(parent_dir + "/data/concept_shapenet/" + args.concept)

    # Take all examples and separate into 0 and 1.
    hdf5cacher = Hdf5Cacher(args.data_dir + "/label.hdf5", "r")
    examples = list(hdf5cacher._hdf5().keys())

    # Check if concept is upright.
    obj_idxes = None
    if args.concept in ["upright", "upright45", "ontop"]:
        obj_idxes = [0, 1, 2, 3, 4, 5, 132, 133, 134, 135, 136, 137, 12, 13, 14, 15, 16, 17, \
                     18, 19, 20, 21, 22, 23, 36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, \
                     78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, \
                     96, 97, 98, 99, 100, 101, 126, 127, 128, 129, 130, 131]
    elif args.concept in ["alignedvertical"]:
        obj_idxes = [0, 1, 2, 3, 4, 5, 132, 133, 134, 135, 136, 137, 12, 13, 14, 15, 16, 17, \
                     36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 108, 109, 110, 111, 112, 113]
    elif args.concept in ["alignedhorizontal"]:
        obj_idxes = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 54, 55, 56, 57, 58, 59, \
                     63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 90, 91, 92, \
                     102, 103, 104, 105, 106, 107, 114, 115, 116, 117, 118, 119]
    elif args.concept in ["front", "front45"]:
        obj_idxes = [66, 67, 68, 69, 70, 71, 90, 91, 92, 126, 127, 128, 129, 130, 131]
    elif args.concept in ["left", "left45", "right", "right45"]:
        obj_idxes = [90, 91, 92]

    if obj_idxes is not None:        
        restricted = []
        for example in examples:
            objs = example.split("_")
            if args.concept in ["alignedvertical", "alignedhorizontal"]:
                if int(objs[0]) in obj_idxes and int(objs[1]) in obj_idxes:
                    restricted.append(example)
            elif args.concept in ["upright", "upright45"]:
                if int(objs[0]) in obj_idxes:
                    restricted.append(example)
            else:
                if int(objs[1]) in obj_idxes:
                    restricted.append(example)
        examples = restricted

    zero_examples = []
    one_examples = []
    for example in examples:
        sample = hdf5cacher.__getitem__(example)
        label = sample["label"].astype(np.float32)
        if np.round(label) == 1:
            one_examples.append(example)
        else:
            zero_examples.append(example)
    hdf5cacher.close()

    # Split into train, test.
    n_train, n_test = 80000, 20000
    random.shuffle(zero_examples)
    random.shuffle(one_examples)
    zero_train, zero_test = zero_examples[:int(0.8*len(zero_examples))], zero_examples[int(0.8*len(zero_examples)):]
    one_train, one_test = one_examples[:int(0.8*len(one_examples))], one_examples[int(0.8*len(one_examples)):]
    train = random.choices(zero_train, k=int(n_train/2)) + random.choices(one_train, k=int(n_train/2))
    test = random.choices(zero_test, k=int(n_test/2)) + random.choices(one_test, k=int(n_test/2))
    random.shuffle(train)
    random.shuffle(test)

    # Save the train and test files.
    train_split = args.data_dir + "/train.txt"
    with open(train_split, 'w') as f:
        for item in train:
            f.write("%s\n" % item)
    test_split = args.data_dir + "/test.txt"
    with open(test_split, 'w') as f:
        for item in test:
            f.write("%s\n" % item)