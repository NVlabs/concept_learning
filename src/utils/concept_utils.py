# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import numpy as np

import trimesh
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull

import pytorch3d.transforms as tra3d
import torch


def concept_value(raw_state, concept):
    # Returns the concept value for the current world_instance
    if concept == "above180":
        return above_angle(raw_state)
    elif concept == "above45":
        return above_angle(raw_state, thresh=45*np.pi/180.0)
    elif concept == "abovebb":
        return above_bb(raw_state)
    elif concept == "upright":
        return upright(raw_state)
    elif concept == "upright45":
        return upright(raw_state, thresh=45*np.pi/180.0)
    elif concept == "near":
        return near(raw_state, thresh=0.3)
    elif concept == "alignedvertical":
        return aligned_vertical(raw_state, thresh=45*np.pi/180.0)
    elif concept == "alignedhorizontal":
        return aligned_horizontal(raw_state, thresh=45*np.pi/180.0)
    elif concept == "front":
        return front(raw_state)
    elif concept == "left":
        return left(raw_state)
    elif concept == "right":
        return right(raw_state)
    elif concept == "front45":
        return front(raw_state, thresh=45*np.pi/180.0)
    elif concept == "left45":
        return left(raw_state, thresh=45*np.pi/180.0)
    elif concept == "right45":
        return right(raw_state, thresh=45*np.pi/180.0)
    elif concept == "ontop":
        return ontop(raw_state, thresh=45*np.pi/180.0)
    else:
        raise NotImplementedError

def above_angle(raw_state, thresh=np.pi):
    # Return how much obj1 is above obj2.
    rel_pos = raw_state[:3] - raw_state[7:10]
    rel_pos /= np.linalg.norm(rel_pos)
    angle = np.arccos(rel_pos[2])
    return max(0.0, 1.0 - angle / thresh) # Normalize to be between 0 and 1.

def above_bb(raw_state):
    # Return how much obj1 is above obj2.
    if raw_state[2] < raw_state[9]:
        return 0

    # Unpack raw state.
    obj1_quat = torch.tensor([raw_state[6], raw_state[3], raw_state[4], raw_state[5]]).unsqueeze(0)
    obj2_quat = torch.tensor([raw_state[13], raw_state[10], raw_state[11], raw_state[12]]).unsqueeze(0)
    obj1_bb, obj2_bb = raw_state[14:20].reshape((2,3)), raw_state[20:].reshape((2,3))
    obj1_corners, obj2_corners = trimesh.bounds.corners(obj1_bb), trimesh.bounds.corners(obj2_bb)

    obj1_corners = tra3d.quaternion_apply(obj1_quat, torch.tensor(obj1_corners))
    obj1_corners += raw_state[:3]
    obj2_corners = tra3d.quaternion_apply(obj2_quat, torch.tensor(obj2_corners))
    obj2_corners += raw_state[7:10]

    obj1_corners = obj1_corners[:, :2].detach().numpy()
    obj2_corners = obj2_corners[:, :2].detach().numpy()

    obj1_hull = ConvexHull(obj1_corners)
    obj2_hull = ConvexHull(obj2_corners)

    obj1_poly = Polygon(obj1_hull.points[obj1_hull.vertices])
    obj2_poly = Polygon(obj2_hull.points[obj2_hull.vertices])
    return obj1_poly.intersection(obj2_poly).area / min(obj1_poly.area, obj2_poly.area)

def near(raw_state, thresh=0.3):
    # Return how much obj1 is near obj2.
    length = np.linalg.norm(raw_state[:3] - raw_state[7:10])
    return max(0.0, 1.0 - length / thresh) # Normalize to be between 0 and 1.

def upright(raw_state, thresh=np.pi):
    # Return how much obj1 is upright
    q = [raw_state[6], raw_state[3], raw_state[4], raw_state[5]]
    R = tra3d.quaternion_to_matrix(torch.tensor(q)).detach().numpy()
    angle = np.arccos(R[2, 2])
    return max(0.0, 1.0 - angle / thresh) # Normalize to be between 0 and 1.

def aligned_vertical(raw_state, thresh=90*np.pi/180.0):
    q1 = [raw_state[6], raw_state[3], raw_state[4], raw_state[5]]
    q2 = [raw_state[13], raw_state[10], raw_state[11], raw_state[12]]
    R1 = tra3d.quaternion_to_matrix(torch.tensor(q1)).detach().numpy()
    R2 = tra3d.quaternion_to_matrix(torch.tensor(q2)).detach().numpy()
    v1 = R1[:, 2]
    v2 = R2[:, 2]
    dot_product1 = np.dot(v1, v2)
    dot_product2 = np.dot(v1, -v2)
    angle1 = np.arccos(dot_product1)
    angle2 = np.arccos(dot_product2)
    return max(0.0, 1.0 - min(angle1, angle2) / thresh) # Normalize to be between 0 and 1.

def aligned_horizontal(raw_state, thresh=90*np.pi/180.0):
    q1 = [raw_state[6], raw_state[3], raw_state[4], raw_state[5]]
    q2 = [raw_state[13], raw_state[10], raw_state[11], raw_state[12]]
    R1 = tra3d.quaternion_to_matrix(torch.tensor(q1)).detach().numpy()
    R2 = tra3d.quaternion_to_matrix(torch.tensor(q2)).detach().numpy()
    v1 = R1[:, 0]
    v2 = R2[:, 0]
    dot_product1 = np.dot(v1, v2)
    dot_product2 = np.dot(v1, -v2)
    angle1 = np.arccos(dot_product1)
    angle2 = np.arccos(dot_product2)
    return max(0.0, 1.0 - min(angle1, angle2) / thresh) # Normalize to be between 0 and 1.

def front(raw_state, thresh=np.pi):
    rel_pos = raw_state[:3] - raw_state[7:10]
    rel_pos /= np.linalg.norm(rel_pos)
    q = [raw_state[13], raw_state[10], raw_state[11], raw_state[12]]
    R = tra3d.quaternion_to_matrix(torch.tensor(q))
    v = R[:, 0]
    dot_product = np.dot(v, rel_pos)
    angle = np.arccos(dot_product)
    return max(0.0, 1.0 - angle / thresh) # Normalize to be between 0 and 1.

def right(raw_state, thresh=np.pi):
    rel_pos = raw_state[7:10] - raw_state[:3]
    rel_pos /= np.linalg.norm(rel_pos)
    q = [raw_state[13], raw_state[10], raw_state[11], raw_state[12]]
    R = tra3d.quaternion_to_matrix(torch.tensor(q))
    v = R[:, 1]
    dot_product = np.dot(v, rel_pos)
    angle = np.arccos(dot_product)
    return max(0.0, 1.0 - angle / thresh) # Normalize to be between 0 and 1.

def left(raw_state, thresh=np.pi):
    rel_pos = raw_state[:3] - raw_state[7:10]
    rel_pos /= np.linalg.norm(rel_pos)
    q = [raw_state[13], raw_state[10], raw_state[11], raw_state[12]]
    R = tra3d.quaternion_to_matrix(torch.tensor(q))
    v = R[:, 1]
    dot_product = np.dot(v, rel_pos)
    angle = np.arccos(dot_product)
    return max(0.0, 1.0 - angle / thresh) # Normalize to be between 0 and 1.

def ontop(raw_state, thresh=np.pi):
    rel_pos = raw_state[:3] - raw_state[7:10]
    rel_pos /= np.linalg.norm(rel_pos)
    q = [raw_state[13], raw_state[10], raw_state[11], raw_state[12]]
    R = tra3d.quaternion_to_matrix(torch.tensor(q))
    v = R[:, 2]
    dot_product = np.dot(v, rel_pos)
    angle = np.arccos(dot_product)
    return max(0.0, 1.0 - angle / thresh) # Normalize to be between 0 and 1.