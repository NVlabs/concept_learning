# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import numpy as np
import enum
import copy
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import pytorch3d.transforms as tra3d

import torch

class SegLabel(enum.Enum):
    BACKGROUND = 0
    OBJECTS = 1
    MOVING = 2
    ANCHOR = 3
    ROBOT = 4

def get_pointcloud_from_depth(camera_data):
    proj_matrix = camera_data['proj_matrix']

    fu = 2 / proj_matrix[0, 0]
    fv = 2 / proj_matrix[1, 1]
    seg_buffer = camera_data['mask']
    depth_buffer = camera_data['depth']
    cam_width = camera_data['depth'].shape[1]
    cam_height = camera_data['depth'].shape[0]
    cu = cam_width / 2
    cv = cam_height / 2

    # Get the camera transform.
    cp = np.matrix(camera_data['view_matrix'])

    grid = np.indices((cam_width, cam_height))
    rows, cols = np.ravel(grid[0]), np.ravel(grid[1])  
    camu = fu * (cu - rows) / cam_width # Translation in image-space coordinate
    camv = fv * (cols - cv) / cam_height # Translation in image-space coordinate
    points = np.vstack((camu, camv, np.ones((1, cam_width*cam_height))))
    points = np.multiply(np.ravel(depth_buffer.T), points)
    points = np.vstack((points, np.ones((1, cam_width*cam_height))))
    points = (cp * points).T

    camera_data['pc'] = points[:,:3]
    camera_data['pc_seg'] = np.ravel(seg_buffer.T)
    return camera_data

def get_geometries_from_pc(x, shadow=False):
    import open3d
    if shadow:
        moving_idx = np.where(x[:,3]>0)[0]
        rest_idx = np.where(x[:,3]==0)[0]
        moving_pts = x[moving_idx]
        rest_pts = x[rest_idx]
        pcd = open3d.geometry.PointCloud()
        xyz1 = rest_pts[:,:3]
        xyz2 = moving_pts[:,:3]
        colors1 = rest_pts[:,3:]
        colors2 = moving_pts[:,3:]

        pcd.points = open3d.utility.Vector3dVector(xyz1)
        pcd.colors = open3d.utility.Vector3dVector(colors1)
        pcd.transform([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])# Transform it so it's not upside down
        geometries = [pcd]
        for id in range(xyz2.shape[0]):
            mesh = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(colors2[id])
            mesh.translate(xyz2[id, :], relative=False)
            mesh.transform([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])# Transform it so it's not upside down
            geometries.append(mesh)
    else:
        pcd = open3d.geometry.PointCloud()
        xyz = x[:,:3]
        colors = x[:,3:]
        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(colors)
        pcd.transform([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])# Transform it so it's not upside down
        geometries = [pcd]
    return geometries

def create_mesh(vertices, faces, color=[0.16, 0.62, 0.56]):
    import open3d
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(vertices)
    mesh.triangles = open3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)   
    return mesh

def show_pcs_with_mesh(x, vertices, faces, shadow=False):
    """ Display point clouds with mesh """
    import open3d

    # add pointcloud.
    geometries = get_geometries_from_pc(x, shadow=shadow)

    # add mesh
    mesh = create_mesh(vertices, faces, color=[0.16, 0.62, 0.56])
    geometries.append(mesh)
    open3d.visualization.draw_geometries(geometries)

def show_pcs_with_frame(x, new_origin, shadow=False):
    """ Display point clouds """
    import open3d
    geometries = get_geometries_from_pc(x, shadow=shadow)
    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=new_origin)
    mesh_frame.transform([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])# Transform it so it's not upside down
    geometries.append(mesh_frame)
    open3d.visualization.draw_geometries(geometries)

def show_pcs(x, shadow=False):
    """ Display point clouds """
    import open3d
    geometries = get_geometries_from_pc(x, shadow=shadow)
    open3d.visualization.draw_geometries(geometries)

def show_pcs_gif(pcd_list, shadow=False):
    import open3d
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    frames = []
    for i in range(len(pcd_list)):
        pc = pcd_list[i]
        geometries = get_geometries_from_pc(pc, shadow=shadow)
        for pcd in geometries:
            vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(True)
        frames.append((np.asarray(img)*255).astype(np.uint8))
        for pcd in geometries:
            vis.remove_geometry(pcd)

    # save it as a gif
    clip = ImageSequenceClip(frames, fps=4)
    clip.write_gif('test.gif', fps=4)

def initialize_T_around_pts(moving_pts, anchor_pts):
    anchor_pts[:, :, :3] -= np.mean(moving_pts, axis=1).reshape((anchor_pts.shape[0],1,3)).repeat(anchor_pts.shape[1],axis=1)
    pts_center = np.mean(anchor_pts, axis=1)
    # Add small noise to xyz.
    xyz = pts_center + np.random.randn(*pts_center.shape) * 0.1
    # Initialize random rotation.
    quat = np.random.randn(pts_center.shape[0], 4)
    quat = quat / np.linalg.norm(quat, axis=1)[:, None]
    return np.hstack((xyz, quat))

def move_points(pts, T):
    moving_idx = pts[0, :, 3]==1
    rest_idx = pts[0, :, 3]==0

    # Convert quat to rotation matrix.
    R = tra3d.quaternion_to_matrix(T[:,3:])
    # Zero-center the points before applying rotation.
    moving_pts = pts[:,moving_idx,:3]
    movingpts_rot = torch.matmul(moving_pts, torch.transpose(R, 1, 2))
    # Move back to original translation before applying the translation.
    movingpts_trans = torch.add(movingpts_rot, T[:,:3].unsqueeze(1).repeat(1, sum(moving_idx), 1))
    movingpts_new = torch.cat((movingpts_trans, pts[:,moving_idx,3:]), axis=2)
    pts_new = torch.cat((movingpts_new, pts[:,rest_idx,:]), dim=1)
    return pts_new

def add_noise_to_pose(pose, p_sigma=0.01, r_sigma=0.3):
    pose[:3] += np.random.randn(3) * p_sigma
    q = [pose[6], pose[3], pose[4], pose[5]]
    R = tra3d.quaternion_to_matrix(torch.tensor(q))
    euler = tra3d.matrix_to_euler_angles(R, "XYZ")
    e_new = euler + np.random.randn(3) * r_sigma
    R_new = tra3d.euler_angles_to_matrix(e_new, "XYZ")
    q_new = tra3d.matrix_to_quaternion(R_new)
    pose[3:7] = [q_new[1], q_new[2], q_new[3], q_new[0]]

def transform_rawstate(rawstate, T):
    pose1 = copy.deepcopy(rawstate[:,:7])
    pose2 = torch.cat((T[:,:3], T[:,4:7], T[:,3].unsqueeze(1)), dim=1)
    pose_new = compose_qq(pose2, pose1).to(pose1.device)
    return torch.cat((pose_new, rawstate[:,7:]), dim=1)

def pose_to_T(pose):
    q = torch.cat((pose[:,6].unsqueeze(1), pose[:,3:6]), axis=1)
    R = tra3d.quaternion_to_matrix(q)
    T = torch.cat((R, torch.zeros(R.shape[0], 1, 3).to(pose.device)), axis=1)
    T = torch.cat((T, torch.cat((pose[:,:3], torch.ones(pose.shape[0],1).to(pose.device)), axis=1).unsqueeze(-1)), axis=2)
    return T

def relative_pose(pose1, pose2):
    T1 = pose_to_T(pose1)
    T2 = pose_to_T(pose2)
    T_delta = torch.matmul(torch.inverse(T2), T1)
    q_delta = tra3d.matrix_to_quaternion(T_delta[:,:3,:3])
    return torch.cat((T_delta[:, :3, 3], q_delta[:,1:], q_delta[:,0].unsqueeze(1)), axis=1)

def relative_difference(pose1, pose2):
    q1 = torch.cat((pose1[:,6].unsqueeze(1), pose1[:,3:6]), axis=1)
    R1 = tra3d.quaternion_to_matrix(q1)
    q2 = torch.cat((pose2[:,6].unsqueeze(1), pose2[:,3:6]), axis=1)
    R2 = tra3d.quaternion_to_matrix(q2)
    R_delta = torch.matmul(torch.transpose(R2, 1, 2), R1)
    q_delta = tra3d.matrix_to_quaternion(R_delta)
    return torch.cat((pose1[:,:3] - pose2[:,:3], q_delta[:,1:], q_delta[:,0].unsqueeze(1)), axis=1)

def corners_from_rawstate(x):
    import trimesh
    obj1_quat = torch.cat((x[:,6].unsqueeze(1), x[:,3:6]), axis=1)
    obj2_quat = torch.cat((x[:,13].unsqueeze(1), x[:,10:13]), axis=1)
    obj1_bb, obj2_bb = x[:,14:20].reshape((x.shape[0],2,3)), x[:,20:].reshape((x.shape[0],2,3))
    #obj1_corners, obj2_corners = trimesh.bounds.corners(obj1_bb), trimesh.bounds.corners(obj2_bb)
    obj1_corners = torch.stack([torch.tensor(trimesh.bounds.corners(bb.cpu())).to(x.device) for bb in obj1_bb])
    obj2_corners = torch.stack([torch.tensor(trimesh.bounds.corners(bb.cpu())).to(x.device) for bb in obj2_bb])

    obj1_corners = tra3d.quaternion_apply(obj1_quat, torch.tensor(obj1_corners))
    obj1_corners += x[:,:3]
    obj2_corners = tra3d.quaternion_apply(obj2_quat, torch.tensor(obj2_corners))
    obj2_corners += x[:,7:10]

    obj1_corners = obj1_corners[:, :2]
    obj2_corners = obj2_corners[:, :2]
    return torch.cat((obj1_corners, obj2_corners), axis=1)

def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4). [qw, qx,qy,qz]
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    zero = matrix.new_zeros((1,))
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * torch.sqrt(torch.max(zero, 1 + m00 + m11 + m22))
    x = 0.5 * torch.sqrt(torch.max(zero, 1 + m00 - m11 - m22))
    y = 0.5 * torch.sqrt(torch.max(zero, 1 - m00 + m11 - m22))
    z = 0.5 * torch.sqrt(torch.max(zero, 1 - m00 - m11 + m22))
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def compose_qp(q, pt):
    '''
    Assume we're dealing with unit quaternions. $q$ is a quaternion, $p$ is a
    position.
    '''
    pt2 = torch.zeros(q.shape[0], 3)
    px, py, pz = pt[:,0], pt[:,1], pt[:,2]
    x = q[:,0]
    y = q[:,1]
    z = q[:,2]
    qx = q[:,3]
    qy = q[:,4]
    qz = q[:,5]
    qw = q[:,6]
    qxx = qx**2
    qyy = qy**2
    qzz = qz**2
    qwx = qw*qx
    qwy = qw*qy
    qwz = qw*qz
    qxy = qx*qy
    qxz = qx*qz
    qyz = qy*qz
    pt2[:, 0] = x + px + 2*(
            (-1*(qyy + qzz)*px) +
            ((qxy-qwz)*py) +
            ((qwy+qxz)*pz))
    pt2[:,1] = y + py + 2*(
            ((qwz+qxy)*px) +
            (-1*(qxx+qzz)*py) +
            ((qyz-qwx)*pz))
    pt2[:,2] = z + pz + 2*(
            ((qxz-qwy)*px) +
            ((qwx+qyz)*py) +
            (-1*(qxx+qyy)*pz)
            )
    return pt2

def compose_qq(q1, q2):
    '''
    Compose two poses represented as:
        [x,y,z,qx,qy,qz,qw]
    '''
    QX, QY, QZ, QW = 3, 4, 5, 6
    qww = q1[:,QW]*q2[:,QW]
    qxx = q1[:,QX]*q2[:,QX]
    qyy = q1[:,QY]*q2[:,QY]
    qzz = q1[:,QZ]*q2[:,QZ]
    # For new qx
    q1w2x = q1[:,QW]*q2[:,QX]
    q2w1x = q2[:,QW]*q1[:,QX]
    q1y2z = q1[:,QY]*q2[:,QZ]
    q2y1z = q2[:,QY]*q1[:,QZ]
    # For new qy
    q1w2y = q1[:,QW]*q2[:,QY]
    q2w1y = q2[:,QW]*q1[:,QY]
    q1z2x = q1[:,QZ]*q2[:,QX]
    q2z1x = q2[:,QZ]*q1[:,QX]
    # For new qz
    q1w2z = q1[:,QW]*q2[:,QZ]
    q2w1z = q2[:,QW]*q1[:,QZ]
    q1x2y = q1[:,QX]*q2[:,QY]
    q2x1y = q2[:,QX]*q1[:,QY]
    q3 = torch.zeros(q1.shape)
    q3[:,:3] = compose_qp(q1,q2[:,:3])
    q3[:,QX] = (q1w2x+q2w1x+q1y2z-q2y1z)
    q3[:,QY] = (q1w2y+q2w1y+q1z2x-q2z1x)
    q3[:,QZ] = (q1w2z+q2w1z+q1x2y-q2x1y)
    q3[:,QW] = (qww - qxx - qyy - qzz)
    return q3