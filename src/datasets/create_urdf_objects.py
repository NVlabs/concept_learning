# concept_learning
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the NVIDIA Source Code License [see LICENSE for details]

import os
import glob

urdf_template = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="{name}_link">
  <visual>
    <origin xyz="0.0 0.0 0.0"/>
    <geometry>
      <mesh filename="{obj_path}" scale="{scale} {scale} {scale}"/>
    </geometry>
    <material>
      <texture filename="{mtl_path}"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0.0 0.0 0.0"/>
    <geometry>
      <mesh filename="{obj_path}" scale="{scale} {scale} {scale}"/>
    </geometry>
  </collision>
  </link>
</robot>"""


def obj_to_urdf(obj_dir):
    """
    Takes a directory of object directories containing meshes and textures and converts them to urdf.
    """
    obj_asset_dirs = sorted(glob.glob("{}/**".format(obj_dir)))

    for obj in obj_asset_dirs:
        name = os.path.relpath(obj, obj_dir)
        obj_filename = obj + "/textured.obj"
        mtl_filename = obj + "/textured.mtl"
        urdf_filename = obj + "/textured.urdf"
        urdf = urdf_template.format(name=name,
                                    obj_path=obj_filename,
                                    mtl_path=mtl_filename,
                                    scale=1.0)
        with open(urdf_filename, 'w') as f:
            f.write(urdf + "\n")

if __name__ == '__main__':
    parent_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/..")
    obj_asset_root = os.path.abspath(parent_dir + "/data/ycb_objects/")
    obj_to_urdf(obj_asset_root)