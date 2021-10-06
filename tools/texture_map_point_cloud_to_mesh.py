#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

'''
Take point cloud data and create a texture mapping to 'paint' that data on to
a mesh.
'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pdal
import sys

from pathlib import Path

from kwiver.vital.types import Mesh
from kwiver.vital.types.point import Point3d
from kwiver.vital.algo import UVUnwrapMesh
from kwiver.arrows.core import mesh_triangulate
from kwiver.arrows.core import mesh_closest_point

from kwiver.vital.modules import load_known_modules

# Load point cloud data with pdal
def load_point_cloud(filename):
    pdal_input = u"""
    {{
        "pipeline":
        [
          "{}"
        ]
    }}"""
    print("Loading Point Cloud")
    pdal_input = pdal_input.format(filename)

    print(pdal_input)
    pipeline = pdal.Pipeline(pdal_input)
    pipeline.validate()  # check if our JSON and options were good
    pipeline.execute()
    points = pipeline.arrays[0]
    pipeline = None
    return points

def main(args):
    parser = argparse.ArgumentParser(
        description="Take point cloud data and add a texture map to a mesh that "
                    " paints that data on the mesh.")
    parser.add_argument("mesh_file", help="path to mesh file")
    parser.add_argument("point_cloud_file", help="path to point cloud file")
    parser.add_argument("out_mesh_file", help="path new mesh file with texture map")
    parser.add_argument("--mtl_file", help="path to mtl file for mesh. "
                        " Default to out file name with .mtl extension.")
    args = parser.parse_args(args)

    # Load kwiver modules
    load_known_modules()

    new_mesh = Mesh.from_obj_file(args.mesh_file)

    print('mesh loaded')

    mesh_triangulate(new_mesh)

    uv_unwrap_mesh = UVUnwrapMesh.create('core')
    uv_unwrap_mesh.unwrap(new_mesh)

    pc_data = load_point_cloud(args.point_cloud_file)

    points = np.stack([pc_data['X'], pc_data['Y'], pc_data['Z']], axis=1)

    img_size = (1000, 1000, 3)
    img_arr = np.zeros(img_size, dtype=np.uint8)

    coords = []
    idx = 0
    u = v = 0.0
    closest_point = Point3d()
    point = Point3d()

    pc_min = np.zeros(3)
    pc_max = np.zeros(3)

    for i in range(3):
      pc_min[i] = np.min(points[:,i])
      pc_max[i] = np.max(points[:,i])
    pc_len = pc_max - pc_min

    for pt in points:
        point.value = pt
        (idx, u, v) = mesh_closest_point(point, new_mesh, closest_point)
        tx_coord = new_mesh.texture_map(idx, u, v)
        coords.append(tx_coord)
        for i in range(3):
            val = 63 + int(192.0*(pt[i] - pc_min[i])/pc_len[i])
            img_arr[int((1.-tx_coord[1])*img_size[1]), int(tx_coord[0]*img_size[0]), i] = val

    img_arr = 255.0*img_arr/np.max(img_arr)
    img_arr = img_arr.astype(np.uint8)

    plt.imsave(Path(args.mesh_file).with_suffix('.png'), img_arr)

    new_mesh.set_tex_source(Path(args.out_mesh_file).with_suffix('.mtl').name)
    Mesh.to_obj_file(args.out_mesh_file, new_mesh)

if __name__ == '__main__':
    main(sys.argv[1:])
