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
from kwiver.arrows.core import mesh_closest_points

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
    parser.add_argument("--bit_depth", help="bit depth of the colors in "
                        "the las file.", type=int, default=8)
    args = parser.parse_args(args)

    # Load kwiver modules
    load_known_modules()

    # Check for UTM corrections in mesh file header
    with open(args.mesh_file, "r") as in_f:
        header = [next(in_f) for x in range(3)]

    # Set the shift values for the mesh from header data
    utm_shift = np.zeros(3)
    for l in header:
      cols = l.split()
      if '#x' in cols[0]:
          utm_shift[0] = float(cols[2])
      elif '#y' in cols[0]:
          utm_shift[1] = float(cols[2])
      elif '#z' in cols[0]:
          utm_shift[2] = float(cols[2])

    new_mesh = Mesh.from_obj_file(args.mesh_file)

    print('mesh loaded')

    mesh_triangulate(new_mesh)

    uv_unwrap_mesh = UVUnwrapMesh.create('core')
    uv_unwrap_mesh.unwrap(new_mesh)

    pc_data = load_point_cloud(args.point_cloud_file)

    points = []

    for i in range(len(pc_data)):
        point = Point3d()
        point.value = (pc_data['X'][i] - utm_shift[0],
                       pc_data['Y'][i] - utm_shift[1],
                       pc_data['Z'][i] - utm_shift[2])
        points.append(point)

    rgb_data = np.stack([pc_data['Red'], pc_data['Green'], pc_data['Blue']], axis=1)

    img_size = (1000, 1000, 3)
    img_pre_arr = [ [ [] for i in range(img_size[0]) ] for j in range(img_size[1]) ]
    img_arr = np.zeros(img_size, dtype=np.float64)

    closest_points = []
    uv_coords = mesh_closest_points(points, new_mesh, closest_points)

    print("UV coordinates calculated")

    for (idx, u, v), rgb in zip(uv_coords, rgb_data):
        tx_coord = new_mesh.texture_map(idx, u, v)
        px, py = int((1.-tx_coord[1])*img_size[1]), int(tx_coord[0]*img_size[0])
        img_pre_arr[px][py].append(rgb.astype(np.float64))

    # Take the average at each pixel
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            if img_pre_arr[i][j]:
                img_arr[i, j] = np.array(img_pre_arr[i][j]).mean(axis=0)

    img_arr = img_arr/2**args.bit_depth

    plt.imsave(Path(args.out_mesh_file).with_suffix('.png'), img_arr)

    new_mesh.set_tex_source(Path(args.out_mesh_file).with_suffix('.mtl').name)
    Mesh.to_obj_file(args.out_mesh_file, new_mesh)

if __name__ == '__main__':
    main(sys.argv[1:])
