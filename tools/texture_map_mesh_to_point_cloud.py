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
import math
import matplotlib.pyplot as plt
import numpy as np
import pdal
import sys

from pathlib import Path
from scipy.spatial import KDTree

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

def barycentric(r, p1, p2, p3):
    denom = (p2[1]-p3[1])*(p1[0]-p3[0]) + (p3[0]-p2[0])*(p1[1]-p3[1])
    return ( ((p2[1]-p3[1])*(r[0]-p3[0]) + (p3[0]-p2[0])*(r[1]-p3[1]))/denom,
             ((p3[1]-p1[1])*(r[0]-p3[0]) + (p1[0]-p3[0])*(r[1]-p3[1]))/denom )

def main(args):
    parser = argparse.ArgumentParser(
        description="Take point cloud data and add a texture map to a mesh that "
                    " paints that data on the mesh.")
    parser.add_argument("mesh_file", help="path to mesh file")
    parser.add_argument("point_cloud_file", help="path to point cloud file")
    parser.add_argument("out_mesh_file", help="path new mesh file with texture map")
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

    print('triangulated')

    uv_unwrap_mesh = UVUnwrapMesh.create('core')
    uv_unwrap_mesh.unwrap(new_mesh)

    print('mesh uv unwrapped')

    pc_data = load_point_cloud(args.point_cloud_file)
    points = (np.stack([pc_data['X'], pc_data['Y'], pc_data['Z']], axis=1)
              - utm_shift)

    print('point cloud data loaded')

    rgb_data = np.stack([pc_data['Red'], pc_data['Green'], pc_data['Blue']], axis=1)

    search_tree = KDTree(points)

    img_size = (1000, 1000, 3)
    img_dx = 1./img_size[0]
    img_dy = 1./img_size[1]
    img_arr = np.zeros(img_size, dtype=np.float64)

    faces = new_mesh.faces()
    vertices = new_mesh.vertices()

    for i in range(new_mesh.num_faces()):
        x_min = y_min = 1.
        x_max = y_max = 0.
        tx_coords = [new_mesh.texture_map(i, u, v)
                     for (u,v) in [(0,1), (1,0), (0,0)]]
        for tmp_crds in tx_coords:
            x_min = min(tmp_crds[0], x_min)
            y_min = min(tmp_crds[1], y_min)
            x_max = max(tmp_crds[0], x_max)
            y_max = max(tmp_crds[1], y_max)

        indices = faces[i]
        corners = np.array([vertices[idx] for idx in indices])

        pixel_points = []
        pixel_indices = []

        for x in np.arange(x_min, x_max + img_dx, img_dx):
            for y in np.arange(y_min, y_max + img_dy, img_dy):
                (u,v) = barycentric([x, y], tx_coords[0], tx_coords[1], tx_coords[2])
                if (0. <= u <= 1.) and (0. <= v <= 1.) and (u + v <= 1.):
                    pixel_points.append(u*corners[0, :] +
                                        v*corners[1, :] +
                                        (1. - u - v)*corners[2, :])

                    pixel_indices.append([int((1. - y)*img_size[1]),
                                          int(x*img_size[0])])

        closest_indices = search_tree.query(pixel_points)

        for px, ci in zip(pixel_indices, closest_indices[1]):
            img_arr[px[0], px[1], :] = rgb_data[ci]

        sys.stdout.write('\rMesh {}/{}'.format((i+1), new_mesh.num_faces()))
        sys.stdout.flush()

    img_arr = img_arr/np.max(rgb_data)

    plt.imsave(Path(args.out_mesh_file).with_suffix('.png'), img_arr)

    new_mesh.set_tex_source(Path(args.out_mesh_file).with_suffix('.mtl').name)
    Mesh.to_obj_file(args.out_mesh_file, new_mesh)

if __name__ == '__main__':
    main(sys.argv[1:])
