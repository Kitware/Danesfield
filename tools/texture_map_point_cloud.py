#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

'''
Take point cloud data and create a texture mapping to 'paint' that data on to
a mesh. Two methods are supported. 'Splatting' where each point is mapped to the
nearest mesh triangle or 'sampling' where each mesh triangle uses the value from
the nearest point in the point cloud.
'''

import argparse
import imageio
import json
import numpy as np
import pdal
import sys

from pathlib import Path
from scipy.spatial import KDTree
from time import time

from danesfield.gpm import GPM
from danesfield.gpm_decode64 import json_numpy_array_hook

from kwiver.vital.types import Mesh
from kwiver.vital.algo import UVUnwrapMesh
from kwiver.arrows.core import mesh_triangulate

from kwiver.vital.modules import load_known_modules

mtl_template = ("newmtl mat\n"
                "Ka 1.0 1.0 1.0\n"
                "Kd 1.0 1.0 1.0\n"
                "d 1\n"
                "Ns 75\n"
                "illum 1\n"
                "map_Kd {}\n")

# Timer decorator to assess performance
def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

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

# Get the barycentric coordinates for a point on a triangle
def barycentric(r, p1, p2, p3):
    denom = (p2[1]-p3[1])*(p1[0]-p3[0]) + (p3[0]-p2[0])*(p1[1]-p3[1])
    return ( ((p2[1]-p3[1])*(r[0]-p3[0]) + (p3[0]-p2[0])*(r[1]-p3[1]))/denom,
             ((p3[1]-p1[1])*(r[0]-p3[0]) + (p1[0]-p3[0])*(r[1]-p3[1]))/denom )

# Get the square of the area of a triangle with Heron's formula
def tri_area(tri):
    lens = np.array([np.linalg.norm(tri[i]-tri[(i+1)%3]) for i in range(3)])
    s = 0.5*np.sum(lens)
    return s*(s-lens[0])*(s-lens[1])*(s-lens[2])

# Class to support texture mapping point cloud data to meshes
class pointCloudTextureMapper(object):
    def __init__(self, points, output_dir, color_data, err_data=None):

        # Size of texture image
        self.img_size = (500, 500)

        self.points = points

        # KDTree for efficient searches of closest point
        self.search_tree = KDTree(points)

        # Color data to be mapped to meshes
        self.color_data = color_data

        # Error data to be mapped to meshes
        self.err_data = err_data

        # Location to save data
        self.output_dir = output_dir

    # Create a texture map image by finding the nearest point to a pixel and using
    # its value to set the color.
    def texture_sample(self, img, mesh, data, utm_shift):

        faces = mesh.faces()
        vertices = np.array(mesh.vertices())

        for v in vertices:
            v += utm_shift

        img_size = img.shape
        img_dx = 1./img_size[0]
        img_dy = 1./img_size[1]

        for i in range(mesh.num_faces()):
            x_min = y_min = 1.
            x_max = y_max = 0.
            tx_coords = [mesh.texture_map(i, u, v)
                         for (u,v) in [(0,1), (1,0), (0,0)]]
            for tmp_crds in tx_coords:
                x_min = min(tmp_crds[0], x_min)
                y_min = min(tmp_crds[1], y_min)
                x_max = max(tmp_crds[0], x_max)
                y_max = max(tmp_crds[1], y_max)

            indices = faces[i]
            corners = np.array([vertices[idx] for idx in indices])
            if tri_area(corners) <= 0. or np.isnan([x_min, x_max, y_min, y_max]).any():
                continue

            pixel_points = []
            pixel_indices = []

            for x in np.arange(x_min, x_max + img_dx, img_dx):
                for y in np.arange(y_min, y_max + img_dy, img_dy):
                    (u,v) = barycentric([x, y], tx_coords[0], tx_coords[1], tx_coords[2])
                    if (0. <= u <= 1.) and (0. <= v <= 1.) and (u + v <= 1.):
                        pixel_points.append((1. - u - v)*corners[0, :] +
                                            v*corners[1, :] +
                                            u*corners[2, :])

                        pixel_indices.append([int((1. - y)*img_size[1]),
                                              int(x*img_size[0])])

            closest_indices = self.search_tree.query(pixel_points)

            for px, ci in zip(pixel_indices, closest_indices[1]):
                img[px[0], px[1], :] = data[ci]

    def utm_shift(self, meshfile):
    # Check for UTM corrections in mesh file header
        with open(meshfile, "r") as in_f:
            header = [next(in_f) for x in range(3)]

        # Set the shift values for the mesh from header data
        shift = np.zeros(3)
        for l in header:
            cols = l.split()
            if '#x' in cols[0]:
                shift[0] = float(cols[2])
            elif '#y' in cols[0]:
                shift[1] = float(cols[2])
            elif '#z' in cols[0]:
                shift[2] = float(cols[2])

        return shift

    def process_mesh(self, meshfile):
        print('Processing mesh ', meshfile)
        new_mesh = Mesh.from_obj_file(str(meshfile))
        new_name = self.output_dir / meshfile.stem

        mesh_triangulate(new_mesh)
        uv_unwrap_mesh = UVUnwrapMesh.create('core')
        uv_unwrap_mesh.unwrap(new_mesh)

        # Create the color texture image
        color_img = np.zeros(self.img_size + (3,), dtype=np.float32)

        self.texture_sample(color_img, new_mesh, self.color_data, self.utm_shift(meshfile))
        imageio.imwrite(new_name.with_suffix('.png'), color_img)

        with open(new_name.with_suffix('.mtl'), 'w') as f:
            f.write(mtl_template.format(new_name.with_suffix('.png').name))

        new_mesh.set_tex_source(new_name.with_suffix('.mtl').name)

        if self.err_data is not None:
            err_img = np.zeros(self.img_size + (3, 3), dtype=np.float32)
            err_img[:] = np.nan
            self.texture_sample(err_img, new_mesh, self.err_data, self.utm_shift(meshfile))
            for i in range(3):
                for j in range(i+1):
                    tiff_name = Path(str(new_name) + f'_{i}_{j}')
                    imageio.imwrite(tiff_name.with_suffix('.tiff'), err_img[:,:,i,j])

        Mesh.to_obj_file(str(new_name.with_suffix('.obj')), new_mesh)

def main(args):
    parser = argparse.ArgumentParser(
        description="Texture map either point cloud data or GPM error data "
                    "associated with the point cloud.")
    parser.add_argument("mesh_dir", help="path to directory holding the mesh files")
    parser.add_argument("point_cloud", help="path to point cloud file")
    parser.add_argument("--gpm_json", help="JSON file with extracted GPM metadata")
    parser.add_argument("--output_dir", help="directory to save the results. "
                        "Defaults to mesh_dir")
    args = parser.parse_args(args)

    # Load kwiver modules
    timer_func(load_known_modules())

    # Get list of mesh files
    mesh_files = list(Path(args.mesh_dir).glob('*.obj'))

    # Create output directory if needed
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_dir():
            output_dir.mkdir()
    else:
        output_dir = Path(args.mesh_dir)

    pc_data = load_point_cloud(args.point_cloud)
    points = np.stack([pc_data['X'], pc_data['Y'], pc_data['Z']], axis=1)

    # Get color data from point cloud
    color_data = np.stack([pc_data['Red'], pc_data['Green'], pc_data['Blue']], axis=1)
    color_data = color_data/np.max(color_data)

    # Get error data from gpm metadata if available
    if args.gpm_json:
        with open(args.gpm_json) as mf:
            gpm = GPM(json.load(mf, object_hook=json_numpy_array_hook))

        if 'PPE_LUT_Index' in pc_data.dtype.names:
            gpm.setupPPELookup(points, pc_data['PPE_LUT_Index'])

        print("Getting error")
        err_data = gpm.get_per_point_error(points)
    else:
        err_data = None

    texMapper = pointCloudTextureMapper(points, output_dir, color_data, err_data)

    for mf in mesh_files:
        texMapper.process_mesh(mf)

if __name__ == '__main__':
    main(sys.argv[1:])
