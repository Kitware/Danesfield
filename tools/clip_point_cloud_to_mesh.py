#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

import argparse
import logging
import numpy as np
import open3d as o3d
import pdal

def main(args):
    parser = argparse.ArgumentParser(
        description='Clips las point cloud data to the bounding box of a mesh')
    parser.add_argument("source_point_cloud", help="Source point cloud file name")
    parser.add_argument("mesh", help="Mesh that we want to clip around")
    parser.add_argument("output_point_cloud", help="Destination file name")

    args = parser.parse_args(args)

    # Read input OBJ header
    with open(args.mesh, "r") as in_f:
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

    # Read mesh to get bounds
    print(args.mesh)
    mesh = o3d.io.read_triangle_mesh(args.mesh)

    # Get shifted mesh bounds
    mesh_min = mesh.get_min_bound() + utm_shift
    mesh_max = mesh.get_max_bound() + utm_shift

    # Clip and translate las file
    pdal_input = u"""
    {{
        "pipeline":
        [
          "{}",
          {{
            "type":"filters.range",
            "limits":"X[{}:{}],Y[{}:{}],Z[{}:{}]"
          }},
          {{
            "type":"writers.las",
            "filename":"{}",
            "offset_x":"{}",
            "offset_y":"{}",
            "offset_z":"{}"
          }}
        ]
    }}"""
    print("Loading Point Cloud")
    pdal_input = pdal_input.format(args.source_point_cloud,
                                   mesh_min[0], mesh_max[0],
                                   mesh_min[1], mesh_max[1],
                                   mesh_min[2], mesh_max[2],
                                   args.output_point_cloud,
                                   utm_shift[0], utm_shift[1], utm_shift[2])

    pipeline = pdal.Pipeline(pdal_input)
    pipeline.validate()  # check if our JSON and options were good
    # this causes a segfault at the end of the program
    # pipeline.loglevel = 8  # really noisy
    pipeline.execute()
    points = pipeline.arrays[0]
    pipeline = None


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)