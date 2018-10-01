#!/usr/bin/env python3

import argparse
import logging
import os
import sys

import vtk
from vtk.util.numpy_support import vtk_to_numpy

""" This script triangulate a mesh using the VTK Triangle filter """


def main(args):
    parser = argparse.ArgumentParser(description="Transform a mesh into a pure triangular mesh")
    parser.add_argument('input_mesh', type=str, help='Input mesh')
    parser.add_argument('output_dir', type=str, help='Output directory')
    # Parse arguments
    args = parser.parse_args(args)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    output_mesh = os.path.join(args.output_dir, os.path.basename(args.input_mesh))

    # Read OBJ
    reader = vtk.vtkOBJReader()
    reader.SetFileName(args.input_mesh)

    # Read input OBJ header
    with open(args.input_mesh, "r") as in_f:
        lines = in_f.readlines()
    if len(lines) > 0:
        i = 0
        while i < len(lines) and len(lines[i]) > 0 and lines[i][0] == "#":
            i += 1
    else:
        print("Warning: empty input file.")
        sys.exit(0)
    header = lines[:i]

    # Triangle filter
    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputConnection(reader.GetOutputPort())
    tri_filter.Update()
    mesh = tri_filter.GetOutput()

    # Write OBJ (header + data)
    faces = mesh.GetPolys().GetData()
    faces = vtk_to_numpy(faces).reshape((-1, 4))
    vertices = mesh.GetPoints().GetData()
    vertices = vtk_to_numpy(vertices)
    with open(output_mesh, "w") as out_f:
        out_f.writelines(header)
        for v in vertices:
            out_f.write("v " + " ".join(map(str, v)) + "\n")
        for f in faces[:, 1:]:
            out_f.write("f " + " ".join(map(lambda x: str(x+1), f)) + "\n")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
