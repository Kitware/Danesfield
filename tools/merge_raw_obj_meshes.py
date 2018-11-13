#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################


import argparse
import glob
import logging
import os
import sys

# This script merges several OBJ meshes into one file and check that their
# offsets are consistent (only handles vertices and faces)


def merge_files(mesh_files, output_file, check_offsets):
    offset_x_keyword = "#x offset"
    offset_y_keyword = "#y offset"
    offset_z_keyword = "#z offset"

    merged_lines = []
    current_nb_vertices = 0

    # reference offset (they are initialized with the first mesh)
    ref_offset_x = 0
    ref_offset_y = 0
    ref_offset_z = 0
    ref_use_offset = False
    need_to_init_ref_offset = True

    for filename in mesh_files:
        with open(filename) as f:
            lines = f.readlines()

        # Read mesh offset
        with_offset = False
        x_offset, y_offset, z_offset = 0, 0, 0
        if lines[0].find(offset_x_keyword) == 0:
            x_offset = float(lines[0][10:])
            with_offset = True
        if lines[1].find(offset_y_keyword) == 0:
            y_offset = float(lines[1][10:])
        if lines[2].find(offset_z_keyword) == 0:
            z_offset = float(lines[2][10:])

        if check_offsets:
            # Initialize the reference offset if needed
            if need_to_init_ref_offset:
                ref_use_offset = with_offset
                ref_offset_x = x_offset
                ref_offset_y = y_offset
                ref_offset_z = z_offset
                need_to_init_ref_offset = False
            else:
                # Else check the mesh offset with the reference
                if (ref_use_offset != with_offset or ref_offset_x != x_offset or
                        ref_offset_y != y_offset or ref_offset_z != z_offset):
                            logging.exception("Error: offsets are not consistent "
                                              "over the meshes (" + filename + ")")
                            sys.exit(1)

        # Increment the vertices index by the number of current vertices when
        # a new mesh is merged
        nb_vertices = 0
        for i in range(len(lines)):
            if lines[i][0] == "v":
                nb_vertices += 1
            elif lines[i][0] == "f":
                f = list(map(lambda x: str(int(x) + current_nb_vertices),
                             lines[i].split(" ")[1:]))
                lines[i] = " ".join(["f"] + f)+"\n"

        # Update the nb of vertices with the count of the current mesh
        current_nb_vertices += nb_vertices

        # Append the lines to the gloabl list
        # If there is an offset, we do not repeat it each time
        if with_offset:
            merged_lines += lines[3:]
        else:
            merged_lines += lines

    # If there is an offset, it is added only once at the top of the output file
    offset_lines = []
    if ref_use_offset:
        offset_lines.append(offset_x_keyword + ": " + str(ref_offset_x) + "\n")
        offset_lines.append(offset_y_keyword + ": " + str(ref_offset_y) + "\n")
        offset_lines.append(offset_z_keyword + ": " + str(ref_offset_z) + "\n")

    # Write the output file
    with open(output_file, "w") as out:
        out.writelines(offset_lines + merged_lines)


def main(args):
    parser = argparse.ArgumentParser(description='Merge meshes (only handles vertices and faces)')
    parser.add_argument("input_dir", help="Input directory containing the meshes")
    parser.add_argument("output_mesh", help="Output mesh")
    parser.add_argument("--check_offsets", action="store_true",
                        help="Check that the meshes offsets are the same")
    args = parser.parse_args(args)

    input_dir = args.input_dir
    output_mesh = args.output_mesh
    check_offsets = args.check_offsets
    mesh_files = glob.glob(os.path.join(input_dir, "*.obj"))

    if len(mesh_files) > 0:
        merge_files(mesh_files, output_mesh, check_offsets)
        print("Check offsets:", check_offsets)
        print(output_mesh + " created.")


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
