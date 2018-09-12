#!/usr/bin/env python

"""
Wrapper tool to run Purdue and Columbia's roof geon extraction.

This script encapsulates running the Purdue and Columbia roof geon
extraction pipeline, including segmentation, curve fitting,
reconstruction, conversion from PLY to OBJ, and conversion from PLY to
geon JSON.
"""

import argparse
import itertools
import os
import shutil
import subprocess
import sys
import re

from pathlib import Path

import roof_segmentation
import fitting_curved_plane
import geon_to_mesh
import ply2geon
import ply2obj


def convert_lastext_to_las(infile, outfile):
    subprocess.run(['txt2las', '-i', infile, '-o', outfile, '-parse', 'xyzc'],
                   check=True)


def main(args):
    # Configure argument parser
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--las',
        type=str,
        required=True,
        help='Point Cloud File in LAS format')
    parser.add_argument(
        '--cls',
        type=str,
        required=True,
        help='Class Label (CLS) file')
    parser.add_argument(
        '--dtm',
        type=str,
        required=True,
        help='Digital Terrain Model (DTM) file')
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory')
    parser.add_argument(
        '--model_prefix',
        type=str,
        required=True,
        help='Prefix for model files (e.g. "dayton_geon")')
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directory containing the model files')

    # Parse arguments
    args = parser.parse_args(args)

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Step #1
    # Run segmentation executable
    print("* Running Purdue's segmentation on P3D point cloud")
    subprocess.run(['segmentation', args.las, args.cls, args.dtm], check=True)
    # Output files:
    building_segmentation_txt = "{}_bd.txt".format(args.las)
    # Road segmentation output only produced when we have roads
    # labeled in the CLS file
    road_segmentation_txt = "{}_road_seg.txt".format(args.las)

    # Step #2
    # Run Columbia's roof segmentation script
    print("* Running Columbia's roof segmentation")
    roof_segmentation_png = os.path.join(args.output_dir, "roof_seg.png")
    roof_segmentation_txt = os.path.join(args.output_dir,
                                         "roof_seg_outlas.txt")
    roof_segmentation.main(['--model_prefix', args.model_prefix,
                            '--model_dir', args.model_dir,
                            '--input_pc', building_segmentation_txt,
                            '--output_png', roof_segmentation_png,
                            '--output_txt', roof_segmentation_txt])

    # Step #3
    # Run Columbia curve plane fitting
    print("* Running Columbia's curve fitting")
    curve_fitting_png = os.path.join(args.output_dir, "curve_fit.png")
    curve_fitting_geon = os.path.join(args.output_dir,
                                      "curve_fitting_output_geon.geon")
    curve_fitting_remaining_txt = \
        os.path.join(args.output_dir,
                     "curve_fitting_remaining_outlas.txt")
    fitting_curved_plane.main(['--input_pc', roof_segmentation_txt,
                               '--output_png', curve_fitting_png,
                               '--output_txt', curve_fitting_remaining_txt,
                               '--output_geon', curve_fitting_geon])

    # Step #4
    # Run Columbia curve mesh generation
    print("* Running Columbia's geon to mesh")
    mesh_output = os.path.join(args.output_dir, "output_curves.ply")
    geon_to_mesh.main(['--input_geon', curve_fitting_geon,
                       '--input_dtm', args.dtm,
                       '--output_mesh', mesh_output])

    # Step #3_5 (Note the step numbering here is in reference to the
    # data flow diagram provided by Purdue / Columbia)
    # Run Purdue's Segmentation / Reconstruction on the points
    # leftover from Columbia's roof segmentation
    # Purdue's Segmentation code expects a binary LAS file, so we
    # first convert it
    print("* Converting remaining points from las text to las")
    curve_fitting_remaining_las = \
        os.path.join(args.output_dir,
                     "curve_fitting_remaining_outlas.las")
    convert_lastext_to_las(curve_fitting_remaining_txt,
                           curve_fitting_remaining_las)

    print("* Running Purdue's segmentation on remaining points las")
    subprocess.run(['segmentation', curve_fitting_remaining_las], check=True)
    # Output:
    curve_fitting_remaining_las_seg = "{}_seg.txt".format(
        curve_fitting_remaining_las)

    # Step #3_6
    print("* Running Purdue's reconstruction on segmented remaining points")
    subprocess.run(['reconstruction',
                    curve_fitting_remaining_las_seg],
                   check=True)

    # Road files will get written to the following paths if the road
    # segmentation exists
    road_segmentation_las = "{}_road_seg.las".format(args.las)
    road_segmentation_las_seg = "{}_seg.txt".format(
        road_segmentation_las)
    if os.path.exists(road_segmentation_txt):
        print("* Found road segmentation output")

        # Step #1_5
        # Process road segmentation results
        print("* Converting road segmentation points from las text to las")
        convert_lastext_to_las(road_segmentation_txt,
                               road_segmentation_las)

        print("* Running Purdue's segmentation on road points las")
        subprocess.run(['segmentation',
                        road_segmentation_las],
                       check=True)

        # Step #1_6
        # Reconstruct road segmentation results
        print("* Running Purdue's reconstruction on segmented road points")
        subprocess.run(['reconstruction',
                        road_segmentation_las_seg], check=True)

    # Collate our ply files
    remaining_ply_dir = "{}_plys".format(curve_fitting_remaining_las_seg)
    road_ply_dir = "{}_plys".format(road_segmentation_las_seg)

    all_ply_dir = os.path.join(args.output_dir, "all_plys")
    # Create all ply output directory
    if not os.path.exists(all_ply_dir):
        os.makedirs(all_ply_dir)

    # Move all ply files to the same directory
    for f in itertools.chain(Path(remaining_ply_dir).glob("*.ply"),
                             # Path.glob doesn't complain if the directory
                             # doesn't exist
                             Path(road_ply_dir).glob("*.ply"),
                             [mesh_output]):
        shutil.move(str(f), all_ply_dir)

    # Step #7
    # Convert all of our PLY files to OBJ
    print("* Converting PLY files to OBJ")
    ply2obj.main(['--ply_dir', all_ply_dir,
                  '--dem', args.dtm,
                  '--offset'])

    print("* Converting PLY files to geon JSON")
    # Convert all PLY files to geon JSON
    ply2geon.main(['--ply_dir', all_ply_dir,
                   '--dem', args.dtm])

    # The buildings_to_dsm.py script requires a certain naming
    # convention for the OBJ files based on whether they're roads or
    # buildings, this step copies / renames the OBJ files to suit that
    # convention
    print("* Collating OBJ files for buildings_to_dsm.py")
    # Output obj directory; want to collate the renamed OBJ files into
    # a single output directory
    output_obj_dir = os.path.join(args.output_dir, "output_obj")
    if not os.path.exists(output_obj_dir):
        os.makedirs(output_obj_dir)

    all_obj_dir = "{}_obj".format(all_ply_dir)
    road_re = re.compile("road_([0-9]+)\\.obj")
    for f in Path(all_obj_dir).glob("road_*.obj"):
        match = re.match(road_re, os.path.basename(str(f)))
        if match:
            out_fname = "Road_{}.obj".format(match[1])
            shutil.copy(str(f), os.path.join(output_obj_dir, out_fname))
        else:
            print("** Warning: couldn't match road OBJ filename '{}' with \
expected regexp.  Skipping".format(str(f)))

    # Copy the rest of the OBJ files
    for i, f in enumerate(sorted(Path(all_obj_dir).glob("*.obj"), key=str)):
        basename = os.path.basename(str(f))
        # Prefix the filename with a number
        shutil.copy(str(f), os.path.join(output_obj_dir, "{}_{}".format(i, basename)))


if __name__ == '__main__':
    main(sys.argv[1:])
