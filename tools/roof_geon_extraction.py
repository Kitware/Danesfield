#!/usr/bin/env python

"""
Wrapper tool to run Purdue roof geon extraction.

This script encapsulates running the Purdue roof geon extraction pipeline,
including segmentation, reconstruction, conversion from PLY to OBJ, and
conversion from PLY to geon JSON.
"""

import argparse
import itertools
import os
import shutil
import subprocess
import sys

from pathlib import Path

import ply2geon
import ply2obj


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

    # Parse arguments
    args = parser.parse_args(args)

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Construct file and directory names defined by tools
    segmentationFile = args.las + '_seg.txt'
    plyDir = Path(segmentationFile + '_plys')
    objDir = Path(segmentationFile + '_plys_obj')
    jsonDir = Path(segmentationFile + '_plys_json')

    # Run segmentation executable
    subprocess.run(['segmentation', args.las, args.cls, args.dtm], check=True)

    # Run reconstruction executable
    subprocess.run(['reconstruction', segmentationFile], check=True)

    # Convert PLY files to OBJ
    ply2obj.main([
        '--ply_dir', str(plyDir),
        '--dem', args.dtm,
        '--offset'
    ])

    # Convert PLY files to geon JSON
    ply2geon.main([
        '--ply_dir', str(plyDir),
        '--dem', args.dtm
    ])

    # Move output files to output directory
    for outputFile in itertools.chain(
        objDir.glob('*.obj'),
        jsonDir.glob('*.json')
    ):
        shutil.move(str(outputFile), args.output_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
