#!/usr/bin/env python

"""
Wrapper tool to run Purdue roof geon extraction.

This script encapsulates running the Purdue roof geon extraction pipeline,
including segmentation and reconstruction.
"""

import argparse
import os
import shutil
import subprocess
import sys

from pathlib import Path


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

    # Run segmentation executable
    subprocess.run(['segmentation', args.las, args.cls, args.dtm], check=True)

    # Run reconstruction executable
    segmentationFile = args.las + '_seg.txt'
    subprocess.run(['reconstruction', segmentationFile], check=True)

    # TODO: convert to OBJ; see https://gitlab.kitware.com/core3d/danesfield/merge_requests/113

    # Move generated PLY files to output directory
    plyDir = Path(args.las).parent
    for plyFile in plyDir.glob('*.ply'):
        shutil.move(str(plyFile), args.output_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
