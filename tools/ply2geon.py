#!/usr/bin/env python

import os
import sys
import time
import argparse
from danesfield.surface.geon import Geon


def main(args):
    parser = argparse.ArgumentParser(
        description='Generate OBJ file from PLY file.')
    parser.add_argument('-p', '--ply_dir',
                        help='PLY file folder to read', required=True)
    parser.add_argument('-d', '--dem',
                        help='DEM file name to read', required=True)
    args = parser.parse_args(args)

    if not os.path.isdir(args.ply_dir):
        raise RuntimeError(
            "Error: Failed to open PLY folder {}".format(args.ply_dir))

    if not os.path.exists(args.dem):
        raise RuntimeError("Error: Failed to open DEM {}".format(args.dem))

    start_time = time.time()
    m = Geon()
    m.initialize(args.ply_dir, args.dem)
    m.get_geons()
    generate_model_time = time.time()
    m.geons_to_json()
    m.write_geonjson()
    write_obj_time = time.time()
    print(args.ply_dir + " completed!")
    print("generate geons time: " + str(generate_model_time - start_time))
    print("write geon json file time: " + str(write_obj_time - start_time))


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        print(e)
