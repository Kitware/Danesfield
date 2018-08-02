#!/usr/bin/env python

import os
import sys
import time
import argparse
from danesfield.surface.scene import Model


def main(args):
    parser = argparse.ArgumentParser(description='Generate OBJ file from PLY file.')
    parser.add_argument('-p', '--ply_dir',
                        help='PLY file folder to read', required=True)
    parser.add_argument('-d', '--dem',
                        help='DEM file name to read', required=True)
    parser.add_argument('-o', '--offset', action='store_true', default=True,
                        help='Apply an offset', required=False)
    args = parser.parse_args(args)

    if not os.path.isdir(args.ply_dir):
        raise RuntimeError("Error: Failed to open PLY folder {}".format(args.PLY_folder))

    if not os.path.exists(args.dem):
        raise RuntimeError("Error: Failed to open DEM {}".format(args.DEM_file_name))

    start_time = time.time()
    m = Model()
    m.initialize(args.ply_dir, args.dem)
    generate_model_time = time.time()
    m.write_model(args.offset)
    write_obj_time = time.time()
    m.write_surface()
    print(args.ply_dir + " completed!")
    print("generate time: " + str(generate_model_time - start_time))
    print("write obj file time: " + str(write_obj_time - start_time))


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        print(e)
        