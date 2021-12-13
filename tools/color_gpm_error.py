#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

'''
Create a new version of a point cloud that colorizes the GPM error
'''

import argparse
import json
import numpy as np
import pdal
import sys

from danesfield.gpm import GPM

from pathlib import Path

def main(args):
    parser = argparse.ArgumentParser(
        description="Colorize the GPM error of a point cloud.")
    parser.add_argument("input_file", help="path to input file")
    parser.add_argument("output_file", help="path to output file")
    args = parser.parse_args(args)

    pdal_input = {
        'pipeline': [args.input_file]
    }

    pipeline = pdal.Pipeline(json.dumps(pdal_input))
    pipeline.validate()
    pipeline.execute()

    metadata = json.loads(pipeline.metadata)
    arr = pipeline.arrays[0]

    dec_arr = arr[::10]

    points = np.stack([dec_arr['X'], dec_arr['Y'], dec_arr['Z']], axis=1)

    gpm = GPM(metadata['metadata'])

    if 'PPE_LUT_Index' in dec_arr.dtype.names:
        ppe = gpm.get_per_point_error(points,
                                      dec_arr['PPE_LUT_Index'].astype(np.int32))
        ppe_error = ppe.diagonal(axis1=1, axis2=2)
        print('SHAPE ', ppe_error.shape)

    error = gpm.get_covar(points).diagonal(axis1=1, axis2=2)

    if 'PPE_LUT_Index' in dec_arr.dtype.names:
        total_error = ppe_error[:,2] + error[:,2]
    else:
        total_error = error[:,2]

    max_error = np.max(total_error)
    min_error = np.min(total_error)
    error_norm = max_error - min_error

    print('ERROR: ', min_error, max_error)

    if error_norm == 0.0:
        exit()

    to_color = lambda err : (2**15-1)*(err-min_error)/error_norm + 2**15

    dec_arr['Red'] = to_color(error[:,2]).astype(np.uint16)
    # dec_arr['Green'] = to_color(error[:,2]).astype(np.uint16)
    dec_arr['Blue'] = 255 - to_color(error[:,2]).astype(np.uint16)

    pdal_output = {
        'pipeline': [
            {
                "type": "writers.las",
                "filename": args.output_file
            }
        ]
    }

    pipeline = pdal.Pipeline(json.dumps(pdal_output), [dec_arr])
    pipeline.validate()
    pipeline.execute()

    pipeline = None

if __name__ == '__main__':
    main(sys.argv[1:])
