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
from danesfield.gpm_decode64 import json_numpy_array_hook
from pathlib import Path
from scipy import stats

def main(args):
    parser = argparse.ArgumentParser(
        description="Colorize the GPM error of a point cloud.")
    parser.add_argument("input_file", help="path to input file")
    parser.add_argument("output_file", help="path to output file")
    parser.add_argument("--metadata_file", help="path for file with"
                        " GPM metadata.")
    parser.add_argument("--included_errors", help="comma separated booleans"
                        " for the three error types: standard, ppe, and"
                        " unmodeled. For example '1,0,1' would include the"
                        " standard and unmodeled errors")
    parser.add_argument("--decimation", help="Only use every nth point.",
                        type=int, default=10)
    args = parser.parse_args(args)

    pdal_input = {
        'pipeline': [args.input_file]
    }

    pipeline = pdal.Pipeline(json.dumps(pdal_input))
    pipeline.validate()
    pipeline.execute()

    metadata = json.loads(pipeline.metadata)
    arr = pipeline.arrays[0]

    dec_arr = arr[::args.decimation]

    points = np.stack([dec_arr['X'], dec_arr['Y'], dec_arr['Z']], axis=1)

    if args.metadata_file:
        with open(args.metadata_file) as mf:
            gpm = GPM(json.load(mf, object_hook=json_numpy_array_hook))
    else:
        gpm = GPM(metadata['metadata'])

    if 'PPE_LUT_Index' in dec_arr.dtype.names:
        gpm.setupPPELookup(points, dec_arr['PPE_LUT_Index'])

    error = np.zeros(points.shape)

    # Determine which error to include
    if args.included_errors:
        inc_err = [(True if int(e) == 1 else False) for
                   e in args.included_errors.split(',')]
    else:
        inc_err = [True, False, False]

    if inc_err[0]:
        error += gpm.get_covar(points).diagonal(axis1=1, axis2=2)
    if inc_err[1] and 'PPE_LUT_Index' in dec_arr.dtype.names:
        error += gpm.get_per_point_error(points).diagonal(axis1=1, axis2=2)
    if inc_err[2]:
        error += gpm.get_unmodeled_error(points).diagonal(axis1=1, axis2=2)

    error_stats = stats.describe(error)
    print(error_stats)
    min_error = error_stats.minmax[0]
    error_norm = error_stats.minmax[1] - error_stats.minmax[0]

    colors = (2**15-1)*(error-min_error)/error_norm + 2**15


    dec_arr['Red'] = colors[:,0].astype(np.uint16)
    dec_arr['Green'] = colors[:,1].astype(np.uint16)
    dec_arr['Blue'] = colors[:,2].astype(np.uint16)

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
