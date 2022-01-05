#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

'''
Load a point cloud and generate the various error for several random points
in the point cloud. The results are stored.
'''

import argparse
import json
import numpy as np
import pdal
import sys
import time

from danesfield.gpm import GPM

from pathlib import Path

def main(args):
    parser = argparse.ArgumentParser(
        description="Get the error from the GPM data for random points "
                    "and write the results to a json file.")
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

    points = np.stack([arr['X'], arr['Y'], arr['Z']], axis=1)

    # Get n random points in the point cloud
    rnd_idx = np.random.randint(0, points.shape[0], 5)
    test_points = points[rnd_idx, :]

    gpm = GPM(metadata['metadata'])

    # Get the PPE indices if they are present
    if 'PPE_LUT_Index' in arr.dtype.names:
        gpm.setupPPELookup(points, arr['PPE_LUT_Index'].astype(np.int32))

    error = gpm.get_covar(test_points)
    if 'PPE_LUT_Index' in arr.dtype.names:
        ppe_error = gpm.get_per_point_error(test_points)
    um_error = gpm.get_unmodeled_error(test_points)

    test_results = {}
    for i in range(len(rnd_idx)):
        idx = str(rnd_idx[i])
        test_results[idx] = {
            'point': test_points[i].tolist(),
            'covar': error[i].tolist(),
            'um_error': um_error[i].tolist(),
        }
        if 'PPE_LUT_Index' in arr.dtype.names:
            test_results[idx]['ppe_error'] = ppe_error[i].tolist()

    with open(args.output_file, 'w') as f:
        json.dump(test_results, f)

if __name__ == '__main__':
    main(sys.argv[1:])
