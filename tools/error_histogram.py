#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

'''
Plots a histogram of the diagonal covariances for each point in a point cloud
'''

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pdal
import sys

from danesfield.gpm import GPM

from pathlib import Path

def main(args):
    parser = argparse.ArgumentParser(
        description="Creates histograms of covariances for a point cloud.")
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
        total_error = ppe_error + error
    else:
        total_error = error

    print('SHAPE: ', total_error.shape)

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1)

    ax0.hist(total_error[:,0], bins=50, color='red')
    ax1.hist(total_error[:,1], bins=50, color='green')
    ax2.hist(total_error[:,2], bins=50, color='blue')

    #plt.show()
    plt.savefig(args.output_file)

if __name__ == '__main__':
    main(sys.argv[1:])
