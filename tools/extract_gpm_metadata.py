#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

import argparse
import json
import logging
import numpy
import os
import pdal

from danesfield.gpm import GPM

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

pdal_json = u"""
    {{
        "pipeline":
        [
          "{}"
        ]
    }}"""

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_file',
        type=str,
        help='Input point cloud file. Can be LAS or BPF format.')
    parser.add_argument(
        'out_file',
        type=str,
        help='Output file for GPM data (json format).')
    args = parser.parse_args(args)

    ext = os.path.splitext(args.input_file)[-1].lower()

    # Read the data with PDAL
    pipeline = pdal.Pipeline(pdal_json.format(args.input_file))
    pipeline.validate()
    pipeline.execute()
    metadata = json.loads(pipeline.metadata)

    pipeline = None

    gpm = GPM(metadata['metadata'])

    if gpm.ap_search:
        print('MIN: ', gpm.ap_search.mins)
        print('MAX: ', gpm.ap_search.maxes)

    if gpm.metadata:
        with open(args.out_file, 'w') as f:
            json.dump(gpm.metadata, f, cls=NumpyArrayEncoder)

if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
