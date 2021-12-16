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
from pathlib import Path
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
        '--output_file',
        type=str,
        help='Output file for GPM data (json format).')
    args = parser.parse_args(args)

    input_file = Path(args.input_file)

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = Path(args.input_file).with_suffix(".json")

    if not os.path.exists(input_file):
        print(input_file, 'does not exists.')
        exit()

    # Read the data with PDAL
    pipeline = pdal.Pipeline(pdal_json.format(input_file))
    pipeline.validate()
    pipeline.execute()
    metadata = json.loads(pipeline.metadata)

    pipeline = None

    gpm = GPM(metadata['metadata'])

    if gpm.metadata:
        with open(output_file, 'w') as f:
            json.dump(gpm.metadata, f, cls=NumpyArrayEncoder)
    else:
        print('No GPM metadata found.')

if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
