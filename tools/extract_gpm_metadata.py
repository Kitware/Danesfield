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
[
  "%s"
]"""

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
    pipeline = pdal.Pipeline(pdal_json % args.input_file)
    pipeline.validate()
    pipeline.execute()
    metadata = json.loads(pipeline.metadata)

    if ext == '.bpf':
      bundled_file = metadata['metadata']['readers.bpf'][0]['bundled_file']

      gpm_metadata = {
        list(el.keys())[0] : list(el.values())[0] for el in bundled_file
      }

      pipeline = None

    elif ext == '.las':
      # Read the data with PDAL
      pipeline = pdal.Pipeline(pdal_json % args.input_file)
      pipeline.validate()
      pipeline.execute()
      metadata = json.loads(pipeline.metadata)

      las_metadata = metadata['metadata']['readers.las'][0]

      gpm_metadata = {}

      for k in las_metadata:
        if 'vlr' in k:
          if type(las_metadata[k]) is dict:
            if ('GPM' in las_metadata[k]['description'] or
                'Per_Point_Lookup_Error_Data' in las_metadata[k]['description']):
              gpm_metadata[las_metadata[k]['description']] = las_metadata[k]['data']

      pipeline = None

    else:
      print('Unknown file extension')
      sys.exit(1)

    gpm = GPM(gpm_metadata)

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
