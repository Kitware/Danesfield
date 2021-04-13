#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

import argparse
import json
import logging
import os
import pdal

from danesfield import gpm_decode

bpf_json = u"""
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

    if ext == '.bpf':
      print(args.input_file)

      # Read the data with PDAL
      pipeline = pdal.Pipeline(bpf_json % args.input_file)
      pipeline.validate()
      pipeline.execute()
      metadata = json.loads(pipeline.metadata)

      bundled_file = metadata['metadata']['readers.bpf'][0]['bundled_file']

      pipeline = None

    else:
      print('Unknown file extension')
      sys.exit(1)

    gpm_metadata = {
      list(el.keys())[0] : list(el.values())[0] for el in bundled_file
    }

    print(gpm_metadata.keys())

    if 'GPM_Master' in gpm_metadata:
      GPM_Master = gpm_decode.load_GPM_Master(
        gpm_metadata['GPM_Master'])
      # print(GPM_Master)

    if 'GPM_GndSpace_Direct' in gpm_metadata:
      GPM_GndSpace_Direct = gpm_decode.load_GPM_GndSpace_Direct(
        gpm_metadata['GPM_GndSpace_Direct'])
      # print(GPM_GndSpace_Direct)

    if 'Per_Point_Lookup_Error_Data' in gpm_metadata:
      Per_Point_Lookup_Error_Data = gpm_decode.load_Per_Point_Lookup_Error_Data(
        gpm_metadata['Per_Point_Lookup_Error_Data'])
      # print(Per_Point_Lookup_Error_Data)

    if 'GPM_Unmodeled_Error_Data' in gpm_metadata:
      GPM_Unmodeled_Error_Data = gpm_decode.load_GPM_Unmodeled_Error_Data(
        gpm_metadata['GPM_Unmodeled_Error_Data'])
      # print(GPM_Unmodeled_Error_Data)


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
