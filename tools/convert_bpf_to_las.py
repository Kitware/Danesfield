#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

'''
Convert bpf file to las while maintaining metadata. Has option to crop the
point cloud
'''

import argparse
import json
import pdal
import sys

from pathlib import Path

def main(args):
    parser = argparse.ArgumentParser(
        description="Convert bpf file to las. Has option to crop the point cloud.")
    parser.add_argument("input_file", help="path to input bpf file")
    parser.add_argument("--output_file", help="path to output las file. Defaults"
                        " to input path with .las extension.")
    parser.add_argument("--bounds", nargs="*", help="optional bounds for "
                        "cropping the point cloud. The bounds should be ordered "
                        "as follows: xmin xmax ymin ymax zmin zmax")
    args = parser.parse_args(args)

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = Path(args.input_file).with_suffix(".las")

    # Construct pdal pipline
    pdal_json = {
        'pipeline':
        [
            {
                "type": "readers.bpf",
                "filename": args.input_file
            },
            {
                "type" : "writers.las",
                "pdal_metadata":"true",
                "filename" : str(output_file)
            }
        ]
    }

    if args.bounds:
        bnds_str = ""
        for mn, mx in zip(args.bounds[::2],args.bounds[1::2]):
            bnds_str = bnds_str + "[{},{}],".format(mn, mx)
        pdal_json['pipeline'].insert(1,
            {
                "type":"filters.crop",
                "bounds": "({})".format(bnds_str[:-1])
            }
        )

    pipeline = pdal.Pipeline(json.dumps(pdal_json))
    pipeline.validate()
    pipeline.execute()
    pipeline = None

if __name__ == '__main__':
    main(sys.argv[1:])
