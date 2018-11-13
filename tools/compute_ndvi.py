#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################


import argparse
import logging

import gdal

from danesfield.gdal_utils import gdal_open, gdal_save
from danesfield.ndvi import compute_ndvi


def main(args):
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description='Compute Normalized Difference Vegetation Index (NDVI)')
    parser.add_argument("source_msi",
                        help="List of registered MSI images from which to estimate NDVI",
                        nargs="+")
    parser.add_argument("output_ndvi",
                        help="File path to write out the NDVI image")
    args = parser.parse_args(args)

    # If more than one input MSI image it is assumed that they are the
    # same size and registered

    avg_ndvi = None
    first_msi = None
    for msi_file in args.source_msi:
        msi = gdal_open(msi_file)
        # Compute normalized difference vegetation index (NDVI)
        ndvi = compute_ndvi(msi)
        if avg_ndvi is None:
            avg_ndvi = ndvi
            first_msi = msi
        else:
            avg_ndvi += ndvi

    if len(args.source_msi) > 1:
        avg_ndvi /= len(args.source_msi)

    gdal_save(avg_ndvi, first_msi, args.output_ndvi, gdal.GDT_Float32,
              options=['COMPRESS=DEFLATE'])


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
