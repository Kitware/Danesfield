#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

'''
Compute visible Normalized Difference Vegetation Index (vNDVI)
using RGB colored point cloud, basically following the approach in
Costa et al. A new visible band index (vNDVI) for estimating NDVI values on RGB images utilizing genetic algorithms. CEA 2020
'''

import argparse, logging, gdal
from danesfield.gdal_utils import gdal_open, gdal_save
from danesfield.ndvi import compute_ndvi


def vNDVI(pcd, img): # return visible Normalized Difference Vegetation Index (vNDVI)
    # TODO use pcd color values for RGB bands
    band = img.GetRasterBand(1)
    red = band.ReadAsArray()
    mask = red != band.GetNoDataValue()
    green = blue = red
    C = 0.5268
    rp = -0.1294
    gp = 0.3389
    bp = -0.3118
    return C * red**rp * green**gp * blue**bp * mask


def main(args):
    # Configure argument parser
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('image', help='path/to/DSM.tif')
    parser.add_argument('input', help='path/to/colored/point/cloud.las')
    parser.add_argument('output', help='path/to/NDVI.tif')
    args = parser.parse_args(args)
    
    img = gdal_open(args.image)
    vndvi = vNDVI(args.input, img)
    gdal_save(vndvi, img, args.output,
              gdal.GDT_Float32,
              options=['COMPRESS=DEFLATE'])


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
