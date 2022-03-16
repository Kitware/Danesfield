#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

'''
Generate a color (RGB) raster from a colored point cloud (LAS).
'''

import argparse
import json
import logging
import numpy
import subprocess


def getMinMax(json_string):
    j = json.loads(json_string)
    j = j['stats']['statistic']
    minX = j[0]['minimum']
    maxX = j[0]['maximum']
    minY = j[1]['minimum']
    maxY = j[1]['maximum']
    return minX, maxX, minY, maxY


def main(argv):
    print(f'RGB argv={argv}')
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--source_points', nargs='+', help='source points file[s]')
    parser.add_argument('output', help='path/to/output/folder for RGB.tif')
    parser.add_argument('--bounds', nargs=4, type=float, action='store',
                        help='Destination image bounds (using the coordinate system '
                             'of the source_points file): minX, maxX, minY, maxY. '
                             'If not specified, it is computed from source_points files')
    parser.add_argument('--gsd', type=float, default=0.25, help='ground sample distance')
    args = parser.parse_args(argv)

    if not args.source_points:
        raise RuntimeError('Error: At least one source_points file required')
    if args.bounds:
        minX, maxX, minY, maxY = args.bounds
    else:
        cnt = len(args.source_points)
        print(f'Computing the bounding box for {cnt} point cloud files ...')
        minX = numpy.inf
        maxX = -numpy.inf
        minY = numpy.inf
        maxY = -numpy.inf
        pdal_info_template = ['pdal', 'info', '--stats', '--dimensions', 'X,Y']
        for i, s in enumerate(args.source_points):
            pdal_info_args = pdal_info_template + [s]
            out = subprocess.check_output(pdal_info_args)
            tempMinX, tempMaxX, tempMinY, tempMaxY = getMinMax(out)
            if tempMinX < minX:
                minX = tempMinX
            if tempMaxX > maxX:
                maxX = tempMaxX
            if tempMinY < minY:
                minY = tempMinY
            if tempMaxY > maxY:
                maxY = tempMaxY
            if i % 10 == 0:
                print('Iteration {}'.format(i))
    print('Bounds ({}, {}, {}, {})'.format(minX, maxX, minY, maxY))

    # compensate for PDAL expanding the extents by 1 pixel
    GSD = float(args.gsd)
    maxX -= GSD
    maxY -= GSD

    # read the pdal file and project the points
    print('Generating RGB ...')
    all_sources = ',\n'.join('"' + str(e) + '"' for e in args.source_points)
    jsonTemplate = '''
    {
      "pipeline": [
        %s,
        {
          "type": "filters.crop",
          "bounds": "([%s, %s], [%s, %s])"
        },
        {
          "type": "writers.gdal",
          "resolution": %s,
          "dimension": "%s",
          "data_type": "float",
          "output_type": "mean",
          "window_size": "20",
          "bounds": "([%s, %s], [%s, %s])",
          "filename": "%s/%s.tif",
          "gdalopts": "COMPRESS=DEFLATE"
        }
      ]
    }'''
    def genBand(band, FN):
        pipeline = jsonTemplate % (all_sources,
                                   minX, maxX, minY, maxY,
                                   args.gsd, band,
                                   minX, maxX, minY, maxY,
                                   args.output, FN
                                   )
        logging.info(pipeline)
        pdal_pipeline_args = ['pdal', 'pipeline', '--stream', '--stdin']
        response = subprocess.run(pdal_pipeline_args, input=pipeline.encode(),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if response.returncode != 0:
            print('STDERR')
            print(response.stderr)
            print('STDOUT')
            print(response.stdout)
            raise RuntimeError(f'PDAL failed with error code {response.returncode}')

    genBand('Red', 'R')
    genBand('Green', 'G')
    genBand('Blue', 'B')


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
