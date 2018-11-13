#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################


import argparse
import logging
from gaia.geo.geo_inputs import VectorFileIO, FeatureIO
from gaia.geo.processes_vector import IntersectsProcess
from osgeo import gdal, osr


def GetExtent(gt, cols, rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float, ..., float]}
        @return:   coordinates of each corner
    '''
    ext = []
    xarr = [0, cols]
    yarr = [0, rows]

    for px in xarr:
        for py in yarr:
            x = gt[0]+(px*gt[1])+(py*gt[2])
            y = gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x, y])
        yarr.reverse()
    return ext


def ReprojectCoords(coords, src_srs, tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords = []
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    for x, y in coords:
        x, y, z = transform.TransformPoint(x, y)
        trans_coords.append([x, y])
    return trans_coords


def main(args):
    parser = argparse.ArgumentParser(
        description="Write a destination file with all features from source file "
                    "overlapping specified polygon")
    parser.add_argument("source", help="Source vector file name that contain all features")
    parser.add_argument("-p", "--polygon", nargs="+",
                        help="xmin ymin xmax ymax or vector file specifying the polygon")
    parser.add_argument("destination",
                        help="Destination vector file with only features "
                             "overlapping the specified polygon")
    args = parser.parse_args(args)

    if (len(args.polygon) == 4):
        xmin, ymin, xmax, ymax = [float(val) for val in args.polygon]
        polygon = FeatureIO(features=[
            {"geometry":
                {"type": "Polygon",
                 "coordinates":
                    [
                        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                    ]},
             "properties": {"id": "Bounding box"}}, ])
    elif (len(args.polygon) == 1):
        polygonImage = gdal.Open(args.polygon[0], gdal.GA_ReadOnly)
        if (polygonImage):
            gt = polygonImage.GetGeoTransform()
            cols = polygonImage.RasterXSize
            rows = polygonImage.RasterYSize
            ext = GetExtent(gt, cols, rows)

            src_srs = osr.SpatialReference()
            src_srs.ImportFromWkt(polygonImage.GetProjection())
            tgt_srs = src_srs.CloneGeogCS()
            p = ReprojectCoords(ext, src_srs, tgt_srs)
            polygon = FeatureIO(features=[
                {"geometry":
                    {"type": "Polygon",
                     "coordinates":
                     [
                        [[p[0][0], p[0][1]], [p[1][0], p[1][1]],
                         [p[2][0], p[2][1]], [p[3][0], p[3][1]]]
                     ]},
                 "properties": {"id": "Bounding box"}}, ])
        else:
            polygon = VectorFileIO(uri=args.polygon[0])
    else:
        raise RuntimeError("Error: wrong number of parameters for polygon: {} "
                           "(can be 4 or 1)".format(len(args.polygon)))

    source = VectorFileIO(uri=args.source)
    destination = VectorFileIO(uri=args.destination)

    intersectProcess = IntersectsProcess(inputs=[source, polygon], output=destination)
    intersectProcess.compute()


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
