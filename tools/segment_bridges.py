#!/usr/bin/env python

import argparse
import logging
import os

import cv2
import gdal
import numpy
import numpy.linalg
import scipy.ndimage.measurements as ndm
import scipy.ndimage.morphology as morphology

from danesfield.gdal_utils import gdal_open, gdal_save
from danesfield.rasterize import ELEVATED_ROADS_QUERY, rasterize_file_dilated_line


def estimate_object_scale(img):
    """
    Given a binary (boolean) image, return a pair estimating the large
    and small dimension of the object in img.  We currently use PCA
    for this purpose.
    """
    points = numpy.transpose(img.nonzero())
    points = points - points.mean(0)
    s = numpy.linalg.svd(points, compute_uv=False) / len(points) ** 0.5
    # If the object was a perfect rectangle, this would calculate the
    # lengths of its axes.  (For an ellipse the value is 1/16)
    VARIANCE_RATIO = 1 / 12
    return s / VARIANCE_RATIO ** 0.5


def main(args):
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description='Use road and bridge labels from OSM to augment CLS')
    parser.add_argument("source_cls",
                        help="Source Building mask (CLS) image file name")
    parser.add_argument('--road-vector-shapefile-dir',
                        help='Path to road vector shapefile directory')
    parser.add_argument('--road-vector-shapefile-prefix', help='Prefix for road vector shapefile')
    parser.add_argument('--road-vector', help='Path to road vector file')
    # XXX this is not ideal
    parser.add_argument('--road-rasterized', help='Path to save rasterized road image')
    parser.add_argument('--road-rasterized-bridge', help='Path to save rasterized bridge image')
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug output and visualization")
    parser.add_argument("destination_cls",
                        help="Destination Building/Bridge mask (CLS) image file name")
    args = parser.parse_args(args)

    # For now assume the input DSM and DTM are in the same resolution,
    # aligned, and in the same coordinates.  Later we can warp the DTM
    # to the DSM, if needed.

    # open the CLS
    cls_file = gdal_open(args.source_cls)
    cls_band = cls_file.GetRasterBand(1)
    cls = cls_band.ReadAsArray()
    logging.debug("CLS raster shape {}".format(cls.shape))

    # Extract building labels as a binary mask
    mask = (cls == 6)

    use_roads = (args.road_vector or
                 args.road_vector_shapefile_dir or
                 args.road_vector_shapefile_prefix or
                 args.road_rasterized or
                 args.road_rasterized_bridge)
    if use_roads:
        use_shapefile_dir_with_prefix = (args.road_vector_shapefile_dir and
                                         args.road_vector_shapefile_prefix)
        if not ((args.road_vector or
                 use_shapefile_dir_with_prefix) and
                args.road_rasterized and
                args.road_rasterized_bridge):
            raise RuntimeError("All road path arguments must be provided if any is provided")

        if args.road_vector and use_shapefile_dir_with_prefix:
            raise RuntimeError("Should specify EITHER --road-vector OR \
both --road-vector-shapefile-dir AND --road-vector-shapefile-prefix")

        if use_shapefile_dir_with_prefix:
            input_road_vector = os.path.join(
                args.road_vector_shapefile_dir,
                "{}.shx".format(args.road_vector_shapefile_prefix))
        else:
            input_road_vector = args.road_vector

        # The dilation is intended to create semi-realistic widths
        roads = rasterize_file_dilated_line(
            input_road_vector, cls_file, args.road_rasterized,
            numpy.ones((3, 3)), dilation_iterations=20,
        )
        road_bridges = rasterize_file_dilated_line(
            input_road_vector, cls_file, args.road_rasterized_bridge,
            numpy.ones((3, 3)), dilation_iterations=20,
            query=ELEVATED_ROADS_QUERY,
        )

        # Remove building candidates that overlap with a road
        mask[roads] = False
        road_bridges = morphology.binary_closing(road_bridges, numpy.ones((3, 3)), iterations=3)

    # label the larger mask image
    label_img = ndm.label(mask)[0]
    # extract the unique labels that match the seeds
    selected, counts = numpy.unique(label_img, return_counts=True)
    # filter out very oblong objects
    subselected = []
    for i, n in zip(selected, counts):
        # skip the background and small components
        if i == 0 or n < 64:
            continue
        dim_large, dim_small = estimate_object_scale(label_img == i)
        if dim_small > 0 and dim_large / dim_small < 6:
            subselected.append(i)

    logging.info("Keeping {} connected components".format(len(subselected)))

    # keep only the mask components selected above
    good_mask = numpy.isin(label_img, subselected)

    # a final mask cleanup
    good_mask = morphology.binary_closing(good_mask, numpy.ones((3, 3)), iterations=3)

    # visualize final mask if in debug mode
    if args.debug:
        cv2.imshow('mask', good_mask.astype(numpy.uint8)*255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # convert the mask to label map with value 2 (background),
    # 6 (building), and 17 (elevated roadway)
    cls = numpy.full(good_mask.shape, 2)
    cls[good_mask] = 6
    if use_roads:
        cls[road_bridges] = 17

    # create the mask image
    logging.info("Create destination mask of size:({}, {}) ..."
                 .format(cls_file.RasterXSize, cls_file.RasterYSize))
    gdal_save(cls, cls_file, args.destination_cls, gdal.GDT_Byte,
              options=['COMPRESS=DEFLATE'])


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
