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
from danesfield.ndvi import compute_ndvi


def save_ndvi(ndvi, msi_file, filename):
    """
    Save an NDVI image using the same metadata as the given MSI file
    """
    gdal_save(ndvi, msi_file, filename, gdal.GDT_Float32)


def save_ndsm(ndsm, dsm_file, filename):
    """
    Save a normalized DSM image using the same metadata as the source DSM
    """
    ndsm_file = gdal_save(ndsm, dsm_file, filename, gdal.GDT_Float32)
    no_data_val = dsm_file.GetRasterBand(1).GetNoDataValue()
    ndsm_file.GetRasterBand(1).SetNoDataValue(no_data_val)


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
        description='Segment buildings by comparing a DSM to a DTM')
    parser.add_argument("source_dsm",
                        help="Digital surface model (DSM) image file name")
    parser.add_argument("source_dtm",
                        help="Digital terrain model (DTM) image file name")
    parser.add_argument("--msi",
                        help="Optional MSI image from which to compute NDVI "
                             "for vegetation removal")
    parser.add_argument("--ndsm",
                        help="Write out the normalized DSM image")
    parser.add_argument("--ndvi",
                        help="Write out the Normalized Difference Vegetation "
                             "Index image")
    parser.add_argument('--road-vector-shapefile-dir',
                        help='Path to road vector shapefile directory')
    parser.add_argument('--road-vector-shapefile-prefix', help='Prefix for road vector shapefile')
    parser.add_argument('--road-vector', help='Path to road vector file')
    # XXX this is not ideal
    parser.add_argument('--road-rasterized', help='Path to save rasterized road image')
    parser.add_argument('--road-rasterized-bridge', help='Path to save rasterized bridge image')
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug output and visualization")
    parser.add_argument("destination_mask",
                        help="Building mask output image")
    args = parser.parse_args(args)

    # For now assume the input DSM and DTM are in the same resolution,
    # aligned, and in the same coordinates.  Later we can warp the DTM
    # to the DSM, if needed.

    # open the DSM
    dsm_file = gdal_open(args.source_dsm)
    dsm_band = dsm_file.GetRasterBand(1)
    dsm = dsm_band.ReadAsArray()
    dsm_nodata_value = dsm_band.GetNoDataValue()
    print("DSM raster shape {}".format(dsm.shape))

    # open the DTM
    dtm_file = gdal_open(args.source_dtm)
    dtm_band = dtm_file.GetRasterBand(1)
    dtm = dtm_band.ReadAsArray()
    print("DTM raster shape {}".format(dtm.shape))

    # Compute the normalized DSM by subtracting the terrain
    ndsm = dsm - dtm

    # consider any point above 2m as possible buildings
    mask = ndsm > 2
    # Use any point above 4m as a high confidence seed point
    seeds = ndsm > 4

    # if requested, write out the normalized DSM
    if args.ndsm:
        ndsm[dsm == dsm_nodata_value] = dsm_nodata_value
        save_ndsm(ndsm, dsm_file, args.ndsm)

    # if an MSI images was specified, use it to filter by NDVI
    if args.msi:
        msi_file = gdal_open(args.msi)
        # Compute normalized difference vegetation index (NDVI)
        ndvi = compute_ndvi(msi_file)
        # if requested, write out the NDVI image
        if args.ndvi:
            save_ndvi(ndvi, msi_file, args.ndvi)
        # remove building candidates with high vegetation likelihood
        mask[ndvi > 0.2] = False
        # reduce seeds to areas with high confidence non-vegetation
        seeds[ndvi > 0.1] = False

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
            input_road_vector, dsm_file, args.road_rasterized,
            numpy.ones((3, 3)), dilation_iterations=20,
        )
        road_bridges = rasterize_file_dilated_line(
            input_road_vector, dsm_file, args.road_rasterized_bridge,
            numpy.ones((3, 3)), dilation_iterations=20,
            query=ELEVATED_ROADS_QUERY,
        )

        # Remove building candidates that overlap with a road
        mask[roads] = False
        seeds[roads] = False

    # use morphology to clean up the mask
    mask = morphology.binary_opening(mask, numpy.ones((3, 3)), iterations=1)
    mask = morphology.binary_closing(mask, numpy.ones((3, 3)), iterations=1)
    # use morphology to clean up the seeds
    seeds = morphology.binary_opening(seeds, numpy.ones((3, 3)), iterations=1)
    seeds = morphology.binary_closing(seeds, numpy.ones((3, 3)), iterations=1)

    # compute connected components on the seeds
    label_img = ndm.label(seeds)[0]
    # compute the size of each connected component
    labels, counts = numpy.unique(label_img, return_counts=True)
    # filter seed connected components to keep only large areas
    to_remove = numpy.extract(counts < 500, labels)
    print("Removing {} small connected components".format(len(to_remove)))
    seeds[numpy.isin(label_img, to_remove)] = False

    # visualize initial seeds if in debug mode
    if args.debug:
        cv2.imshow('seeds', mask.astype(numpy.uint8)*127 + seeds.astype(numpy.uint8)*127)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # label the larger mask image
    label_img = ndm.label(mask)[0]
    # extract the unique labels that match the seeds
    selected = numpy.unique(numpy.extract(seeds, label_img))
    # filter out very oblong objects
    subselected = []
    for i in selected:
        dim_large, dim_small = estimate_object_scale(label_img == i)
        if dim_large / dim_small < 6:
            subselected.append(i)

    print("Keeping {} connected components".format(len(subselected)))

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
    print("Create destination mask of size:({}, {}) ..."
          .format(dsm_file.RasterXSize, dsm_file.RasterYSize))
    gdal_save(cls, dsm_file, args.destination_mask, gdal.GDT_Byte,
              options=['COMPRESS=DEFLATE'])


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
