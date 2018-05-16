#!/usr/bin/env python

import argparse
import logging

import cv2
import gdal
import gdalnumeric
import numpy
import scipy.ndimage.measurements as ndm
import scipy.ndimage.morphology as morphology


def compute_ndvi(msi_file):
    """
    Compute a normalized difference vegetation index (NVDI) image from an MSI file
    """
    num_bands = msi_file.RasterCount
    # Guess band indices based on the number of bands
    if num_bands == 8:
        # for 8-band MSI from WV3 and WV2 the RGB bands have these indices
        red_idx = 5
        nir_idx = 7
    elif num_bands == 4:
        # assume the bands are B,G,R,N (where N is near infrared)
        red_idx = 3
        nir_idx = 4
    else:
        raise RuntimeError("Unknown Red/NIR channels in {}-band image".format(num_bands))

    red_band = msi_file.GetRasterBand(red_idx)
    red = red_band.ReadAsArray()
    red_mask = red != red_band.GetNoDataValue()

    nir_band = msi_file.GetRasterBand(nir_idx)
    nir = nir_band.ReadAsArray()
    nir_mask = nir != nir_band.GetNoDataValue()

    mask = numpy.logical_and(red_mask, nir_mask)

    red = red.astype(numpy.float)
    nir = nir.astype(numpy.float)

    return numpy.divide(nir - red, nir + red, where=mask)


def save_gdal(arr, src_file, filename, eType, options=[]):
    """
    Save the 2D ndarray arr to filename using the same metadata as the
    given source file.  Returns the new gdal file object in case
    additional operations are desired.
    """
    driver = src_file.GetDriver()
    if driver.GetMetadata().get(gdal.DCAP_CREATE) != "YES":
        raise RuntimeError("Driver {} does not support Create().".format(driver))
    arr_file = driver.Create(
        filename, xsize=arr.shape[1], ysize=arr.shape[0],
        bands=1, eType=eType, options=options,
    )
    gdalnumeric.CopyDatasetInfo(src_file, arr_file)
    arr_file.GetRasterBand(1).WriteArray(arr)
    return arr_file


def save_ndvi(ndvi, msi_file, filename):
    """
    Save an NDVI image using the same metadata as the given MSI file
    """
    save_gdal(ndvi, msi_file, filename, gdal.GDT_Float32)


def save_ndsm(ndsm, dsm_file, filename):
    """
    Save a normalized DSM image using the same metadata as the source DSM
    """
    ndsm_file = save_gdal(ndsm, dsm_file, filename, gdal.GDT_Float32)
    no_data_val = dsm_file.GetRasterBand(1).GetNoDataValue()
    ndsm_file.GetRasterBand(1).SetNoDataValue(no_data_val)


def gdal_open(filename):
    """
    Like gdal.Open, but always read-only and raises an OSError instead
    of returning None
    """
    rv = gdal.Open(filename, gdal.GA_ReadOnly)
    if rv is None:
        raise OSError("Unable to open {!r}".format(filename))
    return rv


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
    print("Keeping {} connected components".format(len(selected)))

    # keep only the mask components selected above
    good_mask = numpy.isin(label_img, selected)

    # a final mask cleanup
    good_mask = morphology.binary_closing(good_mask, numpy.ones((3, 3)), iterations=3)

    # visualize final mask if in debug mode
    if args.debug:
        cv2.imshow('mask', good_mask.astype(numpy.uint8)*255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # convert the mask to label map with value 2 (background) and 6 (building)
    cls = numpy.full(good_mask.shape, 2)
    cls[good_mask] = 6

    # create the mask image
    print("Create destination mask of size:({}, {}) ..."
          .format(dsm_file.RasterXSize, dsm_file.RasterYSize))
    save_gdal(cls, dsm_file, args.destination_mask, gdal.GDT_Byte,
              options=['COMPRESS=DEFLATE'])


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
