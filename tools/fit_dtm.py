#!/usr/bin/env python

import argparse
import gdal
import gdalnumeric
import logging
import sys

import danesfield.dtm


def main(args):
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description='Fit a DTM to a DSM')
    parser.add_argument("source_dsm",
                        help="Digital surface model (DSM) image file name")
    parser.add_argument("destination_dtm",
                        help="Digital terrain model (DTM) image file name")
    parser.add_argument('-n', "--num-iterations", type=int, default=100,
                        help="Base number of iteration at the coarsest scale")
    parser.add_argument('-t', "--tension", type=int, default=10,
                        help="Number of inner smoothing iterations, "
                             "greater values increase surface tension.")
    args = parser.parse_args(args)

    # open the DSM
    dsm = gdal.Open(args.source_dsm, gdal.GA_ReadOnly)
    if not dsm:
        print("Unable to open {}".format(args.source_dsm))
        sys.exit(1)
    band = dsm.GetRasterBand(1)
    dsmRaster = band.ReadAsArray(
        xoff=0, yoff=0,
        win_xsize=dsm.RasterXSize, win_ysize=dsm.RasterYSize)
    print("DSM raster shape {}".format(dsmRaster.shape))

    # Estimate the DTM data from the DSM data
    estimator = danesfield.dtm.DTMEstimator(band.GetNoDataValue(),
                                            args.num_iterations,
                                            args.tension)
    dtm = estimator.fit_dtm(dsmRaster)

    # create the DTM image
    driver = dsm.GetDriver()
    driverMetadata = driver.GetMetadata()
    if driverMetadata.get(gdal.DCAP_CREATE) != "YES":
        print("Driver {} does not supports Create().".format(driver))
        sys.exit(1)

    print("Create destination DTM of "
          "size:({}, {}) ...".format(dsm.RasterXSize, dsm.RasterYSize))
    options = ["COMPRESS=DEFLATE", "PREDICTOR=3"]
    destImage = driver.Create(
        args.destination_dtm, xsize=dtm.shape[1],
        ysize=dtm.shape[0],
        bands=dsm.RasterCount, eType=band.DataType,
        options=options)

    print("Copying metadata")
    gdalnumeric.CopyDatasetInfo(dsm, destImage)

    print("Writing image data")
    destBand = destImage.GetRasterBand(1)
    destBand.WriteArray(dtm)
    del destImage

    print("Done")


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
