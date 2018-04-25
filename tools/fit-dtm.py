
import argparse
import gdal
import gdalnumeric
import numpy
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
    args = parser.parse_args()

    # open the DSM
    dsm = gdal.Open(args.source_dsm, gdal.GA_ReadOnly)
    if not dsm:
        print("Unable to open {}".format(args.source_dsm))
        exit(1)
    band = dsm.GetRasterBand(1)
    dsmRaster = band.ReadAsArray(
        xoff=0, yoff=0,
        win_xsize=dsm.RasterXSize, win_ysize=dsm.RasterYSize)
    print("DSM raster shape {}".format(dsmRaster.shape))

    dtm = dsmRaster

    # create the DTM image
    driver = dsm.GetDriver()
    driverMetadata = driver.GetMetadata()
    destImage = None
    if driverMetadata.get(gdal.DCAP_CREATE) == "YES":
        print("Create destination DTM of "
              "size:({}, {}) ...".format(dsm.RasterXSize, dsm.RasterYSize))
        destImage = driver.Create(
            args.destination_dtm, xsize=dtm.shape[1],
            ysize=dtm.shape[0],
            bands=dsm.RasterCount, eType=band.DataType)

        gdalnumeric.CopyDatasetInfo(dsm, destImage)
    else:
        print("Driver {} does not supports Create().".format(driver))
        sys.exit(1)

    estimator = danesfield.dtm.DTMEstimator(band.GetNoDataValue(), args.num_iterations)
    dtm = estimator.fit_dtm(dtm)

    destBand = destImage.GetRasterBand(1)
    destBand.SetNoDataValue(band.GetNoDataValue())
    destBand.WriteArray(dtm)


if __name__ == '__main__':
    main(sys.argv[1:])
