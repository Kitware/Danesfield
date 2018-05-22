#!/usr/bin/env python

from danesfield import rpc
from danesfield import raytheon_rpc

import argparse
import gdal
import logging
import numpy
import pdal
import sys


def main(args):
    parser = argparse.ArgumentParser(
        description='Projects source_points onto a source_image'
        ' using an RPC projection read from source_image or from raytheon_rpc')
    parser.add_argument("source_image", help="Source image file name")
    parser.add_argument("source_points", help="Source points file name")
    parser.add_argument("destination_image", help="Destination image file name")
    parser.add_argument("--raytheon-rpc", type=str,
                        help="Raytheon RPC file name. If not provided, "
                        "the RPC is read from the source_image")
    parser.add_argument(
        "--type", choices=["uint8", "uint16", "float32"],
        help="Specify the type for the height band, default is float32.")
    args = parser.parse_args(args)

    # open the GDAL file
    sourceImage = gdal.Open(args.source_image, gdal.GA_ReadOnly)
    if not sourceImage:
        raise RuntimeError("Error: Failed to open source image {}".format(args.source_image))
    model = None
    if (args.raytheon_rpc):
        # read the RPC from raytheon file
        print("Reading RPC from Raytheon file: {}".format(args.raytheon_rpc))
        model = raytheon_rpc.read_raytheon_rpc_file(args.raytheon_rpc)
    else:
        # read the RPC from RPC Metadata in the image file
        print("Reading RPC Metadata from {}".format(args.source_image))
        rpcMetaData = sourceImage.GetMetadata('RPC')
        model = rpc.rpc_from_gdal_dict(rpcMetaData)

    driver = sourceImage.GetDriver()
    driverMetadata = driver.GetMetadata()
    destImage = None
    if driverMetadata.get(gdal.DCAP_CREATE) == "YES":
        print("Create destination image (height is {}), "
              "size:({}, {}) ...".format("float32" if not args.type else args.type,
                                         sourceImage.RasterXSize,
                                         sourceImage.RasterYSize))
        # georeference information
        projection = sourceImage.GetProjection()
        transform = sourceImage.GetGeoTransform()
        gcpProjection = sourceImage.GetGCPProjection()
        gcps = sourceImage.GetGCPs()
        options = []
        # ensure that space will be reserved for geographic corner coordinates
        # (in DMS) to be set later
        if (driver.ShortName == "NITF" and not projection):
            options.append("ICORDS=G")
        if (args.type == "uint8"):
            eType = gdal.GDT_Byte
            dtype = numpy.uint8
            MAX_VALUE = 255
        elif (args.type == "uint16"):
            eType = gdal.GDT_UInt16
            dtype = numpy.uint16
            MAX_VALUE = 65535
        else:
            eType = gdal.GDT_Float32
            dtype = numpy.float32
        destImage = driver.Create(
            args.destination_image, xsize=sourceImage.RasterXSize,
            ysize=sourceImage.RasterYSize, bands=1, eType=eType,
            options=options)
        if (projection):
            # georeference through affine geotransform
            destImage.SetProjection(projection)
            destImage.SetGeoTransform(transform)
        else:
            # georeference through GCPs
            destImage.SetGCPs(gcps, gcpProjection)
        raster = numpy.zeros(
            (sourceImage.RasterYSize, sourceImage.RasterXSize),
            dtype=dtype)
    else:
        raise RuntimeError("Error: driver {} does not supports Create().".format(driver))

    # read the pdal file and project the points
    json = u"""
    {
      "pipeline": [
        "%s",
        {
            "type":"filters.reprojection",
            "out_srs":"EPSG:4326"
        }
      ]
    }"""
    print("Loading Point Cloud")
    json = json % args.source_points
    pipeline = pdal.Pipeline(json)
    pipeline.validate()  # check if our JSON and options were good
    # this causes a segfault at the end of the program
    # pipeline.loglevel = 8  # really noisy
    pipeline.execute()
    arrays = pipeline.arrays
    arrayX = arrays[0]['X']
    arrayY = arrays[0]['Y']
    arrayZ = arrays[0]['Z']
    pipeline = None

    # Sort the points by height so that higher points project last
    print("Sorting by Height")
    heightIdx = numpy.argsort(arrayZ)
    arrayX = arrayX[heightIdx]
    arrayY = arrayY[heightIdx]
    arrayZ = arrayZ[heightIdx]

    minZ = numpy.amin(arrayZ)
    maxZ = numpy.amax(arrayZ)
    # project points to get image indexes and save their height into the image
    print("Project {} points to destination image ...".format(len(arrayX)))
    print("Points min/max Z: {}/{}  ...".format(minZ, maxZ))

    print("Projecting Points")
    imgPoints = model.project(numpy.array([arrayX, arrayY, arrayZ]).transpose())
    intImgPoints = imgPoints.astype(numpy.int).transpose()

    # find indicies of points that fall inside the image bounds
    validIdx = numpy.logical_and.reduce((intImgPoints[1] < raster.shape[0],
                                         intImgPoints[1] >= 0,
                                         intImgPoints[0] < raster.shape[1],
                                         intImgPoints[0] >= 0))

    # keep only the points that are in the image
    numOut = numpy.size(validIdx) - numpy.count_nonzero(validIdx)
    if (numOut > 0):
        print("Skipped {} points outside of image".format(numOut))
    intImgPoints = intImgPoints[:, validIdx]
    if (args.type == "uint16" or args.type == "uint8"):
        quantizedZ = ((arrayZ - minZ) * MAX_VALUE / (maxZ - minZ)).astype(
            numpy.int)
        quantizedZ = quantizedZ[validIdx]
        print("Rendering Image")
        raster[intImgPoints[1], intImgPoints[0]] = quantizedZ
    else:
        arrayZ = arrayZ[validIdx]
        print("Rendering Image")
        raster[intImgPoints[1], intImgPoints[0]] = arrayZ

    # Write the image
    print("Write destination image ...")
    destImage.GetRasterBand(1).WriteArray(raster)

    # close files
    print("Close files ...")
    destImage = None
    sourceImage = None


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
