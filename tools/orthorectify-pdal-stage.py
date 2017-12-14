from danesfield import rpc

import gdal
import numpy

def store_gray(ins,outs):
    print("Load source image ...")

    NODATA_VALUE = 10000
    # open the GDAL file
    sourceImage = gdal.Open(pdalargs['source_image'], gdal.GA_ReadOnly)
    if not sourceImage:
        exit(1)
    band = sourceImage.GetRasterBand(1)
    raster = band.ReadAsArray(
        xoff=0, yoff=0,
        win_xsize=sourceImage.RasterXSize, win_ysize=sourceImage.RasterYSize)
    print("Raster shape {}".format(raster.shape))
    # read the RPC from RPC Metadata in the image file
    print("Reading RPC Metadata from {}".format(pdalargs['source_image']))
    rpcMetaData = sourceImage.GetMetadata('RPC')
    model = rpc.rpc_from_gdal_dict(rpcMetaData)

    # project the points
    arrayX = ins['X']
    arrayY = ins['Y']
    arrayZ = ins['Z']

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

    print("Copying point colors ...")
    intImgPoints = intImgPoints[:, validIdx]
    outs['gray'] = ins['gray']
    outs['gray'].fill(NODATA_VALUE)
    outs['gray'][validIdx] = raster[intImgPoints[1], intImgPoints[0]]

    # close files
    print("Close files ...")
    sourceImage = None
    return True
