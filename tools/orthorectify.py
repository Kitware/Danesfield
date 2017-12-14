from danesfield import rpc

import argparse
import gdal
import numpy

parser = argparse.ArgumentParser(
    description='Orthorectify an image given the DSM')
parser.add_argument("source_image", help="Source image file name")
parser.add_argument("dsm", help="Digital elevation model (DSM) image file name")
parser.add_argument("destination_image", help="Orthorectified image file name")
args = parser.parse_args()

NODATA_VALUE = 10000

# open the source image
sourceImage = gdal.Open(args.source_image, gdal.GA_ReadOnly)
if not sourceImage:
    exit(1)
band = sourceImage.GetRasterBand(1)
sourceRaster = band.ReadAsArray(
    xoff=0, yoff=0,
    win_xsize=sourceImage.RasterXSize, win_ysize=sourceImage.RasterYSize)
print("Source raster shape {}".format(sourceRaster.shape))
# read the RPC from RPC Metadata in the image file
print("Reading RPC Metadata from {}".format(args.source_image))
rpcMetaData = sourceImage.GetMetadata('RPC')
model = rpc.rpc_from_gdal_dict(rpcMetaData)

# open the DSM
dsm = gdal.Open(args.dsm, gdal.GA_ReadOnly)
if not dsm:
    exit(1)
band = dsm.GetRasterBand(1)
dsmRaster = band.ReadAsArray(
    xoff=0, yoff=0,
    win_xsize=dsm.RasterXSize, win_ysize=dsm.RasterYSize)
print("DSM raster shape {}".format(dsmRaster.shape))

# create the rectified image
driver = dsm.GetDriver()
driverMetadata = driver.GetMetadata()
destImage = None
arrayX = None
arrayY = None
arrayZ = None
if driverMetadata.get(gdal.DCAP_CREATE) == "YES":
    print("Create destination image of "
          "size:({}, {}) ...".format(dsm.RasterXSize, dsm.RasterYSize))
    # georeference information
    projection = dsm.GetProjection()
    transform = dsm.GetGeoTransform()
    gcpProjection = dsm.GetGCPProjection()
    gcps = dsm.GetGCPs()
    options = []
    # ensure that space will be reserved for geographic corner coordinates
    # (in DMS) to be set later
    if (driver.ShortName == "NITF" and not projection):
        options.append("ICORDS=G")
    destImage = driver.Create(
        args.destination_image, xsize=dsm.RasterXSize,
        ysize=dsm.RasterYSize, bands=1, eType=gdal.GDT_UInt16,
        options=options)
    if (projection):
        # georeference through affine geotransform
        destImage.SetProjection(projection)
        destImage.SetGeoTransform(transform)
        pixels = numpy.arange(0, dsm.RasterXSize)
        pixels = numpy.tile(pixels, dsm.RasterYSize)
        lines = numpy.arange(0, dsm.RasterYSize)
        lines = numpy.repeat(lines, dsm.RasterXSize)
        arrayX = transform[0] + pixels * transform[1] + lines * transform[2]
        arrayY = transform[3] + pixels * transform[4] + lines * transform[5]
        arrayZ = dsmRaster[lines, pixels]
    else:
        # georeference through GCPs
        destImage.SetGCPs(gcps, gcpProjection)
        # not implemented: compute arrayX, arrayY, arrayZ
        print("Not implemented yet")
        sys.exit(1)
    destRaster = numpy.full(
        (dsm.RasterYSize, dsm.RasterXSize), NODATA_VALUE,
        dtype=numpy.float32)
else:
    print("Driver {} does not supports Create().".format(driver))
    sys.exit(1)

    
# project the points
minZ = numpy.amin(arrayZ)
maxZ = numpy.amax(arrayZ)
# project points to get image indexes and save their height into the image
print("Project {} points to destination image ...".format(len(arrayX)))
print("Points min/max Z: {}/{}  ...".format(minZ, maxZ))

print("Projecting Points")
imgPoints = model.project(numpy.array([arrayX, arrayY, arrayZ]).transpose())
intImgPoints = imgPoints.astype(numpy.int).transpose()

# find indicies of points that fall inside the image bounds
validIdx = numpy.logical_and.reduce((intImgPoints[1] < sourceRaster.shape[0],
                                     intImgPoints[1] >= 0,
                                     intImgPoints[0] < sourceRaster.shape[1],
                                     intImgPoints[0] >= 0))

# keep only the points that are in the image
numOut = numpy.size(validIdx) - numpy.count_nonzero(validIdx)
if (numOut > 0):
    print("Skipped {} points outside of image".format(numOut))

print("Copying point colors ...")
intImgPoints = intImgPoints[:, validIdx]
destRaster[lines[validIdx], pixels[validIdx]] = sourceRaster[intImgPoints[1], intImgPoints[0]]

print("Write destination image ...")
band = destImage.GetRasterBand(1)
band.SetNoDataValue(NODATA_VALUE)
band.WriteArray(destRaster)

# close files
print("Close files ...")
sourceImage = None
dsm = None
destImage = None
