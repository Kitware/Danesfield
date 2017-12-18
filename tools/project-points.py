from danesfield import rpc
from danesfield import raytheon_rpc

import argparse
import gdal
import numpy
import pdal
import sys

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
    "--create", action='store_true',
    help="Create an new image with a band height of type float."
    " By default, this script copies the source image so "
    "the height is quantized to int")
args = parser.parse_args()
MAX_VALUE = 65535

# open the GDAL file
sourceImage = gdal.Open(args.source_image, gdal.GA_ReadOnly)
if not sourceImage:
    exit(1)
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
if args.create:
    if driverMetadata.get(gdal.DCAP_CREATE) == "YES":
        print("Create destination image (height is float), "
              "size:({}, {}) ...".format(sourceImage.RasterXSize,
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
        destImage = driver.Create(
            args.destination_image, xsize=sourceImage.RasterXSize,
            ysize=sourceImage.RasterYSize, bands=1, eType=gdal.GDT_Float32,
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
            dtype=numpy.float32)
    else:
        print("Driver {} does not supports Create().".format(driver))
        sys.exit(1)
else:
    if driverMetadata.get(gdal.DCAP_CREATECOPY) == "YES":
        print("Copy source to destination image (height is int16), "
              "size:({}, {}) ...".format(sourceImage.RasterXSize,
                                         sourceImage.RasterYSize))
        destImage = driver.CreateCopy(
            args.destination_image, sourceImage, strict=0)
        # create an emtpy image
        raster = numpy.zeros(
            (sourceImage.RasterYSize, sourceImage.RasterXSize),
            dtype=numpy.uint16)
    else:
        print(
            "Driver {} does not supports CreateCopy().".format(driver))
        sys.exit(1)

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
count = pipeline.execute()
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
if (args.create):
    arrayZ = arrayZ[validIdx]
    print("Rendering Image")
    raster[intImgPoints[1], intImgPoints[0]] = arrayZ
else:
    quantizedZ = ((arrayZ - minZ) * MAX_VALUE / (maxZ - minZ)).astype(
        numpy.int)
    quantizedZ = quantizedZ[validIdx]
    print("Rendering Image")
    raster[intImgPoints[1], intImgPoints[0]] = quantizedZ

# Write the image
print("Write destination image ...")
destImage.GetRasterBand(1).WriteArray(raster)

# close files
print("Close files ...")
destImage = None
sourceImage = None
