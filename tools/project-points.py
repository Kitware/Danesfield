import gdal
import numpy
import pdal
from danesfield import rpc
from danesfield import raytheon_rpc
import sys

if (len(sys.argv) < 4):
    print("{} <source_image> <source_points> <destination_image> [<raytheon_rpc>]".format(sys.argv[0]))
    sys.exit(1)

imageFileName = sys.argv[1]
pointsFileName = sys.argv[2]
destImageFileName = sys.argv[3]
MAX_VALUE = 65535

# open the GDAL file
sourceImage = gdal.Open(imageFileName, gdal.GA_ReadOnly)
model = None
if (len(sys.argv) == 5):
    # read the RPC from raytheon file
    rpcFileName = sys.argv[4]
    model = raytheon_rpc.read_raytheon_rpc_file(rpcFileName)
    print("Using RPC from Raytheon file: {}".format(rpcFileName))
else:
    # read the RPC from RPC Metadata in the image file
    print("Using RPC Metadata from {}".format(imageFileName))
    rpcMetaData = sourceImage.GetMetadata('RPC')
    model = rpc.rpc_from_gdal_dict(rpcMetaData)

driver = sourceImage.GetDriver()
driverMetadata = driver.GetMetadata()
destImage = None
if driverMetadata.get(gdal.DCAP_CREATECOPY) == "YES":
    print("Copy source to destination image, size:({}, {}) ...".format(
        sourceImage.RasterXSize, sourceImage.RasterYSize))
    destImage = driver.CreateCopy(destImageFileName, sourceImage, strict=0)
    # create an emtpy image
    raster = numpy.zeros((sourceImage.RasterYSize, sourceImage.RasterXSize), dtype=numpy.uint16)
else:
    print("Driver {} does not supports CreateCopy() method.".format(driver))
    sys.exit(0)

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
json = json % pointsFileName
pipeline = pdal.Pipeline(json)
pipeline.validate()  # check if our JSON and options were good
pipeline.loglevel = 8  # really noisy
count = pipeline.execute()
arrays = pipeline.arrays
arrayX = arrays[0]['X']
arrayY = arrays[0]['Y']
arrayZ = arrays[0]['Z']
minZ = numpy.amin(arrayZ)
maxZ = numpy.amax(arrayZ)
# project points to get image indexes and save their height into the image
print("Project {} points to destination image ...".format(len(arrayX)))
print("Points min/max Z: {}/{}  ...".format(minZ, maxZ))
underPoint = 0
outPoint =  0
for i in range(0, len(arrayX)):
    point = [arrayX[i], arrayY[i], arrayZ[i]]
    # z is uint16
    quantizedZ = int((arrayZ[i] - minZ) * MAX_VALUE / (maxZ - minZ))
    rpcPoint = model.project(point)
    intRpcPoint = [int(rpcPoint[1]), int(rpcPoint[0])]
    if (intRpcPoint[0] < raster.shape[0] and intRpcPoint[0] >= 0 and
        intRpcPoint[1] < raster.shape[1] and intRpcPoint[1] >= 0):
        if (raster[intRpcPoint[0], intRpcPoint[1]] < quantizedZ):
            raster[intRpcPoint[0], intRpcPoint[1]] = quantizedZ
        else:
            underPoint += 1
    else:
        if (outPoint < 10):
            print("outside point {}, image_coords {}".format(point, rpcPoint))
        outPoint += 1

if (underPoint > 0):
    print("Skipped {} points of lower Z value".format(underPoint))
if (outPoint > 0):
    print("Skipped {} points outside image".format(outPoint))

# Write the image
print("Write destination image ...")
destImage.GetRasterBand(1).WriteArray(raster)

# close the gdal files
destImage = None
sourceImage = None
