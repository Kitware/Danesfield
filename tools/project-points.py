import gdal
import numpy
import pdal
from danesfield import rpc
import sys

if (len(sys.argv) < 4):
    print("{} <source_image> <source_points> <destination_image>".format(sys.argv[0]))
    sys.exit(1)

imageFileName = sys.argv[1]
pointsFileName = sys.argv[2]
destImageFileName = sys.argv[3]

# open the GDAL file
sourceImage = gdal.Open(imageFileName, gdal.GA_ReadOnly)
rpcMetaData = sourceImage.GetMetadata('RPC')
driver = sourceImage.GetDriver()
driverMetadata = driver.GetMetadata()
destImage = None
if driverMetadata.get(gdal.DCAP_CREATECOPY) == "YES":
    print("Copy source to destination image ...")
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

print("Project points to destination image ...")

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
model = rpc.rpc_from_gdal_dict(rpcMetaData)
# project points to get image indexes and save their height into the image
for i in range(0, len(arrayX)):
    point = [arrayX[i], arrayY[i], arrayZ[i]]
    # z is uint16
    quantizedZ = arrayZ[i] * 60000 / (maxZ - minZ)
    rpcPoint = model.project(point)
    if (raster[int(rpcPoint[0]), int(rpcPoint[1])] < int(quantizedZ)):
        raster[int(rpcPoint[0]), int(rpcPoint[1])] = quantizedZ

# Write the image
print("Write destination image ...")
destImage.GetRasterBand(1).WriteArray(raster)

# close the gdal files
destImage = None
sourceImage = None
