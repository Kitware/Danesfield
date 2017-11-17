# Opening the gdal file
import gdal
import math
import numpy
import pdal
import rpc
import sys

if (len(sys.argv) < 4):
    print("{} <source_image> <source_points> <destination_image>".format(sys.argv[0]))
    exit(1)

imageFileName = sys.argv[1]
pointsFileName = sys.argv[2]
destImageFileName = sys.argv[3]
sourceImage = gdal.Open(imageFileName, gdal.GA_ReadOnly)

# Getting Dataset Information
print("Driver: {}/{}".format(sourceImage.GetDriver().ShortName,
                             sourceImage.GetDriver().LongName))
print("Size is {} x {} x {}".format(sourceImage.RasterXSize,
                                    sourceImage.RasterYSize,
                                    sourceImage.RasterCount))
print("Projection is {}".format(sourceImage.GetProjection()))
geotransform = sourceImage.GetGeoTransform()
if geotransform:
    print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
    print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

rpcMetaData = sourceImage.GetMetadata('RPC')


# Fetching a Raster Band
band = sourceImage.GetRasterBand(1)
print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

min = band.GetMinimum()
max = band.GetMaximum()
if not min or not max:
    (min,max) = band.ComputeRasterMinMax(True)
print("Min={:.3f}, Max={:.3f}".format(min,max))

if band.GetOverviewCount() > 0:
    print("Band has {} overviews".format(band.GetOverviewCount()))

if band.GetRasterColorTable():
    print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))


# Test if the driver supports Create() and CreateCopy()
fileformat = "NITF"
driver = gdal.GetDriverByName(fileformat)
metadata = driver.GetMetadata()
destImage = None
if metadata.get(gdal.DCAP_CREATE) == "YES":
    print("Driver {} supports Create() method.".format(fileformat))
    # destImage = driver.Create(destImageFileName, xsize=sourceImage.RasterXSize,
    #                        ysize = sourceImage.RasterYSize, bands=1, eType=gdal.GDT_Float32)

    # raster = numpy.zeros((sourceImage.RasterYSize, sourceImage.RasterXSize), dtype=numpy.float32)
    # #raster = numpy.zeros((3000, 3000), dtype=numpy.uint16)
    # raster[0:3000,0:3000] = 25
    # raster[3000:6000,3000:6000] = 50
    # raster[6000:9000,6000:9000] = 75
    # raster[9000:12000,9000:12000] = 100
    # destImage.GetRasterBand(1).WriteArray(raster)
    # destImage = None


if metadata.get(gdal.DCAP_CREATECOPY) == "YES":
    print("Driver {} supports CreateCopy() method.".format(fileformat))
    destImage = driver.CreateCopy(destImageFileName, sourceImage, strict=0)

    raster = numpy.zeros((sourceImage.RasterYSize, sourceImage.RasterXSize), dtype=numpy.uint16)
    #raster = numpy.zeros((3000, 3000), dtype=numpy.uint16)

if (destImage is None):
    exit(0)

# read the pdal file and project the points
json = u"""
{
  "pipeline": [
    "%s",
    {
        "type":"filters.reprojection",
        "in_srs":"EPSG:32616",
        "out_srs":"EPSG:4326"
    }
  ]
}"""
json = json % pointsFileName

pipeline = pdal.Pipeline(json)
pipeline.validate() # check if our JSON and options were good
pipeline.loglevel = 8 #really noisy
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
    raster[int(rpcPoint[0]), int(rpcPoint[1])] = quantizedZ
# close the gdal files
destImage.GetRasterBand(1).WriteArray(raster)
destImage = None
sourceImage = None
