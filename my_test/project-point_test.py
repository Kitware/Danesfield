import gdal
import math
import numpy
import rpc
import sys

if (len(sys.argv) < 3):
    print("{} <source_image> <destination_image>".format(sys.argv[0]))
    exit(1)

imageFileName = sys.argv[1]
destImageFileName = sys.argv[2]

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
    print((sourceImage.RasterYSize, sourceImage.RasterXSize))
    raster = numpy.zeros((sourceImage.RasterYSize, sourceImage.RasterXSize), dtype=numpy.uint8)
else:
    print("Driver {} does not supports CreateCopy() method.".format(fileformat))
    exit(0)

# Write the image
print("Write destination image ...")
#destImage.GetRasterBand(1).WriteArray(raster)
destImage.FlushCache()
# close the gdal files
destImage = None
sourceImage = None
