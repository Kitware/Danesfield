from danesfield import rpc

import argparse
import gdal
import osr
import pyproj
import numpy
from scipy.ndimage import morphology
import sys

parser = argparse.ArgumentParser(
    description='Orthorectify an image given the DSM')
parser.add_argument("source_image", help="Source image file name")
parser.add_argument("dsm", help="Digital surface model (DSM) image file name")
parser.add_argument("destination_image", help="Orthorectified image file name")
parser.add_argument('-t', "--occlusion-thresh", type=float, default=1.0,
                    help="Threshold on height difference for detecting "
                    "and masking occluded regions (in meters)")
parser.add_argument('-d', "--denoise-radius", type=float, default=2,
                    help="Apply morphological operations with this radius "
                    "to the DSM reduce speckled noise")
args = parser.parse_args()

def circ_structure(n):
    """generate a circular binary mask of radius n for morphology
    """
    nf = numpy.floor(n)
    a = numpy.arange(-nf, nf+1)
    x, y = numpy.meshgrid(a, a)
    return (x**2 + y**2) <= n**2

# open the source image
sourceImage = gdal.Open(args.source_image, gdal.GA_ReadOnly)
if not sourceImage:
    exit(1)
sourceBand = sourceImage.GetRasterBand(1)
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

# apply morphology to denoise the DSM
if (args.denoise_radius > 0):
    morph_struct = circ_structure(args.denoise_radius)
    dsmRaster = morphology.grey_opening(dsmRaster, structure=morph_struct)
    dsmRaster = morphology.grey_closing(dsmRaster, structure=morph_struct)

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
    # If I try to use AddBand with GTiff I get:
    # Dataset does not support the AddBand() method.
    # So I create all bands using the same type at the begining
    destImage = driver.Create(
        args.destination_image, xsize=dsm.RasterXSize,
        ysize=dsm.RasterYSize,
        bands=sourceImage.RasterCount, eType=sourceBand.DataType,
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
else:
    print("Driver {} does not supports Create().".format(driver))
    sys.exit(1)


# convert coordinates to Long/Lat
srs = osr.SpatialReference(wkt=projection)
proj_srs = srs.ExportToProj4()
inProj = pyproj.Proj(proj_srs)
outProj = pyproj.Proj('+proj=longlat +datum=WGS84')
arrayX, arrayY = pyproj.transform(inProj, outProj, arrayX, arrayY)

# Sort the points by height so that higher points project last
if (args.occlusion_thresh > 0):
    print("Sorting by Height")
    heightIdx = numpy.argsort(arrayZ)
    arrayX = arrayX[heightIdx]
    arrayY = arrayY[heightIdx]
    arrayZ = arrayZ[heightIdx]
    lines = lines[heightIdx]
    pixels = pixels[heightIdx]

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
sourceRaster = sourceBand.ReadAsArray(
    xoff=0, yoff=0,
    win_xsize=sourceImage.RasterXSize, win_ysize=sourceImage.RasterYSize)
print("Source raster shape {}".format(sourceRaster.shape))
validIdx = numpy.logical_and.reduce((intImgPoints[1] < sourceRaster.shape[0],
                                     intImgPoints[1] >= 0,
                                     intImgPoints[0] < sourceRaster.shape[1],
                                     intImgPoints[0] >= 0))
intImgPoints = intImgPoints[:, validIdx]

# keep only the points that are in the image
numOut = numpy.size(validIdx) - numpy.count_nonzero(validIdx)
if (numOut > 0):
    print("Skipped {} points outside of image".format(numOut))

# use a height map to test for occlusion
if (args.occlusion_thresh > 0):
    print("Mapping occluded points")
    valid_arrayZ = arrayZ[validIdx]
    # render a height map in the source image space
    height_map = numpy.full(sourceRaster.shape, -numpy.inf, dtype=numpy.float32)
    height_map[intImgPoints[1], intImgPoints[0]] = valid_arrayZ

    # get a mask of points that locally are (approximately)
    # the highest point in the map
    is_max_height = height_map[intImgPoints[1], intImgPoints[0]] \
                    <= valid_arrayZ + args.occlusion_thresh
    num_occluded = numpy.size(is_max_height) - numpy.count_nonzero(is_max_height)
    print("Skipped {} occluded points".format(num_occluded))

    # keep only non-occluded image points
    intImgPoints = intImgPoints[:, is_max_height]
    # disable occluded points in the valid pixel mask
    validIdx[numpy.nonzero(validIdx)[0][numpy.logical_not(is_max_height)]] = False

for bandIndex in range(1, sourceImage.RasterCount + 1):
    print("Processing band {} ...".format(bandIndex))
    if bandIndex > 1:
        sourceBand = sourceImage.GetRasterBand(bandIndex)
        sourceRaster = sourceBand.ReadAsArray(
            xoff=0, yoff=0, win_xsize=sourceImage.RasterXSize,
            win_ysize=sourceImage.RasterYSize)

    print("Copying colors ...")
    nodata_value = sourceBand.GetNoDataValue()
    destRaster = numpy.full(
        (dsm.RasterYSize, dsm.RasterXSize), nodata_value,
        dtype=sourceRaster.dtype)
    destRaster[lines[validIdx], pixels[validIdx]] = sourceRaster[
        intImgPoints[1], intImgPoints[0]]

    print("Write band ...")
    destBand = destImage.GetRasterBand(bandIndex)
    destBand.SetNoDataValue(nodata_value)
    destBand.WriteArray(destRaster)
