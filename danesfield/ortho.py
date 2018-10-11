from danesfield import rpc
from danesfield import raytheon_rpc

import gdal
import numpy
import osr
import pyproj
from scipy.ndimage import morphology


def circ_structure(n):
    """generate a circular binary mask of radius n for morphology
    """
    nf = numpy.floor(n)
    a = numpy.arange(-nf, nf+1)
    x, y = numpy.meshgrid(a, a)
    return (x**2 + y**2) <= n**2


COMPLETE_DSM_INTERSECTION = 0
PARTIAL_DSM_INTERSECTION = 1
EMPTY_DSM_INTERSECTION = 2
ERROR = 10


def orthorectify(args_source_image, args_dsm, args_destination_image,
                 args_occlusion_thresh=1.0, args_denoise_radius=2,
                 args_raytheon_rpc=None, args_dtm=None):
    """
    Orthorectify an image given the DSM

    Args:
        source_image: Source image file name
        dsm: Digital surface model (DSM) image file name
        destination_image: Orthorectified image file name
        occlusion-thresh: Threshold on height difference for detecting
                          and masking occluded regions (in meters)
        denoise-radius: Apply morphological operations with this radius
                        to the DSM reduce speckled noise
        raytheon-rpc: Raytheon RPC file name. If not provided
                      the RPC is read from the source_image

    Returns:
        COMPLETE_DSM_INTERSECTION = 0
        PARTIAL_DSM_INTERSECTION = 1
        EMPTY_DSM_INTERSECTION = 2
        ERROR = 10
    """
    returnValue = COMPLETE_DSM_INTERSECTION
    # open the source image
    sourceImage = gdal.Open(args_source_image, gdal.GA_ReadOnly)
    if not sourceImage:
        return ERROR
    sourceBand = sourceImage.GetRasterBand(1)

    if (args_raytheon_rpc):
        # read the RPC from raytheon file
        print("Reading RPC from Raytheon file: {}".format(args_raytheon_rpc))
        model = raytheon_rpc.read_raytheon_rpc_file(args_raytheon_rpc)
    else:
        # read the RPC from RPC Metadata in the image file
        print("Reading RPC Metadata from {}".format(args_source_image))
        rpcMetaData = sourceImage.GetMetadata('RPC')
        model = rpc.rpc_from_gdal_dict(rpcMetaData)
    if model is None:
        print("Error reading the RPC")
        return ERROR

    # open the DSM
    dsm = gdal.Open(args_dsm, gdal.GA_ReadOnly)
    if not dsm:
        return ERROR
    band = dsm.GetRasterBand(1)
    dsmRaster = band.ReadAsArray(
        xoff=0, yoff=0,
        win_xsize=dsm.RasterXSize, win_ysize=dsm.RasterYSize)
    dsm_nodata_value = band.GetNoDataValue()
    print("DSM raster shape {}".format(dsmRaster.shape))

    if args_dtm:
        dtm = gdal.Open(args_dtm, gdal.GA_ReadOnly)
        if not dtm:
            return ERROR
        band = dtm.GetRasterBand(1)
        dtmRaster = band.ReadAsArray(
            xoff=0, yoff=0,
            win_xsize=dtm.RasterXSize, win_ysize=dtm.RasterYSize)
        newRaster = numpy.where(dsmRaster != dsm_nodata_value, dsmRaster, dtmRaster)
        dsmRaster = newRaster

    # apply morphology to denoise the DSM
    if (args_denoise_radius > 0):
        morph_struct = circ_structure(args_denoise_radius)
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
        options = ["COMPRESS=DEFLATE"]
        # ensure that space will be reserved for geographic corner coordinates
        # (in DMS) to be set later
        if (driver.ShortName == "NITF" and not projection):
            options.append("ICORDS=G")
        # If I try to use AddBand with GTiff I get:
        # Dataset does not support the AddBand() method.
        # So I create all bands using the same type at the begining
        destImage = driver.Create(
            args_destination_image, xsize=dsm.RasterXSize,
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
            validIdx = arrayZ != dsm_nodata_value
            pixels = pixels[validIdx]
            lines = lines[validIdx]
            arrayX = arrayX[validIdx]
            arrayY = arrayY[validIdx]
            arrayZ = arrayZ[validIdx]

        else:
            # georeference through GCPs
            destImage.SetGCPs(gcps, gcpProjection)
            # not implemented: compute arrayX, arrayY, arrayZ
            print("Not implemented yet")
            return ERROR
    else:
        print("Driver {} does not supports Create().".format(driver))
        return ERROR

    # convert coordinates to Long/Lat
    srs = osr.SpatialReference(wkt=projection)
    proj_srs = srs.ExportToProj4()
    inProj = pyproj.Proj(proj_srs)
    outProj = pyproj.Proj('+proj=longlat +datum=WGS84')
    arrayX, arrayY = pyproj.transform(inProj, outProj, arrayX, arrayY)

    # Sort the points by height so that higher points project last
    if (args_occlusion_thresh > 0):
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

    # coumpute the bound of the relevant AOI in the source image
    print("Source Image size: ", [sourceImage.RasterXSize, sourceImage.RasterYSize])
    minPoint = numpy.maximum([0, 0], numpy.min(intImgPoints, 1))
    print("AOI min: ", minPoint)
    maxPoint = numpy.minimum(numpy.max(intImgPoints, 1),
                             [sourceImage.RasterXSize,
                              sourceImage.RasterYSize])
    print("AOI max: ", maxPoint)
    cropSize = maxPoint - minPoint
    if numpy.any(cropSize < 1):
        print("DSM does not intersect source image")
        returnValue = EMPTY_DSM_INTERSECTION

    # shift the projected image point to the cropped AOI space
    intImgPoints[0] -= minPoint[0]
    intImgPoints[1] -= minPoint[1]

    # find indicies of points that fall inside the image bounds
    print("Source raster shape {}".format(cropSize))
    validIdx = numpy.logical_and.reduce((intImgPoints[1] < cropSize[1],
                                         intImgPoints[1] >= 0,
                                         intImgPoints[0] < cropSize[0],
                                         intImgPoints[0] >= 0))
    intImgPoints = intImgPoints[:, validIdx]

    # keep only the points that are in the image
    numOut = numpy.size(validIdx) - numpy.count_nonzero(validIdx)
    if (numOut > 0 and not returnValue == EMPTY_DSM_INTERSECTION):
        print("Skipped {} points outside of image".format(numOut))
        returnValue = PARTIAL_DSM_INTERSECTION

    # use a height map to test for occlusion
    if (args_occlusion_thresh > 0):
        print("Mapping occluded points")
        valid_arrayZ = arrayZ[validIdx]
        # render a height map in the source image space
        height_map = numpy.full(cropSize[::-1], -numpy.inf, dtype=numpy.float32)
        height_map[intImgPoints[1], intImgPoints[0]] = valid_arrayZ

        # get a mask of points that locally are (approximately)
        # the highest point in the map
        is_max_height = height_map[intImgPoints[1], intImgPoints[0]] \
            <= valid_arrayZ + args_occlusion_thresh
        num_occluded = numpy.size(is_max_height) - numpy.count_nonzero(is_max_height)
        print("Skipped {} occluded points".format(num_occluded))

        # keep only non-occluded image points
        intImgPoints = intImgPoints[:, is_max_height]
        # disable occluded points in the valid pixel mask
        validIdx[numpy.nonzero(validIdx)[0][numpy.logical_not(is_max_height)]] = False

    for bandIndex in range(1, sourceImage.RasterCount + 1):
        print("Processing band {} ...".format(bandIndex))
        sourceBand = sourceImage.GetRasterBand(bandIndex)
        nodata_value = sourceBand.GetNoDataValue()
        # for now use zero as a no-data value if one is not specified
        # it would probably be better to add a mask (alpha) band instead
        if nodata_value is None:
            nodata_value = 0
        if numpy.any(cropSize < 1):
            # read one value for data type
            sourceRaster = sourceBand.ReadAsArray(
                xoff=0, yoff=0, win_xsize=1, win_ysize=1)
            destRaster = numpy.full(
                (dsm.RasterYSize, dsm.RasterXSize), nodata_value,
                dtype=sourceRaster.dtype)
        else:
            sourceRaster = sourceBand.ReadAsArray(
                xoff=int(minPoint[0]), yoff=int(minPoint[1]),
                win_xsize=int(cropSize[0]), win_ysize=int(cropSize[1]))

            print("Copying colors ...")
            destRaster = numpy.full(
                (dsm.RasterYSize, dsm.RasterXSize), nodata_value,
                dtype=sourceRaster.dtype)
            destRaster[lines[validIdx], pixels[validIdx]] = sourceRaster[
                intImgPoints[1], intImgPoints[0]]

        print("Write band ...")
        destBand = destImage.GetRasterBand(bandIndex)
        destBand.SetNoDataValue(nodata_value)
        destBand.WriteArray(destRaster)
    return returnValue
