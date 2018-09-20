import gdal
import gdalnumeric
import numpy
import pyproj
import re
import ogr
import osr


def gdal_bounding_box(raster, outProj=None):
    """
    Computes the bounding box for an open GDAL raster file.

    The format is [minX, minY, maxX, maxY] in outProj coordinates.
    For instance outProj for lat/lon is pyproj.Proj('+proj=longlat +datum=WGS84')
    Returns None in case of an error.
    """
    projection = raster.GetProjection()
    if (projection):
        transform = raster.GetGeoTransform()
    else:
        projection = raster.GetGCPProjection()
        gcps = raster.GetGCPs()
        transform = gdal.GCPsToGeoTransform(gcps)
        if transform is None:
            print("Unable to extract a geotransform from GCPs")
            return None
    lines = numpy.array([0, 0, raster.RasterYSize, raster.RasterYSize])
    pixels = numpy.array([0, raster.RasterXSize, raster.RasterXSize, 0])
    arrayX = transform[0] + pixels * transform[1] + lines * transform[2]
    arrayY = transform[3] + pixels * transform[4] + lines * transform[5]

    if outProj:
        srs = osr.SpatialReference(wkt=projection)
        proj_srs = srs.ExportToProj4()
        inProj = pyproj.Proj(proj_srs)
        arrayX, arrayY = pyproj.transform(inProj, outProj, arrayX, arrayY)

    minX = numpy.amin(arrayX)
    minY = numpy.amin(arrayY)
    maxX = numpy.amax(arrayX)
    maxY = numpy.amax(arrayY)
    return [minX, minY, maxX, maxY]


def gdal_open(filename, access=gdal.GA_ReadOnly):
    """
    Like gdal.Open, but always read-only and raises an OSError instead
    of returning None
    """
    rv = gdal.Open(filename, access)
    if rv is None:
        raise OSError("Unable to open {!r}".format(filename))
    return rv


def gdal_save(arr, src_file, filename, eType, options=[]):
    """
    Save the 2D ndarray arr to filename using the same metadata as the
    given source file.  Returns the new gdal file object in case
    additional operations are desired.
    """
    if isinstance(arr, list):
        numberOfBands = len(arr)
    else:
        numberOfBands = 1
        arr = [arr]
    driver = src_file.GetDriver()
    if driver.GetMetadata().get(gdal.DCAP_CREATE) != "YES":
        raise RuntimeError("Driver {} does not support Create().".format(driver))
    arr_file = driver.Create(
        filename, xsize=arr[0].shape[1], ysize=arr[0].shape[0],
        bands=numberOfBands, eType=eType, options=options,
    )
    gdalnumeric.CopyDatasetInfo(src_file, arr_file)
    for i, a in enumerate(arr):
        arr_file.GetRasterBand(i + 1).WriteArray(a)
    return arr_file


def ogr_open(filename, update=0):
    """
    Like ogr.Open, but raises an OSError instead
    of returning None
    """
    rv = ogr.Open(filename, update)
    if rv is None:
        raise OSError("Unable to open {!r}".format(filename))
    return rv


def ogr_get_layer(vectorFile, geometryType):
    """
    Returns the layer with geometry type matching 'layerGeometryType'
    from 'vectorFile'
    """
    layerCount = vectorFile.GetLayerCount()
    for i in range(layerCount):
        layer = vectorFile.GetLayerByIndex(i)
        type = layer.GetGeomType()
        if (type == geometryType):
            break
    if i == layerCount:
        raise RuntimeError("No layer with type {} found".format(geometryType))
    return layer


def read_offset(fileName, offset):
    ''' Read an offset X,Y,Z written as comment lines in a file.
        Offsets have to be in the first 3 lines of the file. A line that does
        not match will stop matching.
    '''
    offset[0] = 0.0
    offset[1] = 0.0
    offset[2] = 0.0
    axes = ['x', 'y', 'z']
    reFloatList = list("#. offset: ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)")
    with open(fileName) as f:
        for i in range(3):
            reFloatList[1] = axes[i]
            reFloat = re.compile("".join(reFloatList))
            line = f.readline()
            match = reFloat.search(line)
            if match:
                offset[i] = float(match.group(1))
            else:
                break
