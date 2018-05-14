import gdal
import numpy
import pyproj
import osr

def bounding_box(raster):
    """
    Computes the bounding box for an open GDAL raster file.

    The format is [minX, minY, maxX, maxY] in lat/lon coordinates.
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

    srs = osr.SpatialReference(wkt=projection)
    proj_srs = srs.ExportToProj4()
    inProj = pyproj.Proj(proj_srs)
    outProj = pyproj.Proj('+proj=longlat +datum=WGS84')
    arrayX, arrayY = pyproj.transform(inProj, outProj, arrayX, arrayY)

    minX = numpy.amin(arrayX)
    minY = numpy.amin(arrayY)
    maxX = numpy.amax(arrayX)
    maxY = numpy.amax(arrayY)
    return [minX, minY, maxX, maxY]

