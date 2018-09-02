import numpy as np
import sys
import os
from osgeo import gdal
from osgeo import osr
from fractions import gcd
from rasterio.enums import ColorInterp
from scipy.ndimage import measurements


def get_extent(dataset):
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    transform = dataset.GetGeoTransform()
    minx = transform[0]
    maxx = transform[0] + cols * transform[1] + rows * transform[2]

    miny = transform[3] + cols * transform[4] + rows * transform[5]
    maxy = transform[3]

    return {
        "minX": str(minx), "maxX": str(maxx),
        "minY": str(miny), "maxY": str(maxy),
        "cols": str(cols), "rows": str(rows)
    }


def create_tiles(minx, miny, maxx, maxy, n):
    width = maxx - minx
    height = maxy - miny

    matrix = []

    for j in range(n, 0, -1):
        for i in range(0, n):

            ulx = minx + (width/n) * i  # 10/5 * 1
            uly = miny + (height/n) * j  # 10/5 * 1

            lrx = minx + (width/n) * (i + 1)
            lry = miny + (height/n) * (j - 1)
            matrix.append([[ulx, uly], [lrx, lry]])

    return matrix


def create_square_tiles(minx, miny, maxx, maxy, txlen, tylen):
    width = maxx - minx
    height = maxy - miny

    nw = int(width/txlen)
    nh = int(height/tylen)

    shift_minx = minx + int((width - nw * txlen)*0.5)
    shift_miny = miny + int((height - nh * tylen)*0.5)

    matrix = []

    for j in range(nh, 0, -1):
        for i in range(nw):

            ulx = shift_minx + txlen * i  # 10/5 * 1
            uly = shift_miny + tylen * j  # 10/5 * 1

            lrx = shift_minx + txlen * (i + 1)
            lry = shift_miny + tylen * (j - 1)
            matrix.append([[ulx, uly], [lrx, lry]])

    return matrix


def split(file_name, nbands, stype, tagname, sqwidth=2048):
    raw_file_name = os.path.splitext(os.path.basename(file_name))[
        0].replace("_downsample", "")
    driver = gdal.GetDriverByName('GTiff')
    dataset = gdal.Open(file_name)
    bands = []
    for i in range(nbands):
        bands.append(dataset.GetRasterBand(i+1))

    transform = dataset.GetGeoTransform()

    extent = get_extent(dataset)

    cols = int(extent["cols"])
    rows = int(extent["rows"])

    print("Columns: " + str(cols))
    print("Rows: " + str(rows))

    minx = float(extent["minX"])
    maxx = float(extent["maxX"])
    miny = float(extent["minY"])
    maxy = float(extent["maxY"])

    width = maxx - minx
    height = maxy - miny

    output_path = os.path.join("data", raw_file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("GCD, " + str(gcd(round(width, 0), round(height, 0))))
    print("Width, " + str(width))
    print("height, " + str(height))

    transform = dataset.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    txlen = sqwidth*pixelWidth
    tylen = sqwidth*pixelHeight

    # tiles = create_tiles(minx, miny, maxx, maxy, 15)
    tiles = create_square_tiles(minx, miny, maxx, maxy, txlen, tylen)

    print(str(xOrigin) + ', ' + str(yOrigin))

    tile_num = 0
    for tile in tiles:
        print('tile: {}'.format(tile))

        minx = tile[0][0]
        maxx = tile[1][0]
        miny = tile[1][1]
        maxy = tile[0][1]

        # p1 = (minx, maxy)
        # p2 = (maxx, miny)

        i1 = int((minx - xOrigin) / pixelWidth)
        j1 = int((yOrigin - maxy) / pixelHeight)

        i2 = int((maxx - xOrigin) / pixelWidth)
        print('oldx: ' + str(i1) + ', ' + str(i2) + ' -- ' + str(i2-i1))
        while i2-i1 != sqwidth:
            if i2-i1 > sqwidth:
                maxx -= 0.5*pixelWidth
            else:
                maxx += 0.5*pixelWidth

            i2 = int((maxx - xOrigin) / pixelWidth)
            print('nowx: ' + str(i1) + ', ' + str(i2) + ' -- ' + str(i2-i1))

        j2 = int((yOrigin - miny) / pixelHeight)
        print('oldy: ' + str(j1) + ', ' + str(j2) + ' -- ' + str(j2-j1))
        while j2-j1 != sqwidth:
            if j2-j1 > sqwidth:
                miny += 0.5*pixelHeight
            else:
                miny -= 0.5*pixelHeight

            j2 = int((yOrigin - miny) / pixelHeight)
            print('nowy: ' + str(j1) + ', ' + str(j2) + ' -- ' + str(j2-j1))

        print(str(i1) + ', ' + str(i2) + ' -- ' + str(i2-i1))
        print(str(j1) + ', ' + str(j2) + ' -- ' + str(j2-j1))

        new_cols = i2-i1
        new_rows = j2-j1

        mydata = []
        for i in range(nbands):
            mydata.append(bands[i].ReadAsArray(i1, j1, new_cols, new_rows))

        # print data

        new_x = xOrigin + i1*pixelWidth
        new_y = yOrigin - j1*pixelHeight

        print(str(new_x) + ', ' + str(new_y))

        new_transform = (
            new_x, transform[1], transform[2], new_y, transform[4], transform[5])

        output_file_base = "Tile_" + str(tile_num) + '_' + tagname + ".tif"
        output_file = os.path.join("data", raw_file_name, output_file_base)

        gdaltype = gdal.GDT_Byte
        if stype == 'UInt16':
            gdaltype = gdal.GDT_UInt16
        elif stype == 'UInt32':
            gdaltype = gdal.GDT_UInt32
        elif stype == 'Float32':
            gdaltype = gdal.GDT_Float32
        elif stype == 'Float64':
            gdaltype = gdal.GDT_Float64

        dst_ds = driver.Create(output_file,
                               new_cols,
                               new_rows,
                               nbands,
                               gdaltype)

        # writting output raster
        for i in range(nbands):
            if tagname == 'GTI':
                mydata[i], num = measurements.label(mydata[i])
                print('#(building) = ' + str(num))

            if tagname == 'GTL':
                gtldata = np.zeros(mydata[i].shape) + 2
                for xi in range(gtldata.shape[0]):
                    for yi in range(gtldata.shape[1]):
                        if mydata[i][xi, yi] > 0:
                            gtldata[xi, yi] = 6

                mydata[i] = gtldata

            dst_ds.GetRasterBand(i+1).WriteArray(mydata[i])

        if nbands == 3:
            dst_ds.colorinterp = [ColorInterp.red,
                                  ColorInterp.green, ColorInterp.blue]
        else:
            dst_ds.colorinterp = [ColorInterp.grey]

        tif_metadata = {
            "minX": str(minx), "maxX": str(maxx),
            "minY": str(miny), "maxY": str(maxy)
        }
        dst_ds.SetMetadata(tif_metadata)

        # setting extension of output raster
        # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
        dst_ds.SetGeoTransform(new_transform)

        wkt = dataset.GetProjection()

        # setting spatial reference of output raster
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)
        dst_ds.SetProjection(srs.ExportToWkt())

        # Close output raster dataset
        dst_ds = None

        tile_num += 1

    dataset = None


if __name__ == "__main__":
    split(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
