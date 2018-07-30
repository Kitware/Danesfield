#!/usr/bin/env python

import argparse
import cv2
import gdal
import logging
import numpy
import ogr
import osr
import os
import subprocess
import shutil
from danesfield import gdal_utils
from danesfield import rasterize

VECTOR_TYPES = ["buildings", "roads"]


def shift_vector(inputFeatures, outputVectorFile, outputLayerName, outProjection, offsetGeo):
    outDriver = ogr.GetDriverByName("ESRI Shapefile")
    print("Shifting vector -> {}".format(os.path.basename(outputVectorFile)))
    outVector = outDriver.CreateDataSource(outputVectorFile)
    outSrs = osr.SpatialReference(outProjection)
    # create layer
    outLayer = outVector.CreateLayer(os.path.basename(outputLayerName),
                                     srs=outSrs, geom_type=ogr.wkbPolygon)
    outFeatureDef = outLayer.GetLayerDefn()
    # create rings from input rings by shifting points
    for feature in inputFeatures:
        # create the poly
        outPoly = ogr.Geometry(ogr.wkbPolygon)
        poly = feature.GetGeometryRef()
        for ring_idx in range(poly.GetGeometryCount()):
            ring = poly.GetGeometryRef(ring_idx)
            # create the ring
            outRing = ogr.Geometry(ogr.wkbLinearRing)
            for i in range(0, ring.GetPointCount()):
                pt = ring.GetPoint(i)
                outRing.AddPoint(pt[0] + offsetGeo[0], pt[1] + offsetGeo[1])
            outPoly.AddGeometry(outRing)
        # create feature
        outFeature = ogr.Feature(outFeatureDef)
        outFeature.SetGeometry(outPoly)
        outLayer.CreateFeature(outFeature)


def copy_shapefile(input, output):
    inputNoExt = os.path.splitext(input)[0]
    outputNoExt = os.path.splitext(output)[0]
    for ext in ['.dbf', '.prj', '.shp', '.shx']:
        shutil.copyfile(inputNoExt + ext, outputNoExt + ext)


def remove_shapefile(input):
    inputNoExt = os.path.splitext(input)[0]
    for ext in ['.dbf', '.prj', '.shp', '.shx']:
        os.remove(inputNoExt + ext)


# project a vector point to image
def ProjectPoint(model, pt):
    # simplest projection model
    px = int((pt[0]-model['corners'][0])/model['project_model'][1]*model['scale'])
    py = int((pt[1]-model['corners'][1])/model['project_model'][5]*model['scale'])
    return [px, py]


def computeMatchingPoints(check_point_list, edge_img, dx, dy):
    img_height = edge_img.shape[0]
    img_width = edge_img.shape[1]

    total_value = 0
    # find overlap mask
    for pt in check_point_list:
        if pt[1]+dy < 0 or pt[1]+dy >= img_height or\
                pt[0]+dx < 0 or pt[0]+dx >= img_width:
            continue
        if edge_img[pt[1]+dy, pt[0]+dx] > 200:
            total_value += 1
    return total_value


def spat_vectors(inputVectorFileNames, inputImageCorners, inputImageSrs,
                 outputMaskFileName, debug=False):
    """
    Returns building features and optionally road features.
    """
    global VECTOR_TYPES
    geometryTypes = [ogr.wkbPolygon, ogr.wkbLineString]
    resultList = []
    for typeIndex in range(len(inputVectorFileNames)):
        inputVectorFileName = inputVectorFileNames[typeIndex]
        inputVector = gdal_utils.ogr_open(inputVectorFileName)
        inputLayer = gdal_utils.ogr_get_layer(inputVector, geometryTypes[typeIndex])
        inputVectorSrs = inputLayer.GetSpatialRef()
        imageVectorDifferentSrs = False if inputVectorSrs.IsSame(inputImageSrs) else True

        layerDefinition = inputLayer.GetLayerDefn()
        hasBuildingField = False
        for i in range(layerDefinition.GetFieldCount()):
            if layerDefinition.GetFieldDefn(i).GetName() == "building":
                hasBuildingField = True
                break

        # clip the shape file first
        outputNoExt = os.path.splitext(outputMaskFileName)[0]
        if imageVectorDifferentSrs:
            outputVectorFile = outputNoExt + "_" + VECTOR_TYPES[typeIndex] + "_original.shp"
        else:
            outputVectorFile = outputNoExt + "_" + VECTOR_TYPES[typeIndex] + "_spat_not_aligned.shp"
        ogr2ogr_args = ["ogr2ogr", "-spat",
                        str(inputImageCorners[0]), str(inputImageCorners[2]),
                        str(inputImageCorners[1]), str(inputImageCorners[3])]
        if imageVectorDifferentSrs:
            ogr2ogr_args.extend(["-spat_srs", str(inputImageSrs)])
        if hasBuildingField:
            ogr2ogr_args.extend(["-where", "building is not null"])
        ogr2ogr_args.extend([outputVectorFile, inputVectorFileName])
        ogr2ogr_args.append(inputLayer.GetName())
        print("Spatial query (clip): {} -> {}".format(
            os.path.basename(inputVectorFileName), os.path.basename(outputVectorFile)))
        response = subprocess.run(ogr2ogr_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if debug:
            print(*ogr2ogr_args)
            print("{}\n{}".format(response.stdout, response.stderr))
        if imageVectorDifferentSrs:
            # convert to the same SRS as the image file
            inputVectorFileName = outputNoExt + "_" + VECTOR_TYPES[typeIndex] + "_original.shp"
            outputVectorFile = outputNoExt + "_" + VECTOR_TYPES[typeIndex] + "_spat_not_aligned.shp"
            ogr2ogr_args = ["ogr2ogr", "-t_srs", str(inputImageSrs),
                            outputVectorFile, inputVectorFileName]
            print("Convert SRS: {} -> {}".format(
                os.path.basename(inputVectorFileName), os.path.basename(outputVectorFile)))
            response = subprocess.run(ogr2ogr_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if debug:
                print(*ogr2ogr_args)
                print("{}\n{}".format(response.stdout, response.stderr))
            else:
                remove_shapefile(inputVectorFileName)

        inputVectorFileName = outputVectorFile
        inputLayerName = os.path.splitext(os.path.basename(inputVectorFileName))[0]

        inputVector = gdal_utils.ogr_open(inputVectorFileName)
        inputLayer = inputVector.GetLayer(inputLayerName)
        inputList = list(inputLayer)
        resultList.append(inputList)
    return resultList


def main(args):
    global VECTOR_TYPES
    parser = argparse.ArgumentParser(
        description="Generate building mask aligned with image. To do that we shift input "
                    "vector to match edges generated from image.")
    parser.add_argument('output_mask',
                        help="Output image mask base name. <output_mask>_buildings.shp, "
                             "<output_mask>_buildings.tif are generated. Optionally "
                             "<output_mask>_roads.tif and <output_mask>_roads.shp are "
                             "also generated. See --input_vectors parameter.")
    parser.add_argument('input_image', help='Orthorectified 8-bit image file')
    parser.add_argument('input_vectors', nargs='+',
                        help='Buildings and optionally road vector files with OSM or '
                             'US Cities data. A polygon layer is chosen for buildings and a '
                             'line string layer is chosen for roads. '
                             'If both building and road layers are in the same vector file just '
                             'pass the file twice. Only elevated bridges are rendered '
                             'by default. If all roads need to be rendered pass --render_roads')
    parser.add_argument('--render_cls', action="store_true",
                        help='Output a CLS image')
    parser.add_argument('--render_roads', action="store_true",
                        help='Render all roads, not only elevated bridges')
    parser.add_argument('--scale', type=float, default=0.2,
                        help='Scale factor. '
                             'We cannot deal with the images with original resolution')
    parser.add_argument('--move_thres', type=float, default=5,
                        help='Distance for edge matching')
    parser.add_argument("--offset", type=float, nargs=2,
                        help="Shift the mask using the offset specified "
                             "(using the SRS of the input_image) instead of the computed offset.")
    parser.add_argument("--debug", action="store_true",
                        help="Print debugging information")
    args = parser.parse_args(args)

    scale = args.scale

    inputImage = gdal_utils.gdal_open(args.input_image, gdal.GA_ReadOnly)
    band = inputImage.GetRasterBand(1)
    if (not band.DataType == gdal.GDT_Byte):
        raise RuntimeError(
            "Input image {} does not have Byte type. Use msi-to-rgb.py to-8bit.py "
            "to convert it.".format(args.input_image))

    projection = inputImage.GetProjection()
    inputImageSrs = osr.SpatialReference(projection)
    gt = inputImage.GetGeoTransform()  # captures origin and pixel size

    left, top = gdal.ApplyGeoTransform(gt, 0, 0)
    right, bottom = gdal.ApplyGeoTransform(gt, inputImage.RasterXSize, inputImage.RasterYSize)
    band = None

    print("Resize and edge detection: {}".format(os.path.basename(args.input_image)))
    color_image = cv2.imread(args.input_image)
    small_color_image = numpy.zeros(
        (int(color_image.shape[0]*scale),
         int(color_image.shape[1]*scale), 3), dtype=numpy.uint8)
    if scale != 1.0:
        small_color_image = cv2.resize(color_image, None, fx=scale, fy=scale)
        color_image = small_color_image
    grayimg = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    edge_img = cv2.Canny(grayimg, 100, 200)
    if args.debug:
        cv2.imwrite(os.path.splitext(args.output_mask)[0] + '_edge.tif', edge_img)

    model = {}
    model['corners'] = [left, top, right, bottom]
    model['project_model'] = gt
    model['scale'] = scale

    inputImageCorners = [left, right, bottom, top]
    features = spat_vectors(
        args.input_vectors, inputImageCorners, inputImageSrs,
        args.output_mask)
    print("Aligning {} buildings ...".format(len(features[0])))
    tmp_img = numpy.zeros([int(color_image.shape[0]), int(color_image.shape[1])],
                          dtype=numpy.uint8)
    for feature in features[0]:
        poly = feature.GetGeometryRef()
        for ring_idx in range(poly.GetGeometryCount()):
            ring = poly.GetGeometryRef(ring_idx)
            rp = []
            for i in range(0, ring.GetPointCount()):
                pt = ring.GetPoint(i)
                rp.append(ProjectPoint(model, pt))
            ring_points = numpy.array(rp)
            ring_points = ring_points.reshape((-1, 1, 2))

            # edge mask of the building cluster
            cv2.polylines(tmp_img, [ring_points], True, (255), thickness=2)

    check_point_list = []
    # build a sparse set to fast process
    for y in range(0, tmp_img.shape[0]):
        for x in range(0, tmp_img.shape[1]):
            if tmp_img[y, x] > 200:
                check_point_list.append([x, y])
    print("Checking {} points ...".format(len(check_point_list)))

    max_value = 0
    index_max_value = 0
    offsetGeo = [0.0, 0.0]
    current = [0, 0]
    if not args.offset:
        offset = [0, 0]
        # shift moves possible from [0, 0]
        moves = [
            [1, 0],    # 0
            [1, 1],    # 1
            [0, 1],    # 2
            [-1, 1],   # 3
            [-1, 0],   # 4
            [-1, -1],  # 5
            [0, -1],   # 6
            [1, -1]]    # 7
        initial_cases = range(8)
        # cases[i] shows shift moves possible after the previous move was cases[i][0]
        # we change direction with at most 45 degrees.
        next_cases = [
            [0, 7, 1],
            [1, 0, 2],
            [2, 1, 3],
            [3, 2, 4],
            [4, 3, 5],
            [5, 4, 6],
            [6, 5, 7],
            [7, 6, 0]
        ]

        # move the mask to match
        cases = initial_cases
        old_max_value = 0
        total_value = computeMatchingPoints(check_point_list, edge_img, 0, 0)
        max_value = total_value
        if args.debug:
            print("Total value for ({}, {}) is: {} (max value: {})".format(
                0, 0, total_value, max_value))
        for i in range(args.move_thres):
            if args.debug:
                print("===== {} =====".format(i))
            while (max_value > old_max_value):
                old_max_value = max_value
                for i in cases:
                    [dx, dy] = moves[i]
                    total_value = computeMatchingPoints(check_point_list, edge_img,
                                                        current[0] + dx, current[1] + dy)
                    if args.debug:
                        print("Total value for ({}, {}) is: {} (max value: {})".format(
                              dx, dy, total_value, max_value))
                    if total_value > max_value:
                        max_value = total_value
                        index_max_value = i
                if (max_value > old_max_value):
                    [dx, dy] = moves[index_max_value]
                    current = [current[0] + dx, current[1] + dy]
                    if args.debug:
                        print("Current: {}".format(current))
                    offset = current
                    cases = next_cases[index_max_value]
                    break

        offsetGeo = gdal.ApplyGeoTransform(gt, offset[0] / scale, offset[1] / scale)
        offsetGeo[0] = offsetGeo[0] - left
        offsetGeo[1] = top - offsetGeo[1]
        print("Using offset: {} ({})".format(offsetGeo, offset))
        if max_value/float(len(check_point_list)) < 0.05:
            print("Fewer than 5% of points match {} / {}. This may happen because of "
                  "missing areas in the orthorectified image. "
                  "Increasing scale may increase the number of points that match.".format(
                    max_value, len(check_point_list)))
    else:
        print("Using offset: {}".format(offsetGeo))
        offsetGeo = args.offset

    for i in range(len(features)):
        outputNoExt = os.path.splitext(args.output_mask)[0]
        outputVectorFile = outputNoExt + "_" + VECTOR_TYPES[i] + "_spat.shp"
        if not (offsetGeo[0] == 0.0 and offsetGeo[1] == 0.0):
            shift_vector(features[i], outputVectorFile, outputNoExt, projection, offsetGeo)
        else:
            inputVectorFileName = outputNoExt + "_" + VECTOR_TYPES[i] + "_spat_not_aligned.shp"
            print("Copy vector -> {}".format(os.path.basename(outputVectorFile)))
            copy_shapefile(inputVectorFileName, outputVectorFile)
        if not args.debug:
            remove_shapefile(outputNoExt + "_" + VECTOR_TYPES[i] + "_spat_not_aligned.shp")

        ogr2ogr_args = ["ogr2ogr", "-clipsrc",
                        str(inputImageCorners[0]), str(inputImageCorners[2]),
                        str(inputImageCorners[1]), str(inputImageCorners[3])]
        outputNoExt = os.path.splitext(args.output_mask)[0]
        ogr2ogr_args.extend([outputNoExt + "_" + VECTOR_TYPES[i] + ".shp",
                            outputNoExt + "_" + VECTOR_TYPES[i] + "_spat.shp"])
        print("Clipping vector file {} -> {}".format(
            os.path.basename(outputNoExt + "_" + VECTOR_TYPES[i] + "_spat.shp"),
            os.path.basename(outputNoExt + "_" + VECTOR_TYPES[i] + ".shp")))
        response = subprocess.run(ogr2ogr_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if args.debug:
            print(*ogr2ogr_args)
            print("{}\n{}".format(response.stdout, response.stderr))
        remove_shapefile(outputNoExt + "_" + VECTOR_TYPES[i] + "_spat.shp")

        if i == 0:
            print("Rasterizing buildings ...")
            if args.render_cls:
                rasterize_args = ["gdal_rasterize", "-ot", "Byte", "-init", "2",
                                  "-burn", "6",
                                  "-ts", str(inputImage.RasterXSize),
                                  str(inputImage.RasterYSize),
                                  "-te", str(inputImageCorners[0]), str(inputImageCorners[2]),
                                  str(inputImageCorners[1]), str(inputImageCorners[3])]
            else:
                # make buildings red
                rasterize_args = ["gdal_rasterize", "-ot", "Byte",
                                  "-burn", "255", "-burn", "0", "-burn", "0", "-burn", "255",
                                  "-ts", str(inputImage.RasterXSize),
                                  str(inputImage.RasterYSize),
                                  "-te", str(inputImageCorners[0]), str(inputImageCorners[2]),
                                  str(inputImageCorners[1]), str(inputImageCorners[3])]

            outputNoExt = os.path.splitext(args.output_mask)[0]
            rasterize_args.extend([outputNoExt + "_" + VECTOR_TYPES[i] + ".shp",
                                  outputNoExt + "_" + VECTOR_TYPES[i] + ".tif"])
            print("Rasterizing {} -> {}".format(
                os.path.basename(outputNoExt + "_" + VECTOR_TYPES[i] + ".shp"),
                os.path.basename(outputNoExt + "_" + VECTOR_TYPES[i] + ".tif")))
            response = subprocess.run(
                rasterize_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if args.debug:
                print(*rasterize_args)
                print("{}\n{}".format(response.stdout, response.stderr))
        else:
                print("Rasterizing bridges ...")
                outputNoExt = os.path.splitext(args.output_mask)[0]
                input = os.path.basename(outputNoExt + "_" + VECTOR_TYPES[i] + ".shp")
                output = os.path.basename(outputNoExt + "_" + VECTOR_TYPES[i] + "_bridges.tif")
                bridges = rasterize.rasterize_file_dilated_line(
                    input, inputImage, output,
                    numpy.ones((3, 3)), dilation_iterations=20,
                    query=rasterize.ELEVATED_ROADS_QUERY,
                )
                if not args.debug:
                    os.remove(output)
                if args.render_roads:
                    output = os.path.basename(outputNoExt + "_" + VECTOR_TYPES[i] + "_roads.tif")
                    roads = rasterize.rasterize_file_dilated_line(
                        input, inputImage, output,
                        numpy.ones((3, 3)), dilation_iterations=20, query=rasterize.ROADS_QUERY)
                    if not args.debug:
                        os.remove(output)
                buildingsData = gdal_utils.gdal_open(
                    os.path.basename(outputNoExt + "_" + VECTOR_TYPES[0] + ".tif"),
                    gdal.GA_ReadOnly)
                if args.render_cls:
                    cls = buildingsData.GetRasterBand(1).ReadAsArray()
                    if args.render_roads:
                        cls[roads] = 11
                    cls[bridges] = 17
                    gdal_utils.gdal_save(cls, inputImage,
                                         os.path.basename(outputNoExt + ".tif"),
                                         gdal.GDT_Byte, options=['COMPRESS=DEFLATE'])
                else:
                    red = buildingsData.GetRasterBand(1).ReadAsArray()
                    green = buildingsData.GetRasterBand(2).ReadAsArray()
                    blue = buildingsData.GetRasterBand(3).ReadAsArray()
                    opacity = buildingsData.GetRasterBand(4).ReadAsArray()
                    if args.render_roads:
                        red[roads] = 0
                        green[roads] = 255
                        blue[roads] = 0
                        opacity[roads] = 255
                    red[bridges] = 0
                    green[bridges] = 0
                    blue[bridges] = 255
                    opacity[bridges] = 255
                    gdal_utils.gdal_save([red, green, blue, opacity], inputImage,
                                         os.path.basename(outputNoExt + ".tif"),
                                         gdal.GDT_Byte, options=['COMPRESS=DEFLATE'])
                if not args.debug:
                    os.remove(os.path.basename(outputNoExt + "_" + VECTOR_TYPES[0] + ".tif"))


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
