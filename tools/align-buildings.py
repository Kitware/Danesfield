"""
Align vector layer to the tiff image. The input image must be warped by gdalwarp first!!!!

Author: Xu Zhang
Email: xu.zhang@columbia.edu.cn
"""

import argparse
import cv2
import gdal
import numpy as np
import ogr
import osr
import os
import pdb
import subprocess
import shutil
import sys

#project a vector point to image
def ProjectPoint(model, pt):
    #simplest projection model
    px = int((pt[0]-model['corners'][0])/model['project_model'][1]*model['scale'])
    py = int((pt[1]-model['corners'][1])/model['project_model'][5]*model['scale'])
    return [px,py]

def computeMatchingPoints(check_point_list, edge_img, dx, dy, debug = False):
    img_height = edge_img.shape[0]
    img_width = edge_img.shape[1]

    total_value = 0
    #find overlap mask
    for pt in check_point_list:
        if pt[1]+dy<0 or pt[1]+dy>=img_height or\
                pt[0]+dx<0 or pt[0]+dx >= img_width:
            continue
        if edge_img[pt[1]+dy,pt[0]+dx] > 200:
            total_value += 1
    if debug:
        print("Total value for ({}, {}) is: {} (max value: {})".format(
            dx, dy, total_value, max_value))
    return total_value

def readAndClipVectorFile(inputVectorFile, inputLayerName, output_mask,
                          inputImageCorners, inputImageSrs, debug = False):
    inputVector = ogr.Open(inputVectorFile)
    if not inputVector:
        print("Invalid vector file: {}".format(inputVectorFile))
        return None
    if inputLayerName:
        inputLayer = inputVector.GetLayer(inputLayerName)
        if not inputLayer:
            print("Invalid layer: {}".format(inputLayerName))
            return None
    else:
        layerCount = inputVector.GetLayerCount()
        for i in range(layerCount):
            inputLayer = inputVector.GetLayerByIndex(i)
            inputLayerName = inputLayer.GetName()
            type = inputLayer.GetGeomType()
            if (type == ogr.wkbMultiPolygon or type == ogr.wkbPolygon):
                break
        if i == layerCount:
            print("No polygon or multipolygon layer found")
            return None
    inputVectorSrs = inputLayer.GetSpatialRef()
    imageVectorDifferentSrs = False if inputVectorSrs.IsSame(inputImageSrs) else True

    layerDefinition = inputLayer.GetLayerDefn()
    hasBuildingField = False
    for i in range(layerDefinition.GetFieldCount()):
        if layerDefinition.GetFieldDefn(i).GetName() == "building":
            hasBuildingField = True
            break

    # clip the shape file first
    outputNoExt = os.path.splitext(output_mask)[0]
    if imageVectorDifferentSrs:
        destinationVectorFile = outputNoExt + "_original.shp"
    else:
        destinationVectorFile = outputNoExt + "_spat_not_aligned.shp"
    ogr2ogr_args = ["ogr2ogr", "-spat",
                    str(inputImageCorners[0]), str(inputImageCorners[2]),
                    str(inputImageCorners[1]), str(inputImageCorners[3])]
    if imageVectorDifferentSrs:
        ogr2ogr_args.extend(["-spat_srs", str(inputImageSrs)])
    if hasBuildingField:
        ogr2ogr_args.extend(["-where", "building is not null"])
    ogr2ogr_args.extend([destinationVectorFile, inputVectorFile])
    if inputLayerName:
        ogr2ogr_args.append(inputLayerName)
    print("Spatial query (clip): {} -> {}".format(
        os.path.basename(inputVectorFile), os.path.basename(destinationVectorFile)))
    response = subprocess.run(ogr2ogr_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if args.debug:
        print(*ogr2ogr_args)
        print("{}\n{}".format(response.stdout, response.stderr))
    if imageVectorDifferentSrs:
        # convert to the same SRS as the image file
        inputVectorFile = outputNoExt + "_original.shp"
        destinationVectorFile = outputNoExt + "_spat_not_aligned.shp"
        ogr2ogr_args = ["ogr2ogr", "-t_srs", str(inputImageSrs),
                        destinationVectorFile, inputVectorFile]
        print("Convert SRS: {} -> {}".format(
            os.path.basename(inputVectorFile), os.path.basename(destinationVectorFile)))
        response = subprocess.run(ogr2ogr_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if args.debug:
            print(*ogr2ogr_args)
            print("{}\n{}".format(response.stdout, response.stderr))

    inputVectorFile = destinationVectorFile
    inputLayerName = os.path.splitext(os.path.basename(inputVectorFile))[0]

    inputVector = ogr.Open(inputVectorFile)
    inputLayer = inputVector.GetLayer(inputLayerName)
    inputList = list(inputLayer)
    if len(inputList) == 0:
        print("No buildings in the clipped vector file")
        return None
    return inputList


draw_color = [(255,0,0,255),(255,255,0,255),(255,0,255,255),(0,255,0,255),(0,255,255,255),\
        (0,0,255,255),(255,255,255,255)]

parser = argparse.ArgumentParser(
    description="Shift buildings to match edges generated from image")
parser.add_argument('input_image', help='Orthorectified 8-bit image file')
parser.add_argument('input_vector', help='Vector file with OSM or US Cities data')
parser.add_argument('output_mask',
                    help="Output image mask (tif and shp) generated from the input_vector "
                         "and aligned with input_image. A _spat.shp file is also "
                         "generated where buildings are not clipped at image boundaries.")

parser.add_argument('--input_layer' ,
                    help='Input layer name that contains buildings in input_vector')
parser.add_argument('--scale' , type=float, default=0.2,
                    help='Scale factor. We cannot deal with the images with original resolution')
parser.add_argument('--move_thres' , type=float, default=5,
                    help='Distance for edge matching')
parser.add_argument("--no_offset", action="store_true",
                    help="Write original OSM data.")
parser.add_argument("--debug", action="store_true",
                    help="Print debugging information")
args = parser.parse_args()

scale = args.scale

# open the GDAL file
inputImage = gdal.Open(args.input_image, gdal.GA_ReadOnly)
band = inputImage.GetRasterBand(1)
if (not band.DataType == gdal.GDT_Byte):
    print("Input image {} does not have Byte type.".format(args.input_image))
    sys.exit(10)

projection = inputImage.GetProjection()
inputImageSrs = osr.SpatialReference(projection)
gt = inputImage.GetGeoTransform() # captures origin and pixel size

left,top = gdal.ApplyGeoTransform(gt,0,0)
right,bottom = gdal.ApplyGeoTransform(gt,inputImage.RasterXSize,inputImage.RasterYSize)
band = None

print("Resize and edge detection: {}".format(os.path.basename(args.input_image)))
color_image = cv2.imread(args.input_image)
small_color_image = np.zeros((int(color_image.shape[0]*scale),\
        int(color_image.shape[1]*scale), 3), dtype=np.uint8)
if scale != 1.0:
    small_color_image = cv2.resize(color_image, None, fx=scale, fy=scale)
    color_image = small_color_image
grayimg = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
edge_img = cv2.Canny(grayimg, 100, 200)
cv2.imwrite(os.path.splitext(args.output_mask)[0] + '_edge.tif',edge_img)

model = {}
model['corners'] = [left, top, right, bottom]
model['project_model'] = gt
model['scale'] = scale

inputImageCorners = [left, right, bottom, top]
building_cluster = readAndClipVectorFile(args.input_vector, args.input_layer,args.output_mask,
                                      inputImageCorners, inputImageSrs)
if not building_cluster:
    sys.exit(10)

print("Aligning {} buildings ...".format(len(building_cluster)))
tmp_img = np.zeros([int(color_image.shape[0]), int(color_image.shape[1])],
                   dtype=np.uint8)
for feature in building_cluster:
    multipoly = feature.GetGeometryRef()
    if multipoly.GetGeometryType() == ogr.wkbMultiPolygon:
        poly = multipoly.GetGeometryRef(0)
    else:
        poly = multipoly
    for ring_idx in range(poly.GetGeometryCount()):
        ring = poly.GetGeometryRef(ring_idx)
        rp = []
        for i in range(0, ring.GetPointCount()):
            pt = ring.GetPoint(i)
            rp.append(ProjectPoint(model,pt))
        ring_points = np.array(rp)
        ring_points = ring_points.reshape((-1,1,2))

        #edge mask of the building cluster
        cv2.polylines(tmp_img,[ring_points],True,(255),thickness=2)
        check_point_list = []

# build a sparse set to fast process
for y in range(0,tmp_img.shape[0]):
    for x in range(0,tmp_img.shape[1]):
        if tmp_img[y,x] > 200:
            check_point_list.append([x,y])
print("Checking {} points ...".format(len(check_point_list)))

max_value = 0
index_max_value = 0
offset = [0, 0]
current = [0, 0]
if not args.no_offset:
    img_height = edge_img.shape[0]
    img_width = edge_img.shape[1]
    # shift moves possible from [0, 0]
    moves = [
        [1, 0],   # 0
        [1, 1],   # 1
        [0, 1],   # 2
        [-1, 1],  # 3
        [-1, 0],  # 4
        [-1, -1], # 5
        [0, -1],  # 6
        [1,-1]]   # 7
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

    #move the mask to match
    cases = initial_cases
    old_max_value = 0
    max_value = computeMatchingPoints(check_point_list, edge_img, 0, 0, args.debug)
    for i in range(args.move_thres):
        if args.debug:
            print("===== {} =====".format(i))
        while (max_value > old_max_value):
            old_max_value = max_value
            for i in cases:
                [dx, dy] = moves[i]
                total_value = computeMatchingPoints(check_point_list, edge_img,
                                                    current[0] + dx, current[1] + dy, args.debug)
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
    print("Resulting offset: {}, {}".format(offset, offsetGeo))
    #do something to deal with the bad data
    if max_value/float(len(check_point_list))<0.05:
        print("Fewer than 5% of points match {} / {}. Set offset to 0.".format(
            max_value, len(check_point_list)))
        offset = [0, 0]

outputNoExt = os.path.splitext(args.output_mask)[0]
destinationVectorFile = outputNoExt + "_spat.shp"
if not (offset[0] == 0 and offset[1] == 0):
    outDriver = ogr.GetDriverByName("ESRI Shapefile")
    print("Shifting vector -> {}".format(os.path.basename(destinationVectorFile)))
    outVector = outDriver.CreateDataSource(destinationVectorFile)
    outSrs = osr.SpatialReference(projection)
    # create layer
    outLayer = outVector.CreateLayer(os.path.basename(outputNoExt),
                                     srs=outSrs, geom_type=ogr.wkbPolygon)
    outFeatureDef = outLayer.GetLayerDefn()
    # create rings from input rings by shifting points
    for feature in building_cluster:
        # create the poly
        outPoly = ogr.Geometry(ogr.wkbPolygon)
        multipoly = feature.GetGeometryRef()
        if multipoly.GetGeometryType() == ogr.wkbMultiPolygon:
            poly = multipoly.GetGeometryRef(0)
        else:
            poly = multipoly
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
    outLayer = None
    outVector = None
else:
    inputVectorFile = os.path.splitext(outputNoExt + "_spat_not_aligned.shp")[0]
    destinationVectorFile = os.path.splitext(destinationVectorFile)[0]
    print("Copy vector -> {}".format(os.path.basename(destinationVectorFile)))
    for ext in ['.dbf', '.prj', '.shp', '.shx']:
        shutil.copyfile(inputVectorFile + ext, destinationVectorFile + ext)

ogr2ogr_args = ["ogr2ogr", "-clipsrc",
                str(inputImageCorners[0]), str(inputImageCorners[2]),
                str(inputImageCorners[1]), str(inputImageCorners[3])]
outputNoExt = os.path.splitext(args.output_mask)[0]
ogr2ogr_args.extend([outputNoExt + ".shp", outputNoExt + "_spat.shp"])
print("Clipping vector file {} -> {}".format(
    os.path.basename(outputNoExt + "_spat.shp"),
    os.path.basename(outputNoExt + ".shp")))
response = subprocess.run(ogr2ogr_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if args.debug:
    print(*ogr2ogr_args)
    print("{}\n{}".format(response.stdout, response.stderr))


rasterize_args = ["gdal_rasterize", "-ot", "Byte",
                  "-burn", "255", "-burn", "0", "-burn", "0", "-burn", "255",
                  "-ts", str(inputImage.RasterXSize),
                  str(inputImage.RasterYSize)]
outputNoExt = os.path.splitext(args.output_mask)[0]
rasterize_args.extend([outputNoExt + ".shp", outputNoExt + ".tif"])
print("Rasterizing {} -> {}".format(os.path.basename(outputNoExt + ".shp"),
                                    os.path.basename(outputNoExt + ".tif")))
response = subprocess.run(rasterize_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if args.debug:
    print(*rasterize_args)
    print("{}\n{}".format(response.stdout, response.stderr))
