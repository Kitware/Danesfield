"""
Align vector layer to the tiff image. The input image must be warped by gdalwarp first!!!!

Author: Xu Zhang
Email: xu.zhang@columbia.edu.cn
"""

import argparse
import gdal, ogr, os, osr
import numpy as np
import cv2
import pdb
import sys
import Utils

def totalMatchingPoints(check_point_list, edge_img, dx, dy, debug = False):
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



draw_color = [(255,0,0,255),(255,255,0,255),(255,0,255,255),(0,255,0,255),(0,255,255,255),\
        (0,0,255,255),(255,255,255,255)]

parser = argparse.ArgumentParser(description='')
parser.add_argument('input_img', help='Orthorectified 8-bit image file')
parser.add_argument('input_osm', help='OSM vector file')
parser.add_argument('output_mask',
                    help='Output image mask (tif) generated from input_osm and aligned with input_img')
parser.add_argument('--scale' , type=float, default=0.2,
                    help='Scale factor. We cannot deal with the images with original resolution')
parser.add_argument('--move_thres' , type=float, default=5,
                    help='Distance for edge matching')
parser.add_argument("--no_offset", action="store_true",
                    help="Write original OSM data.")
parser.add_argument("--clusters", action="store_true",
                    help="Cluster buildings and compute translation for every cluster. "
                    "Otherwise compute a global translation.")
parser.add_argument("--debug", action="store_true",
                    help="Print debugging information")
args = parser.parse_args()

base=os.path.basename(args.output_mask)
basename = os.path.splitext(base)[0]
output_dir = os.path.dirname(args.output_mask)

scale = args.scale

# open the GDAL file
sourceImage = gdal.Open(args.input_img, gdal.GA_ReadOnly)
band = sourceImage.GetRasterBand(1)
if (not band.DataType == gdal.GDT_Byte):
    print("Input image {} does not have Byte type.".format(args.input_img))
    sys.exit(10)

projection = sourceImage.GetProjection()
gt = sourceImage.GetGeoTransform() # captures origin and pixel size
print('Origin:', (gt[0], gt[3]))
print('Pixel size:', (gt[1], gt[5]))

left = gdal.ApplyGeoTransform(gt,0,0)[0]
top = gdal.ApplyGeoTransform(gt,0,0)[1]
right = gdal.ApplyGeoTransform(gt,sourceImage.RasterXSize,sourceImage.RasterYSize)[0]
bottom = gdal.ApplyGeoTransform(gt,sourceImage.RasterXSize,sourceImage.RasterYSize)[1]
band = None

print("Resize and edge detection ...")
color_image = cv2.imread(args.input_img)
small_color_image = np.zeros((int(color_image.shape[0]*scale),\
        int(color_image.shape[1]*scale), 3), dtype=np.uint8)
if scale != 1.0:
    small_color_image = cv2.resize(color_image, None, fx=scale, fy=scale)
    color_image = small_color_image
grayimg = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
edge_img = cv2.Canny(grayimg, 100, 200)
cv2.imwrite(output_dir + '/{}_edge.tif'.format(basename),edge_img)

model = {}
model['corners'] = [left, top, right, bottom]
model['project_model'] = gt
model['scale'] = scale

print("Clipping buildings to source_img ...")
ds = ogr.Open(args.input_osm)
if ds.GetLayerCount()<=1:
    layer = ds.GetLayerByIndex(0)
else:
    layer = ds.GetLayerByIndex(3)
nameList = []

building_list = []
for feature in layer:
    building_flag = False
    try:
        tmp_field = feature.GetField("building")
        if tmp_field != None:
            building_flag = True
        else:
            building_flag = False
    except:
        building_flag = True

    if building_flag:
        flag = True
        geom = feature.GetGeometryRef()
        shape_ptr = None
        g = geom.GetGeometryRef(0)
        #pdb.set_trace()
        #print(g.GetPointCount())
        if g.GetPointCount()<=0:#for osm data
            shape_ptr = g
        else:#for us city data
            shape_ptr = geom
        for shape_idx in range(shape_ptr.GetGeometryCount()):
            polygon = shape_ptr.GetGeometryRef(shape_idx)
            for i in range(0, polygon.GetPointCount()):
                pt = polygon.GetPoint(i)
                if pt[0]>left and pt[0]<right and pt[1]<top and pt[1]>bottom:
                    pass
                else:
                    flag = False
                    break
            if not flag:
                break
        if flag:
            building_list.append(feature)
ds = None

if args.clusters:
    print("Clustering ...")
    building_cluster_list = Utils.GetBuildingCluster(building_list, model)
    print("Got {} clusters".format(len(building_cluster_list)))
else:
    building_cluster_list = []
    building_cluster_list.append(building_list)

full_res_mask = np.zeros((sourceImage.RasterYSize,sourceImage.RasterXSize, 4),
                         dtype=np.uint8)
for cluster_idx, building_cluster in enumerate(building_cluster_list):
    print("Aligning cluster {}: {} buildings ...".format(cluster_idx, len(building_cluster)))
    tmp_img = np.zeros([int(color_image.shape[0]), int(color_image.shape[1])],
                       dtype=np.uint8)
    poly_array_list = []
    for building in building_cluster:
        geom = building.GetGeometryRef()
        g = geom.GetGeometryRef(0)
        if g.GetPointCount()<=0:
            shape_ptr = g
        else:
            shape_ptr = geom
        for shape_idx in range(shape_ptr.GetGeometryCount()):
            polygon = shape_ptr.GetGeometryRef(shape_idx)
            poly_point_list = []
            for i in range(0, polygon.GetPointCount()):
                pt = polygon.GetPoint(i)
                poly_point_list.append(Utils.ProjectPoint(model,pt))
            poly_array = np.array(poly_point_list)
            poly_array = poly_array.reshape((-1,1,2))
            poly_array_list.append(poly_array)

            #edge mask of the building cluster
            cv2.polylines(tmp_img,[poly_array],True,(255),thickness=2)
            check_point_list = []

    # build a sparse set to fast process
    for y in range(0,tmp_img.shape[0]):
        for x in range(0,tmp_img.shape[1]):
            if tmp_img[y,x] > 200:
                check_point_list.append([x,y])

    max_value = 0
    index_max_value = 0
    offsetx = 0
    offsety = 0
    current_dx = 0
    current_dy = 0
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
        max_value = totalMatchingPoints(check_point_list, edge_img, 0, 0, args.debug)
        for i in range(args.move_thres):
            if args.debug:
                print("===== {} =====".format(i))
            while (max_value > old_max_value):
                old_max_value = max_value
                for i in cases:
                    [dx, dy] = moves[i]
                    total_value = totalMatchingPoints(check_point_list, edge_img,
                                                      current_dx + dx, current_dy + dy, args.debug)
                    if total_value > max_value:
                        max_value = total_value
                        index_max_value = i
                if (max_value > old_max_value):
                    [dx, dy] = moves[index_max_value]
                    [current_dx, current_dy] = [current_dx + dx, current_dy + dy]
                    if args.debug:
                        print("Current: {}".format([current_dx, current_dy]))
                    [offsetx, offsety] = [current_dx, current_dy]
                    cases = next_cases[index_max_value]
                    break

        print("Resulting offset: ({}, {})".format(offsetx, offsety))
        #do something to deal with the bad data
        if max_value/float(len(check_point_list))<0.05:
            offsetx = 0
            offsety = 0

    index = 0
    print("Drawing mask ...")
    for building in building_cluster:
        geom = building.GetGeometryRef()
        g = geom.GetGeometryRef(0)
        if g.GetPointCount()<=0:
            shape_ptr = g
        else:
            shape_ptr = geom
        for shape_idx in range(shape_ptr.GetGeometryCount()):
            poly_array = poly_array_list[index]
            index = index+1
            for i in range(len(poly_array)):
                poly_array[i,0,0] = int((poly_array[i,0,0] + offsetx)/scale)
                poly_array[i,0,1] = int((poly_array[i,0,1] + offsety)/scale)
            poly_array = np.array([poly_array])
            cv2.fillPoly(full_res_mask, poly_array, draw_color[cluster_idx%len(draw_color)])


cv2.imwrite(args.output_mask, full_res_mask)

# write spatial reference information
outputImage = gdal.Open(args.output_mask, gdal.GA_Update)
outputImage.SetProjection(projection)
outputImage.SetGeoTransform(gt)
# band = outputImage.GetRasterBand(1)
# band.SetNoDataValue(0)
band = outputImage.GetRasterBand(4)
band.SetRasterColorInterpretation(gdal.GCI_AlphaBand)
outputImage = None
