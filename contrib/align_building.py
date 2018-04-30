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
parser.add_argument("-o", "--no_offset", action="store_true",
                    help="Write original OSM data.")
parser.add_argument("-c", "--clusters", action="store_true",
                    help="Cluster buildings and compute translation for every cluster. "
                    "Otherwise compute a global translation.")
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
full_res_mask = np.zeros((sourceImage.RasterYSize,sourceImage.RasterXSize,4))

projection = sourceImage.GetProjection()
gt = sourceImage.GetGeoTransform() # captures origin and pixel size
print('Origin:', (gt[0], gt[3]))
print('Pixel size:', (gt[1], gt[5]))

left = gdal.ApplyGeoTransform(gt,0,0)[0]
top = gdal.ApplyGeoTransform(gt,0,0)[1]
right = gdal.ApplyGeoTransform(gt,sourceImage.RasterXSize,sourceImage.RasterYSize)[0]
bottom = gdal.ApplyGeoTransform(gt,sourceImage.RasterXSize,sourceImage.RasterYSize)[1]
band = None


color_image = cv2.imread(args.input_img)

small_color_image = np.zeros((int(color_image.shape[0]*scale),\
        int(color_image.shape[1]*scale), 3))

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

ds = ogr.Open(args.input_osm)
if ds.GetLayerCount()<=1:
    layer = ds.GetLayerByIndex(0)
else:
    layer = ds.GetLayerByIndex(3)
nameList = []

draw_flag = False

print("Clipping buildings to source_img ...")
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

if args.clusters:
    print("Clustering ...")
    building_cluster_list = Utils.GetBuildingCluster(building_list, model)
    print("Got {} clusters".format(len(building_cluster_list)))
else:
    building_cluster_list = []
    building_cluster_list.append(building_list)

for cluster_idx, building_cluster in enumerate(building_cluster_list):
    tmp_img = np.zeros((int(color_image.shape[0]),\
        int(color_image.shape[1])))
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

    print("Cluster {}: {} buildings".format(cluster_idx, len(building_cluster)))
    print("======================================================================")

    max_value = 0
    offsetx = 0
    offsety = 0
    if not args.no_offset:
        img_height = edge_img.shape[0]
        img_width = edge_img.shape[1]
        # [0, 0] and shift moves possible from there
        initial_cases = [
            [0,0],
            [1, 0],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
            [0, -1],
            [1,-1]]
        # cases[i] shows shift moves possible after the previous move was cases[i][0]
        # we change direction with at most 45 degrees.
        cases = [
            [[1, 0], [1, 1], [1,-1]],
            [[1, 1], [1, 0], [0, 1]],
            [[0, 1], [1, 1], [-1, 1]],
            [[-1, 1], [0, 1], [-1, 0]],
            [[-1, 0], [-1, 1], [-1, -1]],
            [[-1, -1], [-1, 0], [0, -1]],
            [[0, -1], [-1, -1], [1, -1]],
            [[1,-1], [0, -1], [1, 0]]
        ]

        #move the mask to match
        for dy in range(-1*args.move_thres,args.move_thres,1):
            for dx in range(-1*args.move_thres,args.move_thres,1):
                total_value = 0
                #find overlap mask
                for pt in check_point_list:
                    if pt[1]+dy<0 or pt[1]+dy>=img_height or\
                            pt[0]+dx<0 or pt[0]+dx >= img_width:
                        continue
                    if edge_img[pt[1]+dy,pt[0]+dx] > 200:
                        total_value += 1
                print("Total value for ({}, {}) is: {} (max value: {})".format(
                    dx, dy, total_value, max_value))
                if total_value>max_value:
                    max_value = total_value
                    offsetx = dx
                    offsety = dy

        print("Offset ({}, {})\n".format(offsetx, offsety))
        #do something to deal with the bad data
        if max_value/float(len(check_point_list))<0.05:
            offsetx = 0
            offsety = 0

    index = 0
    #draw final mask
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


ds.Destroy()

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
