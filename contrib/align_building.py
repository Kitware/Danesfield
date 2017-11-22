"""
Align vector layer to the tiff image. The input image must be warped by gdalwarp first!!!!

Author: Xu Zhang
Email: xu.zhang@columbia.edu.cn
"""

import gdal, ogr, os, osr
import numpy as np
import cv2
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
mpl.use('Agg')
import matplotlib.pyplot as plt
import plotly.plotly as py
import argparse

import Utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_img', default='../data/Dayton.tiff', help='Input geotiff file')
parser.add_argument('--input_osm', default='../data/Dayton_map.osm', help='Input osm file')
parser.add_argument('--scale' , type=float, default=0.2, help='Scale factor. We cannot deal with the images with original resolution')
parser.add_argument('--cluster_thres' , type=float, default=100, help='Distance for building clustering')
parser.add_argument('--move_thres' , type=float, default=40, help='Distance for edge matching')
parser.add_argument("-o", "--no_offset", action="store_true",
                    help="Don't align vector based on edge response(original projection)")
args = parser.parse_args()

base=os.path.basename(args.input_img)
basename = os.path.splitext(base)[0]

input_img = args.input_img
scale = args.scale

cluster_thres = args.cluster_thres
color_image = cv2.imread(input_img) 

small_color_image = np.zeros((int(color_image.shape[0]*scale),\
        int(color_image.shape[1]*scale), 3))

if scale != 1.0:
    small_color_image = cv2.resize(color_image, None, fx=scale, fy=scale)
    color_image = small_color_image

grayimg = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

#otsu auto threshold, doesn't work
#high_thresh, thresh_im = cv2.threshold(grayimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#lowThresh = 0.5*high_thresh
#edge_img = cv2.Canny(grayimg, lowThresh, high_thresh)
edge_img = cv2.Canny(grayimg, 200, 300)
cv2.imwrite('../data/{}_edge.jpg'.format(basename),edge_img)

# open the GDAL file
sourceImage = gdal.Open(input_img, gdal.GA_ReadOnly)
rpcMetaData = sourceImage.GetMetadata('RPC')
full_res_mask = np.ones((sourceImage.RasterYSize,sourceImage.RasterXSize))*255

gt = sourceImage.GetGeoTransform() # captures origin and pixel size
print('Origin:', (gt[0], gt[3]))
print('Pixel size:', (gt[1], gt[5]))

left = gdal.ApplyGeoTransform(gt,0,0)[0]
top = gdal.ApplyGeoTransform(gt,0,0)[1]
right = gdal.ApplyGeoTransform(gt,sourceImage.RasterXSize,sourceImage.RasterYSize)[0]
bottom = gdal.ApplyGeoTransform(gt,sourceImage.RasterXSize,sourceImage.RasterYSize)[1]

model = {}
model['corners'] = [left, top, right, bottom]
model['project_model'] = gt 
model['scale'] = scale

ds = ogr.Open(args.input_osm)
layer = ds.GetLayerByIndex(3)
nameList = []

draw_flag = False

building_list = []
for feature in layer:
    if feature.GetField("building") != None:  # only streets
        flag = True
        #print("haha")
        name = feature.GetField("name")
        geom = feature.GetGeometryRef()
        g = geom.GetGeometryRef(0)
        for shape_idx in range(g.GetGeometryCount()):
            polygon = g.GetGeometryRef(shape_idx)
            for i in range(0, polygon.GetPointCount()):
                pt = polygon.GetPoint(i)
                #print(pt[0],pt[1])
                if pt[0]>left and pt[0]<right and pt[1]<top and pt[1]>bottom:
                    pass
                else:
                    flag = False
                    break
            if not flag:
                break
        if flag:
            building_list.append(feature)

building_cluster_list = Utils.GetBuildingCluster(building_list, model)

for cluster_idx, building_cluster in enumerate(building_cluster_list):
    
    tmp_img = np.zeros((int(color_image.shape[0]),\
        int(color_image.shape[1])))
    poly_array_list = []
    for building in building_cluster:
        name = building.GetField("name")
        geom = building.GetGeometryRef()
        g = geom.GetGeometryRef(0)
        for shape_idx in range(g.GetGeometryCount()):
            polygon = g.GetGeometryRef(shape_idx)
            poly_point_list = []
            for i in range(0, polygon.GetPointCount()):
                pt = polygon.GetPoint(i)
                poly_point_list.append(Utils.ProjectPoint(model,pt))
            poly_array = np.array(poly_point_list)
            poly_array = poly_array.reshape((-1,1,2))
            poly_array_list.append(poly_array)

            #edge mask of the building cluster
            cv2.polylines(tmp_img,[poly_array],True,(255),thickness=2)
            #cv2.imshow('image',tmp_img)
            #cv2.imwrite('contour.jpg',tmp_img)
            #cv2.waitKey(0)
            check_point_list = []

    # build a sparse set to fast process
    for y in range(0,tmp_img.shape[0]):
        for x in range(0,tmp_img.shape[1]):
            if tmp_img[y,x] > 200:
                check_point_list.append([x,y])

    print(len(check_point_list))

    max_value = 0
    offsetx = 0
    offsety = 0
    if not args.no_offset:
        img_height = edge_img.shape[0]
        img_width = edge_img.shape[1]

        #move the mask to match
        for dy in range(-1*args.move_thres,args.move_thres,2):
            for dx in range(-1*args.move_thres,args.move_thres,2):
                total_value = 0
                #find overlap mask
                for pt in check_point_list:
                    if pt[1]+dy<0 or pt[1]+dy>=img_height or\
                            pt[0]+dx<0 or pt[0]+dx >= img_width:
                        continue
                    if edge_img[pt[1]+dy,pt[0]+dx] > 200:
                        total_value += 1

                if total_value>max_value:
                    max_value = total_value
                    offsetx = dx
                    offsety = dy

        print(max_value)
        print(offsetx, offsety)
        #do something to deal with the bad data
        if max_value/float(len(check_point_list))<0.1:
            offsetx = 0
            offsety = 0

    index = 0
    #draw final mask
    for building in building_cluster:
        geom = building.GetGeometryRef()
        g = geom.GetGeometryRef(0)
        for shape_idx in range(g.GetGeometryCount()):
            poly_array = poly_array_list[index]
            index = index+1
            for i in range(len(poly_array)):
                poly_array[i,0,0] = int((poly_array[i,0,0] + offsetx)/scale)
                poly_array[i,0,1] = int((poly_array[i,0,1] + offsety)/scale)
            cv2.fillPoly(full_res_mask,[poly_array],True,(0))
            #cv2.imshow('image',full_res_mask)
            #cv2.waitKey(0)
                
#print(nameList)
ds.Destroy()
if not args.no_offset:
    cv2.imwrite('../data/{}_mask.png'.format(basename),full_res_mask)
else:
    cv2.imwrite('../data/{}_original_mask.png'.format(basename),full_res_mask)
sourceImage = None
