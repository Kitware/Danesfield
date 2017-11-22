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
parser.add_argument('--input_img', default='../data/Dayton.tiff', help='folder to output log')
parser.add_argument('--input_osm', default='../data/Dayton_map.osm', help='folder to output log')
parser.add_argument('--scale' , type=float, default=0.2, help='beta hyperparameter value')
parser.add_argument('--cluster_thres' , type=float, default=200, help='beta hyperparameter value')
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
edge_img = cv2.Canny(grayimg, 100, 200)
cv2.imwrite('edge.jpg',edge_img)

# open the GDAL file
sourceImage = gdal.Open(input_img, gdal.GA_ReadOnly)
rpcMetaData = sourceImage.GetMetadata('RPC')
full_res_mask = np.zeros((sourceImage.RasterYSize,sourceImage.RasterXSize))

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

building_list = []
for feature in layer:
    if feature.GetField("building") != None:  # only buildings
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

building_cluster_list = Utils.GetBuildingCluster(building_list, model, cluster_thres)

for cluster_idx, building_cluster in enumerate(building_cluster_list):
    print(len(building_cluster))
    tmp_img = np.zeros((int(color_image.shape[0]),\
        int(color_image.shape[1])))
    poly_array_list = []
    for building in building_cluster:
        name = building.GetField("name")
        geom = building.GetGeometryRef()
        g = geom.GetGeometryRef(0)
        for shape_idx in range(g.GetGeometryCount()):
            #print(g.GetGeometryCount())
            polygon = g.GetGeometryRef(shape_idx)
            poly_point_list = []
            for i in range(0, polygon.GetPointCount()):
                pt = polygon.GetPoint(i)
                poly_point_list.append(Utils.ProjectPoint(model,pt))
            poly_array = np.array(poly_point_list)
            #print(poly_array)
            poly_array = poly_array.reshape((-1,1,2))
            poly_array_list.append(poly_array)

    max_value = 0
    offsetx = 0
    offsety = 0

    index = 0
    for building in building_cluster:
        geom = building.GetGeometryRef()
        g = geom.GetGeometryRef(0)
        for shape_idx in range(g.GetGeometryCount()):
            poly_array = poly_array_list[index]
            index = index+1
            for i in range(len(poly_array)):
                poly_array[i,0,0] = int((poly_array[i,0,0] + offsetx))
                poly_array[i,0,1] = int((poly_array[i,0,1] + offsety))

            cv2.polylines(tmp_img,[poly_array],True,(255,255,255),thickness=2)
            try:
                os.stat('../data/cluster/{}'.format(basename))
            except:
                os.makedirs('../data/cluster/{}'.format(basename))
            cv2.imwrite('../data/cluster/{}/building_cluster_{}.jpg'.\
                    format(basename,cluster_idx),tmp_img)

ds.Destroy()
sourceImage = None
