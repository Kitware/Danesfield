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


def ProjectPoint(model, pt):
    px = int((pt[0]-model['corners'][0])/model['project_model'][1]*model['scale'])
    py = int((pt[1]-model['corners'][1])/model['project_model'][5]*model['scale'])
    return [px,py]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_img', default='../data/test_warped.tiff', help='folder to output log')
parser.add_argument('--input_osm', default='../data/map.osm', help='folder to output log')
parser.add_argument('--scale' , type=float, default=0.2, help='beta hyperparameter value')

args = parser.parse_args()

input_img = args.input_img
scale = args.scale

color_image = cv2.imread(input_img) 
print(color_image.shape)
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
myarray = np.zeros((sourceImage.RasterYSize,sourceImage.RasterXSize))

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
            if flag:
                poly_point_list = []
                tmp_img = np.zeros((int(color_image.shape[0]),\
                    int(color_image.shape[1])))

                for i in range(0, polygon.GetPointCount()):
                    pt = polygon.GetPoint(i)
                    poly_point_list.append(ProjectPoint(model,pt))

                poly_array = np.array(poly_point_list)
                poly_array = poly_array.reshape((-1,1,2))
                cv2.polylines(tmp_img,[poly_array],True,(255),thickness=2)
                #cv2.imshow('image',tmp_img)
                #cv2.imwrite('contour.jpg',tmp_img)
                #cv2.waitKey(0)
                check_point_list = []

                for y in range(0,tmp_img.shape[0]):
                    for x in range(0,tmp_img.shape[1]):
                        if tmp_img[y,x] > 200:
                            check_point_list.append([x,y])

                print(len(check_point_list))

                max_value = 0
                offsetx = 0
                offsety = 0
                for dy in range(-40,40,2):
                    for dx in range(-40,40,2):
                        total_value = 0
                        for pt in check_point_list:
                            if edge_img[pt[1]+dy,pt[0]+dx] > 200:
                                total_value += 1
                                if total_value>max_value:
                                    max_value = total_value
                                    offsetx = dx
                                    offsety = dy
                print(max_value)
                print(offsetx, offsety)
                for i in range(len(poly_array)):
                    poly_array[i,0,0] = int((poly_array[i,0,0] + offsetx))
                    poly_array[i,0,1] = int((poly_array[i,0,1] + offsety))

                cv2.polylines(tmp_img,[poly_array],True,(255,255,255),thickness=2)
                cv2.imwrite('new_contour.jpg',tmp_img)
                cv2.imshow('image',tmp_img)
                cv2.waitKey(0)

                #for i in range(len(poly_array)):
                    #poly_array[i,0,0] = int((poly_array[i,0,0] + offsetx)/scale)
                    #poly_array[i,0,1] = int((poly_array[i,0,1] + offsety)/scale)
                #cv2.fillPoly(myarray,[poly_array],True,(0))

                draw_flag = True
    #if draw_flag:
    #    break
                
#print(nameList)
ds.Destroy()
#cv2.imwrite('../data/mask.png',myarray)
#imgplot = plt.imshow(myarray)
print(myarray.shape)
print(myarray.dtype)
sourceImage = None
