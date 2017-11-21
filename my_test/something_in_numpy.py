import gdal, ogr, os, osr
import numpy as np
import cv2
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
mpl.use('Agg')
import matplotlib.pyplot as plt
import plotly.plotly as py

imageFileName = '../data/test_warped.tiff'

# open the GDAL file
sourceImage = gdal.Open(imageFileName, gdal.GA_ReadOnly)
rpcMetaData = sourceImage.GetMetadata('RPC')
myarray = np.array(sourceImage.GetRasterBand(1).ReadAsArray())
myarray = myarray*0+255

gt = sourceImage.GetGeoTransform() # captures origin and pixel size
print('Origin:', (gt[0], gt[3]))
print('Pixel size:', (gt[1], gt[5]))

gdal.ApplyGeoTransform(gt,0,0)

left = gdal.ApplyGeoTransform(gt,0,0)[0]
top = gdal.ApplyGeoTransform(gt,0,0)[1]
right = gdal.ApplyGeoTransform(gt,sourceImage.RasterXSize,sourceImage.RasterYSize)[0]
bottom = gdal.ApplyGeoTransform(gt,sourceImage.RasterXSize,sourceImage.RasterYSize)[1]

print(left, top, right, bottom)
osmFileName = '../data/map.osm'
#osmFileName = '../data/state/cb_2016_us_state_20m.shp'

ds = ogr.Open(osmFileName)
layer = ds.GetLayerByIndex(3)

nameList = []
for feature in layer:
    if feature.GetField("tourism") != None or feature.GetField("building") != None:  # only streets
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
                for i in range(0, polygon.GetPointCount()):
                    pt = polygon.GetPoint(i)
                    poly_point_list.append([int((pt[0]-left)/gt[1]), int((pt[1]-top)/gt[5])])
                poly_array = np.array(poly_point_list)
                poly_array = poly_array.reshape((-1,1,2))
                cv2.fillPoly(myarray,[poly_array],True,(0))
                print(name)

#print(nameList)
ds.Destroy()
cv2.imwrite('../data/result.png',myarray)
#imgplot = plt.imshow(myarray)
print(myarray.shape)
print(myarray.dtype)
sourceImage = None
