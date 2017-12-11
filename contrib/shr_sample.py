import gdal, ogr, os, osr
import numpy as np

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
mpl.use('Agg')
import matplotlib.pyplot as plt
import plotly.plotly as py

osmFileName = '../data/Jacksonville/BLD2/jacksonville_2d_bldgs_2.shp'

ds = ogr.Open(osmFileName)
print('#Layers: {}'.format(ds.GetLayerCount()))
layerList = []
for i in ds:
    daLayer = i.GetName()
    if not daLayer in layerList:
        layerList.append(daLayer)

layerList.sort()

layer = ds.GetLayerByIndex(0) # layer 1 for polygon
nameList = []
for feature in layer:
    #print(feature)
    #break
    #if feature.GetField("building") != None:  # only buildings
    #    name = feature.GetField("name")
    geom = feature.GetGeometryRef()
    print(geom)
    g = geom.GetGeometryRef(0)
    print(g)
    print(g.GetGeometryName())
    print(g.GetGeometryCount())
    print(g.GetPointCount())
    for shape_idx in range(g.GetGeometryCount()):
        polygon = g.GetGeometryRef(shape_idx)
        print(polygon.GetGeometryName())
        print(polygon.GetPointCount())
    break
    #    if name != None and name not in nameList:
    #        nameList.append(name)

print(nameList)
ds.Destroy()
