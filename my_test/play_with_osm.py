import gdal, ogr, os, osr
import numpy as np

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
mpl.use('Agg')
import matplotlib.pyplot as plt
import plotly.plotly as py

osmFileName = '../data/map.osm'

ds = ogr.Open(osmFileName)
print('#Layers: {}'.format(ds.GetLayerCount()))
layerList = []
for i in ds:
    daLayer = i.GetName()
    if not daLayer in layerList:
        layerList.append(daLayer)

layerList.sort()

layer = ds.GetLayerByIndex(3) # layer 1 for ways
nameList = []
for feature in layer:
    if feature.GetField("tourism") != None:  # only streets
        #print("haha")
        name = feature.GetField("name")
        geom = feature.GetGeometryRef()
        g = geom.GetGeometryRef(0)
        for shape_idx in range(g.GetGeometryCount()):
            polygon = g.GetGeometryRef(shape_idx)
            print(polygon.GetGeometryName())
            print(polygon.GetPointCount())
            #for i in range(0, polygon.GetPointCount()):
            #    pt = polygon.GetPoint(i)
            #    print("{}). POINT {}".format(i, pt))
        if name != None and name not in nameList: # only streets that have a name and are not yet in the list
            nameList.append(name)

print(nameList)
ds.Destroy()
