"""
Align vector layer to the tiff image. The input image must be warped by gdalwarp first!!!!

Author: Xu Zhang
Email: xu.zhang@columbia.edu.cn
"""

import gdal, ogr, os, osr
import numpy as np
#import cv2
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
mpl.use('Agg')
import matplotlib.pyplot as plt
import plotly.plotly as py
import argparse
import Utils

from OCC.gp import gp_Pnt, gp_Vec
import OCC.GeomAPI
import OCC.BRepBuilderAPI
from OCC.TColgp import TColgp_Array1OfPnt
from OCC.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCC.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Display.SimpleGui import init_display
display, start_display, add_menu, add_function_to_menu = init_display()


draw_color = [(255,0,0,255),(255,255,0,255),(255,0,255,255),(0,255,0,255),(0,255,255,255),\
        (0,0,255,255),(255,255,255,255)]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_img', default='../data/Dayton.tiff', help='Input geotiff file')
parser.add_argument('--input_geojson', default='../data/dayton_small.geojson', help='Input osm file')
parser.add_argument('--scale' , type=float, default=0.2, help='Scale factor. We cannot deal with the images with original resolution')
parser.add_argument('--cluster_thres' , type=float, default=100, help='Distance for building clustering')
parser.add_argument('--move_thres' , type=float, default=40, help='Distance for edge matching')
parser.add_argument("-o", "--no_offset", action="store_true",
                    help="Don't align vector based on edge response(original projection)")
args = parser.parse_args()

base=os.path.basename(args.input_geojson)

ds = ogr.Open(args.input_geojson)
layer = ds.GetLayerByIndex(0)
nameList = []

draw_flag = False
index = 0
building_list = []
for feature in layer:
    bottom = feature.GetField("BASEELEV_M")
    base_bottom = feature.GetField("BASE_M")
    top = feature.GetField("TOPELEV_M")
    #print(feature.GetField("ID"))
    geon = feature.GetGeometryRef()
    polygon = geon.GetGeometryRef(0)
    num_point = polygon.GetPointCount()
    poly = OCC.BRepBuilderAPI.BRepBuilderAPI_MakePolygon()
    #array = TColgp_Array1OfPnt(1, num_point)
    for i in range(0, num_point):
        pt = polygon.GetPoint(i)
        poly.Add(gp_Pnt((pt[0]+84.05)*200, (pt[1]-39.78)*200, (bottom-270)/360.0))
        #print(pt[0],pt[1])
    
    poly.Build()
    poly.Close()
    wire = poly.Wire()
    face = BRepBuilderAPI_MakeFace(wire).Face()

    # the linear path
    print(bottom)
    starting_point = gp_Pnt(0., 0., (bottom-270)/360.)

    end_point = gp_Pnt(0., 0., (top-270)/360.)
    vec = gp_Vec(starting_point, end_point)
    path = BRepBuilderAPI_MakeEdge(starting_point, end_point).Edge()
    index = index + 1
    # extrusion
    #prism = BRepPrimAPI_MakePrism(profile, vec).Shape()
    prism = BRepPrimAPI_MakePrism(face, vec, True).Shape()

    display.DisplayShape(face, update=False)
    #display.DisplayShape(starting_point, update=False)
    #display.DisplayShape(end_point, update=False)
    #display.DisplayShape(path, update=False)
    display.DisplayShape(prism, update=True)

start_display()
    #if feature.GetField("building") != None:  # only streets
    #    flag = True
    #    #print("haha")
    #    name = feature.GetField("name")
    #    geom = feature.GetGeometryRef()
    #    g = geom.GetGeometryRef(0)
    #    for shape_idx in range(g.GetGeometryCount()):
    #        polygon = g.GetGeometryRef(shape_idx)
    #        for i in range(0, polygon.GetPointCount()):
    #            pt = polygon.GetPoint(i)
    #            #print(pt[0],pt[1])
    #            if pt[0]>left and pt[0]<right and pt[1]<top and pt[1]>bottom:
    #                pass
    #            else:
    #                flag = False
    #                break
    #        if not flag:
    #            break
    #    if flag:
    #        building_list.append(feature)
