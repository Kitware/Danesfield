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
import pdb
import Utils

draw_color = [(255,0,0,255),(255,255,0,255),(255,0,255,255),(0,255,0,255),(0,255,255,255),\
        (0,0,255,255),(255,255,255,255)]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_img', default='../data/Jacksonville/Jacksonville.tiff', help='Input geotiff file')
#parser.add_argument('--input_osm', default='../data/Jacksonville/BLD2/jacksonville_2d_bldgs_2.shp', help='Input osm file')
parser.add_argument('--input_osm', default='../data/Jacksonville/Jacksonville.osm', help='Input osm file')
#parser.add_argument('--input_img', default='../data/Dayton.tiff', help='Input geotiff file')
#parser.add_argument('--input_osm', default='../data/Dayton_map.osm', help='Input osm file')
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
print(np.amax(color_image))
print(color_image.shape)
color_image = np.transpose(np.array([color_image,color_image,color_image]),(1,2,0))
print(color_image.shape)
