import gdal, ogr, os, osr
import numpy as np
import cv2
import argparse
import Utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_img', default='../data/Dayton.ntf', help='folder to output log')
args = parser.parse_args()

base=os.path.basename(args.input_img)
basename = os.path.splitext(base)[0]

input_img = args.input_img
sourceImage = gdal.Open(input_img, gdal.GA_ReadOnly)
rpcMetaData = sourceImage.GetMetadata('RPC')
image = np.zeros((sourceImage.RasterYSize,sourceImage.RasterXSize,3))
r = np.array(sourceImage.GetRasterBand(1).ReadAsArray())
g = np.array(sourceImage.GetRasterBand(2).ReadAsArray())
b = np.array(sourceImage.GetRasterBand(3).ReadAsArray())

image[:,:,0] = b
image[:,:,1] = g
image[:,:,2] = r

cv2.imwrite('../data/{}.jpg'.format(basename), image)

