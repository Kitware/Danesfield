import gdal, ogr, os, osr
import numpy as np
import cv2
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
mpl.use('Agg')
import matplotlib.pyplot as plt
import plotly.plotly as py

imageFileName = '../data/result.jpg'

color_image = cv2.imread(imageFileName) 
print(color_image.shape)
grayimg = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(grayimg, 100, 200)
cv2.imwrite('../data/edge.jpg',edge)
#imgplot = plt.imshow(myarray)
sourceImage = None
