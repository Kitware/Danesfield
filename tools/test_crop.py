#!/usr/bin/env python

import os
import sys

import numpy as np

from danesfield import rpc
from danesfield import raytheon_rpc

import gdalconst
import gdalnumeric
import gdal
import ogr
import osr


def gdal_get_transform(src_image):
    geo_trans = src_image.GetGeoTransform()
    if geo_trans == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        geo_trans = gdal.GCPsToGeoTransform(src_image.GetGCPs())
    return geo_trans


def gdal_get_projection(src_image):
    projection = src_image.GetProjection()
    if projection == '':
        projection = src_image.GetGCPProjection()
    return projection


def world_to_pixel_poly(model, poly):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    first_point = True
    for p in range(poly.shape[0]):
        point = poly[p]
        proj_point = model.project(point)
        if first_point is True:
            pixel_poly = np.array(proj_point)
            first_point = False
        else:
            pixel_poly = np.vstack([pixel_poly, proj_point])
    return pixel_poly


def world_to_pixel(geoMatrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    tX = x - geoMatrix[0]
    tY = y - geoMatrix[3]
    a = geoMatrix[1]
    b = geoMatrix[2]
    c = geoMatrix[4]
    d = geoMatrix[5]
    div = 1.0 / (a * d - b * c)
    pixel = int((tX * d - tY * b) * div)
    line = int((tY * a - tX * c) * div)
    return (pixel, line)


def read_raytheon_RPC(rpc_path, img_file):
    # image file name for pattern of raytheon RPC
    file_no_ext = os.path.splitext(img_file)[0]
    rpc_file = rpc_path + 'GRA_' + file_no_ext + '.up.rpc'
    if os.path.isfile(rpc_file) is False:
        rpc_file = rpc_path + 'GRA_' + file_no_ext + '_0.up.rpc'
        if os.path.isfile(rpc_file) is False:
                    return None
    with open(rpc_file, 'r') as f:
        return raytheon_rpc.parse_raytheon_rpc_file(f)


elevation = 240
elevation_range = 100

ul_lon = -84.11236693243779
ul_lat = 39.77747025512961

ur_lon = -84.10530109439955
ur_lat = 39.77749705975315

lr_lon = -84.10511182729961
lr_lat = 39.78290042788092

ll_lon = -84.11236485416471
ll_lat = 39.78287156225952


corrected_rpc_dir = '/videonas2/fouo/data_working/CORE3D-Phase1A/AOIs/D1_WPAFB/P3D/D1_ptclds_WPAFB_museum/ba_updated_rpcs/'

src_img_file = '/videonas2/fouo/data_golden/CORE3D-Phase1A/performer_data/performer_source_data/wpafb/satellite_imagery/WV3/PAN/01JAN17WV031200017JAN01163905-P1BS-501073324070_01_P001_________AAE_0AAAAABPABM0.NTF'

file_ = '01JAN17WV031200017JAN01163905-P1BS-501073324070_01_P001_________AAE_0AAAAABPABM0.NTF'
dst_img_file = '/home/david/Desktop/test_crop/out/D1_WPAFB/test_out.tif'

src_image = gdal.Open(src_img_file, gdalconst.GA_ReadOnly)
geo_trans = gdal_get_transform(src_image)

nodata_values = []
nodata = 0
for i in range(src_image.RasterCount):
    nodata_value = src_image.GetRasterBand(i+1).GetNoDataValue()
    if not nodata_value:
        nodata_value = nodata
    nodata_values.append(nodata_value)

poly = np.array([[ul_lon, ul_lat, elevation + elevation_range],
                 [ur_lon, ur_lat, elevation + elevation_range],
                 [lr_lon, lr_lat, elevation + elevation_range],
                 [ll_lon, ll_lat, elevation + elevation_range],
                 [ul_lon, ul_lat, elevation + elevation_range],
                 [ul_lon, ul_lat, elevation - elevation_range],
                 [ur_lon, ur_lat, elevation - elevation_range],
                 [lr_lon, lr_lat, elevation - elevation_range],
                 [ll_lon, ll_lat, elevation - elevation_range],
                 [ul_lon, ul_lat, elevation - elevation_range]])

rpc_md = src_image.GetMetadata('RPC')
model = rpc.rpc_from_gdal_dict(rpc_md)

updated_rpc = read_raytheon_RPC(corrected_rpc_dir, file_)
model = updated_rpc
rpc_md = rpc.rpc_to_gdal_dict(updated_rpc)

pixel_poly = world_to_pixel_poly(model, poly)

ul_x, ul_y = map(int, pixel_poly.min(0))
lr_x, lr_y = map(int, pixel_poly.max(0))
min_x, min_y, z = poly.min(0)
max_x, max_y, z = poly.max(0)

ul_x = max(0, ul_x)
ul_y = max(0, ul_y)
lr_x = min(src_image.RasterXSize - 1, lr_x)
lr_y = min(src_image.RasterYSize - 1, lr_y)

samp_off = rpc_md['SAMP_OFF']
samp_off = float(samp_off) - ul_x
rpc_md['SAMP_OFF'] = str(samp_off)

line_off = rpc_md['LINE_OFF']
line_off = float(line_off) - ul_y
rpc_md['LINE_OFF'] = str(line_off)


# Load the source data as a gdalnumeric array
px_width = int(lr_x - ul_x + 1)
px_height = int(lr_y - ul_y + 1)

clip = src_image.ReadAsArray(ul_x, ul_y, px_width, px_height)

# create output raster
raster_band = src_image.GetRasterBand(1)
output_driver = gdal.GetDriverByName('MEM')

output_dataset = output_driver.Create(
    '', clip.shape[1], clip.shape[0],
    src_image.RasterCount, raster_band.DataType)

# Copy All metadata data from src to dst
domains = src_image.GetMetadataDomainList()
for tag in domains:
    md = src_image.GetMetadata(tag)
    if md:
        output_dataset.SetMetadata(md, tag)

# Rewrite the rpc_md that we modified above.
output_dataset.SetMetadata(rpc_md, 'RPC')
gdalnumeric.CopyDatasetInfo(src_image, output_dataset,
                            xoff=ul_x, yoff=ul_y)

# End logging, print blank line for clarity
outBand = output_dataset.GetRasterBand(1)
outBand.SetNoDataValue(nodata_values[0])
outBand.WriteArray(clip)

output_driver = gdal.GetDriverByName('GTiff')
outfile = output_driver.CreateCopy(
    dst_img_file, output_dataset, False)

# We need to write this data out
# after the CreateCopy call or it's lost
# This change seems to happen in GDAL with python 3

# Create a new geomatrix for the image
geo_trans = list(geo_trans)
geo_trans[0] = min_x
geo_trans[3] = max_y

output_dataset.SetGeoTransform(geo_trans)
gcp_crs_wkt = gdal_get_projection(src_image)
output_dataset.SetProjection(gdal_get_projection(src_image))


##################

# Add GCPs to dest
if src_image.GetGCPCount():
    outgcps = []
#    gcp_srs = osr.SpatialReference()
    gcps = src_image.GetGCPs()
    for gcp in gcps:
        if (gcp.Id == 'UpperLeft'):
            gcp = gdal.GCP(ul_lon, ul_lat, 0, 0, 0, gcp.Info, gcp.Id)
        elif (gcp.Id == 'UpperRight'):
            gcp = gdal.GCP(ur_lon, ur_lat, 0, px_width, 0, gcp.Info, gcp.Id)
        elif (gcp.Id == 'LowerRight'):
            gcp = gdal.GCP(lr_lon, lr_lat, 0, px_width, px_height, gcp.Info, gcp.Id)
        elif (gcp.Id == 'LowerLeft'):
            gcp = gdal.GCP(ll_lon, ll_lat, 0, 0, px_height, gcp.Info, gcp.Id)
        outgcps.append(gcp)
#        gcp_srs.SetFromUserInput("EPSG:4326")
#        gcp_crs_wkt = gcp_srs.ExportToWkt()
        output_dataset.SetGCPs(outgcps, gcp_crs_wkt)

####################
outfile = None
