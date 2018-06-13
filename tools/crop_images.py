#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#  Copyright 2017 by Kitware Inc.
#
#  Licensed under the Apache License, Version 2.0 ( the "License" );
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import os
import argparse
import logging

import numpy as np

from danesfield import rpc
from danesfield import raytheon_rpc

import gdalconst
import gdal


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


def main(args):
    parser = argparse.ArgumentParser(
        description="Crop out images for each of the CORE3D AOIs")
    parser.add_argument("aoi",
                        choices=['D1', 'D2', 'D3', 'D4'],
                        help="dataset_AOI options: D1 (WPAFB), D2 (WPAFB), "
                             "D3 (USCD), D4 (Jacksonville)")
    parser.add_argument("src-root",
                        help="Source imagery root directory")
    parser.add_argument("dest-dir",
                        help="Destination directory for writing crops")
    parser.add_argument("--rpc-dir",
                        help="Source directory for RPC text files")
    args = parser.parse_args(args)

    AOI = args.aoi[1]
    src_root_dir = os.path.join(args.src_root, '')
    dst_root_dir = os.path.join(args.dest_root, '')

    print('Cropping images from: ' + src_root_dir)
    print('Locating crops in directory: ' + dst_root_dir)
    print('Cropping to AOI: ' + AOI)

    if args.rpc_dir:
        corrected_rpc_dir = os.path.join(args.rpc_dir, '')
        print('Using update RPCs from: ' + corrected_rpc_dir)
    else:
        print('Not using updated RPCs')

    elevation_range = 100

    # WPAFB AOI D1
    if AOI == 'D1':
        elevation = 240
        ul_lon = -84.11236693243779
        ul_lat = 39.77747025512961

        ur_lon = -84.10530109439955
        ur_lat = 39.77749705975315

        lr_lon = -84.10511182729961
        lr_lat = 39.78290042788092

        ll_lon = -84.11236485416471
        ll_lat = 39.78287156225952

    # WPAFB AOI D2
    if AOI == 'D2':
        elevation = 300
        ul_lon = -84.08847226672408
        ul_lat = 39.77650841377968

        ur_lon = -84.07992142333644
        ur_lat = 39.77652166058358

        lr_lon = -84.07959205694203
        lr_lat = 39.78413758747398

        ll_lon = -84.0882028871317
        ll_lat = 39.78430009793551

    # UCSD AOI D3
    if AOI == 'D3':
        elevation = 120
        ul_lon = -117.24298768132505
        ul_lat = 32.882791370856857

        ur_lon = -117.24296375496185
        ur_lat = 32.874021450913411

        lr_lon = -117.2323749640905
        lr_lat = 32.874041569804469

        ll_lon = -117.23239784772379
        ll_lat = 32.882811496466012

    # Jacksonville AOI D4
    if AOI == 'D4':
        elevation = 2
        ul_lon = -81.67078466333165
        ul_lat = 30.31698808384777

        ur_lon = -81.65616946309449
        ur_lat = 30.31729872444624

        lr_lon = -81.65620275072482
        lr_lat = 30.329923847788603

        ll_lon = -81.67062242425624
        ll_lat = 30.32997669492018

    for root, dirs, files in os.walk(src_root_dir):
        for file_ in files:
            new_root = root.replace(src_root_dir, dst_root_dir)
            if not os.path.exists(new_root):
                os.makedirs(new_root)
            ext = os.path.splitext(file_)[-1].lower()
            if ext != ".ntf":
                continue

            src_img_file = os.path.join(root, file_)
            dst_img_file = os.path.join(new_root, file_)
            dst_file_no_ext = os.path.splitext(dst_img_file)[0]
            dst_img_file = dst_file_no_ext + ".tif"

            print('Converting img: ' + src_img_file)
            src_image = gdal.Open(src_img_file, gdalconst.GA_ReadOnly)

            nodata_values = []
            nodata = 0
            for i in range(src_image.RasterCount):
                nodata_value = src_image.GetRasterBand(i+1).GetNoDataValue()
                if not nodata_value:
                    nodata_value = nodata
                nodata_values.append(nodata_value)

            # +- elevation_range
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
            if (corrected_rpc_dir):
                updated_rpc = read_raytheon_RPC(corrected_rpc_dir, file_)
                if updated_rpc is None:
                    print('No RPC file exists for image file, skipping: ' +
                          src_img_file + '\n')
                    continue
                else:
                    model = updated_rpc
                    rpc_md = rpc.rpc_to_gdal_dict(updated_rpc)

            # Project the world point locations into the image
            pixel_poly = model.project(poly)

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

            model.image_offset[0] -= ul_x
            model.image_offset[1] -= ul_y

            # Calculate the pixel size of the new image
            # Constrain the width and height to the bounds of the image
            px_width = int(lr_x - ul_x + 1)
            if px_width + ul_x > src_image.RasterXSize - 1:
                px_width = int(src_image.RasterXSize - ul_x - 1)

            px_height = int(lr_y - ul_y + 1)
            if px_height + ul_y > src_image.RasterYSize - 1:
                px_height = int(src_image.RasterYSize - ul_y - 1)

            # We've constrained x & y so they are within the image.
            # If the width or height ends up negative at this point,
            # the AOI is completely outside the image
            if px_width < 0 or px_height < 0:
                print('AOI out of range, skipping\n')
                continue

            corners = [[0, 0], [px_width, 0], [px_width, px_height], [0, px_height]]
            corner_names = ['UpperLeft', 'UpperRight', 'LowerRight', 'LowerLeft']
            world_corners = model.back_project(corners, elevation)

            corner_gcps = []
            for (p, l), (x, y, h), n in zip(corners, world_corners, corner_names):
                corner_gcps.append(gdal.GCP(x, y, h, p, l, "", n))

            # Load the source data as a gdalnumeric array
            clip = src_image.ReadAsArray(ul_x, ul_y, px_width, px_height)

            # create output raster
            raster_band = src_image.GetRasterBand(1)
            output_driver = gdal.GetDriverByName('MEM')

            # In the event we have multispectral images,
            # shift the shape dimesions we are after,
            # since position 0 will be the number of bands
            try:
                clip_shp_0 = clip.shape[0]
                clip_shp_1 = clip.shape[1]
                if clip.ndim > 2:
                    clip_shp_0 = clip.shape[1]
                    clip_shp_1 = clip.shape[2]
            except (AttributeError):
                print('Error decoding image, skipping\n')
                continue

            output_dataset = output_driver.Create(
                '', clip_shp_1, clip_shp_0,
                src_image.RasterCount, raster_band.DataType)

            # Copy All metadata data from src to dst
            domains = src_image.GetMetadataDomainList()
            for tag in domains:
                md = src_image.GetMetadata(tag)
                if md:
                    output_dataset.SetMetadata(md, tag)

            # Rewrite the rpc_md that we modified above.
            output_dataset.SetMetadata(rpc_md, 'RPC')
            output_dataset.SetGeoTransform(gdal.GCPsToGeoTransform(corner_gcps))
            output_dataset.SetProjection(gdal_get_projection(src_image))

            # End logging, print blank line for clarity
            print('')
            bands = src_image.RasterCount
            if bands > 1:
                for i in range(bands):
                    outBand = output_dataset.GetRasterBand(i + 1)
                    outBand.SetNoDataValue(nodata_values[i])
                    outBand.WriteArray(clip[i])
            else:
                outBand = output_dataset.GetRasterBand(1)
                outBand.SetNoDataValue(nodata_values[0])
                outBand.WriteArray(clip)

            if dst_img_file:
                output_driver = gdal.GetDriverByName('GTiff')
                output_driver.CreateCopy(
                    dst_img_file, output_dataset, False)


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
