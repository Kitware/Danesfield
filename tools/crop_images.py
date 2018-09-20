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
from danesfield import gdal_utils

import gdalconst
import gdal

import pyproj
import sys


def gdal_get_projection(src_image):
    projection = src_image.GetProjection()
    if projection == '':
        projection = src_image.GetGCPProjection()
    return projection


def read_raytheon_RPC(rpc_path, img_file):
    file_no_ext = os.path.splitext(os.path.basename(img_file))[0]
    if type(rpc_path) is list:
        rpc_file = [f for f in rpc_path if f.find(file_no_ext) > 0][0]
        if rpc_file is None or os.path.isfile(rpc_file) is False:
            return None
    else:
        rpc_file = rpc_path + 'GRA_' + file_no_ext + '.up.rpc'
        if os.path.isfile(rpc_file) is False:
            rpc_file = rpc_path + 'GRA_' + file_no_ext + '_0.up.rpc'
            if os.path.isfile(rpc_file) is False:
                        return None
    with open(rpc_file, 'r') as f:
        return raytheon_rpc.parse_raytheon_rpc_file(f)


def filesFromArgs(src_root, dest_dir, dest_file_postfix=''):
    if type(src_root) is list:
        for src_file in src_root:
            name_ext = os.path.basename(src_file)
            name, ext = os.path.splitext(name_ext)
            yield src_file, dest_dir + "/" + name + dest_file_postfix + ext
    else:
        for root, dirs, files in os.walk(src_root):
            for file_ in files:
                new_root = root.replace(src_root, dest_dir)
                if not os.path.exists(new_root):
                    os.makedirs(new_root)
                name, ext = os.path.splitext(file_)
                if ext.lower() != ".ntf":
                    continue

                src_img_file = os.path.join(root, file_)
                dst_img_file = os.path.join(new_root, name + dest_file_postfix + ext)
                yield src_img_file, dst_img_file


def main(args):
    parser = argparse.ArgumentParser(
        description="Crop out images for each of the CORE3D AOIs")
    parser.add_argument("aoi",
                        help="dataset AOI: D1 (WPAFB), D2 (WPAFB), "
                             "D3 (USCD), D4 (Jacksonville) or "
                             "DSM used to get the cropping bounds and elevation")
    parser.add_argument("dest_dir",
                        help="Destination directory for writing crops. Crop files have "
                             "the same name as source images + an optional postifx.")
    parser.add_argument("src_root",
                        help="Source imagery root directory or list of images",
                        nargs="+")
    parser.add_argument("--dest_file_postfix",
                        help="Postfix added to destination files, before the extension")
    parser.add_argument("--rpc_dir",
                        help="Source directory for RPCs or list of RPC files",
                        nargs="+")
    args = parser.parse_args(args)

    useDSM = False
    if os.path.isfile(args.aoi):
        useDSM = True
    if os.path.isfile(args.src_root[0]):
        src_root = args.src_root
        print('Cropping a list of {} images'.format(len(args.src_root)))
    else:
        src_root = os.path.join(args.src_root[0], '')
        print('Cropping all images from directory: {}'.format(args.src_root))

    dest_dir = os.path.join(args.dest_dir, '')
    if (not args.dest_file_postfix):
        dest_file_postfix = "_crop"
    else:
        dest_file_postfix = args.dest_file_postfix

    print('Writing crops in directory: ' + dest_dir)
    print('Cropping to AOI: ' + args.aoi)

    rpc_dir = None
    if args.rpc_dir:
        if os.path.isfile(args.rpc_dir[0]):
            rpc_dir = args.rpc_dir
            print('Using a list of {} RPCs.'.format(len(args.rpc_dir)))
        else:
            rpc_dir = os.path.join(args.rpc_dir[0], '')
            print("Using all RPCs from directory: {}".format(rpc_dir))
    else:
        print('Using RPCs from image metadata.')

    if useDSM:
        data_dsm = gdal_utils.gdal_open(args.aoi)
        # Elevation
        dsm = data_dsm.GetRasterBand(1).ReadAsArray(
            0, 0, data_dsm.RasterXSize, data_dsm.RasterYSize, buf_type=gdal.GDT_Float32)
        no_data_value = data_dsm.GetRasterBand(1).GetNoDataValue()
        dsm_without_no_data = dsm[dsm != no_data_value]
        elevations = np.array([[dsm_without_no_data.min()], [dsm_without_no_data.max()]])
        # Cropping bounds from DSM
        [minX, minY, maxX, maxY] = gdal_utils.gdal_bounding_box(
            data_dsm, pyproj.Proj('+proj=longlat +datum=WGS84'))
        latlong_corners = np.array(
            [[minX, minY, elevations[0]],
             [maxX, minY, elevations[0]],
             [maxX, maxY, elevations[0]],
             [minX, maxY, elevations[0]],
             [minX, minY, elevations[1]],
             [maxX, minY, elevations[1]],
             [maxX, maxY, elevations[1]],
             [minX, maxY, elevations[1]]])
        print("Cropping bounds extracted from DSM")
    else:
        elevation_range = 100
        # WPAFB AOI D1
        if args.aoi == 'D1':
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
        if args.aoi == 'D2':
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
        if args.aoi == 'D3':
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
        if args.aoi == 'D4':
            elevation = 2
            ul_lon = -81.67078466333165
            ul_lat = 30.31698808384777

            ur_lon = -81.65616946309449
            ur_lat = 30.31729872444624

            lr_lon = -81.65620275072482
            lr_lat = 30.329923847788603

            ll_lon = -81.67062242425624
            ll_lat = 30.32997669492018

    for src_img_file, dst_img_file in filesFromArgs(src_root, dest_dir, dest_file_postfix):
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

        if useDSM:
            poly = latlong_corners.copy()
            elevation = np.median(dsm)
        else:
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
        if rpc_dir:
            print("Using file RPC: {}".format(rpc_dir))
            model = read_raytheon_RPC(rpc_dir, src_img_file)
            if model is None:
                print('No RPC file exists using image metadata RPC: ' +
                      src_img_file + '\n')
                rpc_md = src_image.GetMetadata('RPC')
                model = rpc.rpc_from_gdal_dict(rpc_md)
            else:
                rpc_md = rpc.rpc_to_gdal_dict(model)
        else:
            print("Using image RPC.")
            rpc_md = src_image.GetMetadata('RPC')
            model = rpc.rpc_from_gdal_dict(rpc_md)

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
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
