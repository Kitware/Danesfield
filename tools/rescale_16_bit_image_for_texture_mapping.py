#!/usr/bin/env python

import argparse
import logging
import sys
import gdal
import gdalconst
import numpy as np
import cv2

""" This script rescales the values of a 16-bit image
between 0-255 without changing its bit depth """


def copy_tif_metadata_from_file_to_file(filename_in, filename_out):
    """ copy TIF metadata from an image to another """
    print("Copy TIF information")
    img_in = gdal.Open(filename_in)
    img_out = gdal.Open(filename_out, gdal.GA_Update)
    metadata_domains = img_in.GetMetadataDomainList()
    for domain in metadata_domains:
        dico = img_in.GetMetadata_Dict(domain)
        for key, val in dico.items():
            img_out.SetMetadataItem(key, val, domain)
    img_out.SetProjection(img_in.GetProjection())
    img_out.SetGeoTransform(img_in.GetGeoTransform())
    del img_out


def rescale_image(image_in, image_out):
    """ pansharpen images from a directory """
    print(image_in)
    img = gdal.Open(image_in, gdalconst.GA_Update)

    width = img.RasterXSize
    height = img.RasterYSize
    depth = img.RasterCount

    bands = np.zeros((height, width, depth), dtype=np.uint16)
    for i in range(depth):
        band = img.GetRasterBand(i+1).ReadAsArray(0, 0, width, height,
                                                  buf_type=gdal.GDT_UInt16)
        mean, std = cv2.meanStdDev(band)
        band = band.astype(float)
        min = 0
        max = mean + 2.5 * std      # this scaling seems to give a visually good result
        band -= min
        band /= (max-min)
        band *= 255.0
        bands[:, :, i] = band.astype(np.uint16)
        bands = np.clip(bands, 0, 255)

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(image_out, width, height, depth, gdal.GDT_UInt16)
    for i in range(depth):
        outdata.GetRasterBand(i+1).WriteArray(bands[:, :, i])
        outdata.GetRasterBand(i+1).SetNoDataValue(0)
    outdata.FlushCache()
    outdata = None
    copy_tif_metadata_from_file_to_file(image_in, image_out)


def main(args):
    parser = argparse.ArgumentParser(description="Rescale a 16-bit image between 0-255 "
                                     "but do not change the image bit depth")
    parser.add_argument("image_in", help="Image to rescale")
    parser.add_argument("image_out", help="Output image")
    args = parser.parse_args(args)
    rescale_image(args.image_in, args.image_out)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
