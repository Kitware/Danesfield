#!/usr/bin/env python

import argparse
import glob
import logging
import os
import sys
import gdal
import gdalconst
import numpy as np

import to_8bit

""" This script preprocesses the images in order to make them compatible with the texture mapping
    and to have nice visual results. It rescales the values of 16-bit images between 0-255
    with a percentile range to ignore the min and max values and add empty bands if needed to get
    8 bands with RGB at positions 5 3 2. """


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


def process_image(image_in, image_out):
    """ process image """

    # rescale bands values between 0-255
    temp_8bit_image = os.path.splitext(image_out)[0] + "_8bit_temp.tif"
    to_8bit.main([image_in, temp_8bit_image, "-p", str(0.0), str(4.0)])

    # convert again to 16-bit with 8 bands
    img = gdal.Open(temp_8bit_image, gdalconst.GA_Update)
    width, height, depth = img.RasterXSize, img.RasterYSize, img.RasterCount
    bands = np.zeros((height, width, depth), dtype=np.uint16)
    for i in range(depth):
        band = img.GetRasterBand(i+1).ReadAsArray(0, 0, width, height, buf_type=gdal.GDT_Byte)
        bands[:, :, i] = band.astype(np.uint16)

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(image_out, width, height, 8, gdal.GDT_UInt16)
    if depth == 8:
        for i in range(depth):
            outdata.GetRasterBand(i+1).WriteArray(bands[:, :, i])
    elif depth == 4:
        # if the input image has only 4 bands, we copy RGB from 3 2 1 -> 5 3 2
        # and the other bands are empty
        empty_band = np.zeros_like(bands[:, :, 0])
        outdata.GetRasterBand(1).WriteArray(empty_band)
        outdata.GetRasterBand(2).WriteArray(bands[:, :, 0])
        outdata.GetRasterBand(3).WriteArray(bands[:, :, 1])
        outdata.GetRasterBand(4).WriteArray(empty_band)
        outdata.GetRasterBand(5).WriteArray(bands[:, :, 2])
        outdata.GetRasterBand(6).WriteArray(empty_band)
        outdata.GetRasterBand(7).WriteArray(empty_band)
        outdata.GetRasterBand(8).WriteArray(empty_band)
    for i in range(8):
        outdata.GetRasterBand(i+1).SetNoDataValue(0)
    outdata.FlushCache()
    outdata = None

    # copy metadata from input to output
    copy_tif_metadata_from_file_to_file(image_in, image_out)

    # remove temporary image
    os.remove(temp_8bit_image)


def filesFromArgs(src_dir, dest_dir, dest_file_postfix):
    if os.path.isfile(src_dir[0]):
        for src_file in src_dir:
            name_ext = os.path.basename(src_file)
            name, ext = os.path.splitext(name_ext)
            yield src_file, dest_dir + "/" + name + dest_file_postfix + ext
    else:
        for src_file in glob.glob(os.path.join(src_dir, "*.tif")):
            name_ext = os.path.basename(src_file)
            name, ext = os.path.splitext(name_ext)
            dest_file = os.path.join(dest_dir, name + dest_file_postfix + ext)
            yield src_file, dest_file


def main(args):
    parser = argparse.ArgumentParser(description='Preprocess images before texture mapping')
    parser.add_argument("input_dir", help="Input directory or list of files",
                        nargs="+")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--dest_file_postfix",
                        help="Postfix added to destination files, before the extension")
    args = parser.parse_args(args)
    input_dir = args.input_dir
    output_dir = args.output_dir

    if (not args.dest_file_postfix):
        dest_file_postfix = "_processed"
    else:
        dest_file_postfix = args.dest_file_postfix

    # create output directory if needed
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # iterate over all .tif images from the input directory
    for input_file, output_file in filesFromArgs(input_dir, output_dir, dest_file_postfix):
        process_image(input_file, output_file)
        print(input_file)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
