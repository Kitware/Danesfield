#!/usr/bin/env python

"""
Crops and pansharpens an image for texture mapping.
This is part of the pipeline to process an AOI from start to finish
"""

import argparse
import crop_images
from danesfield import gdal_utils
import gdal
import logging
import os
import preprocess_images_for_texture_mapping
import subprocess
import sys


def copy_tif_info(in_fpath, out_fpath):
    """
    Copy the TIF metadata from in_fpath and save it to the image at out_fpath.
    :param in_fpath: Filepath for input image which has metadata to copy.
    :type in_fpath: str

    :param out_fpath: Filepath for image to which we want to save the metadata.
    :type out_fpath: str
    """
    try:
        logging.info("---- Copying TIF information ----")
        gdal_in = gdal_utils.gdal_open(in_fpath)
        gdal_out = gdal_utils.gdal_open(out_fpath, gdal.GA_Update)
        metadata_domains = gdal_in.GetMetadataDomainList()
        for domain in metadata_domains:
            dico = gdal_in.GetMetadata_Dict(domain)
            for key, val in dico.items():
                gdal_out.SetMetadataItem(key, val, domain)
        return True
    except Exception as e:
        logging.exception(e)
        return False


def crop_and_pansharpen(dsm_file, files, output_dir):
    '''
    Crops and MSI and PAN image and pansharpens the MSI

    For files, the following keys are possible:
    pan, msi
        image, rpc, crop_img_fpath, 'crop_pansharpened_processed_fpath'
    '''
    for type in ('pan', 'msi'):
        # Crop images
        ntf_fpath = files[type]['image']
        fname = os.path.splitext(os.path.split(ntf_fpath)[1])[0]
        crop_img_fpath = os.path.join(output_dir, '{}_crop.tif'.format(fname))
        cmd_args = [dsm_file, output_dir, ntf_fpath, "--dest_file_postfix", "_crop"]

        rpc_fpath = files[type].get('rpc', None)
        if rpc_fpath:
            cmd_args.extend(['--rpc_dir', rpc_fpath])

        script_call = ["crop_images.py"] + cmd_args
        print(*script_call)
        crop_images.main(cmd_args)
        files[type]['crop_img_fpath'] = crop_img_fpath

    # Pansharpen the cropped images
    pan_ntf_fpath = files['pan']['image']
    pan_fname = os.path.splitext(os.path.split(pan_ntf_fpath)[1])[0]
    crop_pansharpened_image = os.path.join(
        output_dir, '{}_crop_pansharpened.tif'.format(pan_fname))
    cmd_args = ['gdal_pansharpen.py', files['pan']['crop_img_fpath'],
                files['msi']['crop_img_fpath'],
                crop_pansharpened_image]
    print(*cmd_args)
    subprocess.run(cmd_args)

    # copy tif metadata such as RPC from input to output images
    if copy_tif_info(files['pan']['crop_img_fpath'], crop_pansharpened_image):
        # Pre-process images for texture mapping to match the format expected by the C++ code
        # and to have visually nice textures
        cmd_args = [crop_pansharpened_image, ".", "--dest_file_postfix", "_processed"]
        name, ext = os.path.splitext(crop_pansharpened_image)
        preprocess_images_for_texture_mapping.main(cmd_args)


def main(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dsm", help="Digital surface model (DSM) image file name")
    parser.add_argument("output_dir",
                        help="Output directory for cropped pansharpened images "
                             "(*_crop_pansharpened_processed.tif)")
    parser.add_argument("--pan", help="PAN image file name and optional RPC for it",
                        nargs="+", required=True)
    parser.add_argument("--msi", help="MSI image file name and optional RPC for it",
                        nargs="+", required=True)
    args = parser.parse_args(args)
    files = {}
    files['pan'] = {}
    files['msi'] = {}

    files['pan']['image'] = args.pan[0]
    if (len(args.pan) > 1):
        files['pan']['rpc'] = args.pan[1]
    files['msi']['image'] = args.msi[0]
    if (len(args.msi) > 1):
        files['msi']['rpc'] = args.msi[1]

    crop_and_pansharpen(args.dsm, files, args.output_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
