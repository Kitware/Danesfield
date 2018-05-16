from danesfield import ortho
from danesfield import gdal_utils

import argparse
import gdal
import glob
import numpy
import os.path
import re
import sys


def orthoParamsToString(args_source_image, args_dsm, args_destination_image,
                        args_occlusion_thresh, args_denoise_radius, args_raytheon_rpc):
    ret = "orthorectify.py " + args_source_image + " " + args_dsm +\
      " " + args_destination_image
    if args_occlusion_thresh is not None:
        ret = ret + " -t " + str(args_occlusion_thresh)
    if args_denoise_radius is not None:
        ret = ret + " -d " + str(args_denoise_radius)
    ret = ret + " --raytheon-rpc " + args_raytheon_rpc if args_raytheon_rpc else ret
    return ret


def intersection(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x1<x2 and y1<y2:
        return [x1, y1, x2, y2]
    # else return None

parser = argparse.ArgumentParser(
    description='Orthorectify a list of images to cover all DSMs. '
    'The image chosen is the one with maximum coverage of the DSM and '
    'the smallest NITF_CSEXRA_OBLIQUITY_ANGLE (off-nadir angle). The output has the '
    'same name as the source image with postfix or_pan (if image_folders '
    'contain PAN) or or_msi')
parser.add_argument("dsm_folder",
                    help="Folder for all DSMs which follow name_ii_jj.tif pattern, "
                    "where ii and jj are indexes (00, 01, ...)")
parser.add_argument("image_folders", nargs="+", help="Source image folders")
parser.add_argument('-t', "--occlusion_thresh", type=float, default=1.0,
                    help="Threshold on height difference for detecting "
                    "and masking occluded regions (in meters)")
parser.add_argument('-d', "--denoise_radius", type=float, default=2,
                    help="Apply morphological operations with this radius "
                    "to the DSM reduce speckled noise")
parser.add_argument("--rpc_folder", type=str,
                    help="Raytheon RPC folder. If not provided, "
                    "the RPC is read from the source_image")
parser.add_argument("--dense_ids", type=str,
                    help="Process only the DSMs with the specified IDs. "
                    "IDs are listed in the specified file using the format name_000ii_000jj."
                    "Comments are prefixed by #. Otherwise process all DSMs in the image_folders.")
parser.add_argument("--debug", action="store_true",
                    help="Print additional information")
args = parser.parse_args()

imagesList = []
for oneFolder in args.image_folders:
    imagesInFolder = glob.glob(oneFolder + "/*.NTF")
    imagesList.extend(imagesInFolder)

images = numpy.array(imagesList)
angles = numpy.zeros(len(images))
bounds = numpy.zeros([len(images), 4])
for i,f in enumerate(images):
    sourceImage = gdal.Open(f, gdal.GA_ReadOnly)
    metaData = sourceImage.GetMetadata()
    angles[i] = metaData['NITF_CSEXRA_OBLIQUITY_ANGLE']
    bounds[i] = gdal_utils.bounding_box(sourceImage)

# list of dsms
dsmList = glob.glob(args.dsm_folder + "/dsm_*.tif")
dsms = numpy.array(dsmList)
sortIndex = dsms.argsort()
dsms = dsms[sortIndex]

reIndex = re.compile(r'\d+')

ids = dsms
if args.dense_ids:
    f = open(args.dense_ids)
    if not f:
        sys.exit(10)
    ids = [line for line in f if not line[0] == '#']
    ids = [reIndex.findall(line) for line in ids]
    ids = ["dsm_{}_{}.tif".format(line[0][-2:], line[1][-2:])  for line in ids]
else:
    ids = [os.path.basename(line) for line in dsms]
ids = set(ids)

for dsm in dsms:
    dsmBasename = os.path.basename(dsm)
    if not dsmBasename in ids:
        print("Skipping {} not in dense_ids".format(dsmBasename))
        continue
    print("Processing {}".format(dsmBasename))
    index = reIndex.findall(dsm)
    dsmImage = gdal.Open(dsm, gdal.GA_ReadOnly)
    dsmBounds = gdal_utils.bounding_box(dsmImage)
    dsmArea = (dsmBounds[2] - dsmBounds[0]) * (dsmBounds[3] - dsmBounds[1])
    areas = numpy.zeros(len(images))
    for i, source_image in enumerate(images):
        imageIntersectDsm = intersection(dsmBounds, bounds[i])
        areaImageIntersectDsm = (imageIntersectDsm[2] - imageIntersectDsm[0]) *\
            (imageIntersectDsm[3] - imageIntersectDsm[1]) if imageIntersectDsm else 0
        # area of DSM not covered
        areas[i] = dsmArea - areaImageIntersectDsm
    # sort images by areas and angle
    sortIndex = numpy.lexsort((angles, areas))
    images = images[sortIndex]
    angles = angles[sortIndex]
    bounds = bounds[sortIndex]
    areas = areas[sortIndex]
    if args.debug:
        print("========== Sorted list of images ==========")
        for i in range(len(images)):
            print("{} {}: {} {} (dsmBounds: {}, bounds: {}) {}".format(
                index[0], index[1], os.path.basename(images[i]), areas[i] / dsmArea,
                dsmBounds, bounds[i], angles[i]))

    source_image = images[0]
    print("Using {} percentage not covered: {} angle: {}".format(
        source_image, areas[0] / dsmArea, angles[0]))
    destination_image = os.path.basename(source_image)
    destination_image = os.path.splitext(destination_image)[0]
    if args.rpc_folder:
        oargs_raytheon_rpc = glob.glob(
            args.rpc_folder + "/GRA_" + destination_image + '*.up.rpc')[0]
    postfix = "pan" if source_image.find('PAN') > 0 else "msi"
    oargs_destination_image =\
      destination_image + "_or_" + postfix + "_" + index[0] +\
      "_" + index[1] + ".tif"
    ortho_params = (
        source_image, dsm, oargs_destination_image,
        args.occlusion_thresh, args.denoise_radius, oargs_raytheon_rpc)
    print(orthoParamsToString(*ortho_params))
    ortho.orthorectify(*ortho_params)
