import argparse
import glob as glob
import os
from danesfield.materials.pixel_prediction.util.model import Classifier
from danesfield.materials.pixel_prediction.util.misc import create_mode_img
import sys
import numpy as np
from osgeo import gdal


parser = argparse.ArgumentParser(
    description="Classify given orthorectifed image(s). \
                Input: <dataset_path> <outpu_path> <AOI> \
                <OPTIONAL: cuda> <OPTIONAL: aux_out>")
parser.add_argument(
    'dataset_path', help='Path to image or folder of images.')
parser.add_argument(
    'output_path', help='Path to save result images to.')
parser.add_argument(
    'AOI', help='D#, the AOI that the image(s) belongs to. Eg D3')
parser.add_argument('--cuda', action="store_true",
                    help="Use GPU. (May have to adjust batch_size value)")
parser.add_argument('--aux_out', action="store_true",
                    help='Output a class probably image in output \
                    path for each image.')

args = parser.parse_args()

# Check AOI input
accepted_AOIs = ['D1', 'D2', 'D3', 'D4']
if any(aoi == args.AOI in accepted_AOIs for aoi in accepted_AOIs):
    pass
else:
    print("Incorrect AOI value. Please select either D1, D2, D3, or D4.")
    sys.exit()

# Load model and select option to use CUDA
classifier = Classifier(args.cuda, args.output_path, args.AOI,
                        batch_size=2**15, aux_out=args.aux_out)

# Check if dataset_path is folder or single file
folder = os.path.isdir(args.dataset_path)

img_paths = []
# If folder: find all tif's in folder
if folder:
    for img_path in glob.glob(args.dataset_path + '*.tif'):
        img_paths.append(img_path)
else:
    if args.dataset_path[-4:] == ".tif":
        img_paths.append(args.dataset_path)
    else:
        print("Invalid single image path.")
        sys.exit()

if folder:
    img = gdal.Open(img_paths[0], gdal.GA_ReadOnly)
    stack_img = np.empty(
        (img.RasterYSize, img.RasterXSize, img.RasterCount))

# Use CNN to create material map
for i, img_path in enumerate(img_paths):
    print("IMG {}/{}".format(i+1, len(img_paths)))
    result_img = classifier.Evaluate(img_path)
    if folder:
        stack_img[:, :, i] = result_img

# Save mode image in output_path
if folder:
    create_mode_img(stack_img, args.output_path)
