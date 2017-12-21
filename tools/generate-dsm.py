import argparse
import os
import pdal
import gdal

parser = argparse.ArgumentParser(
    description='Generate a Digital Surface Model (DSM) from a point cloud')
parser.add_argument("source_points", help="Source points file name")
parser.add_argument("destination_image", help="Destination image file name")
parser.add_argument("--gsd", help="Ground sample distance")
args = parser.parse_args()
tempImage = "_" + args.destination_image

if (not args.gsd):
    args.gsd = 0.25
    print("Using gsd = 0.25 m")
# read the pdal file and project the points
json = u"""
{
  "pipeline": [
    "%s",
    {
      "resolution": %s,
      "filename":"%s",
      "output_type": "max",
      "window_size": "20"
    }
  ]
}"""
print("Generating DSM ...")
json = json % (args.source_points, args.gsd, tempImage)
pipeline = pdal.Pipeline(json)
pipeline.validate()  # check if our JSON and options were good
# this causes a segfault at the end of the program
# pipeline.loglevel = 8  # really noisy
count = pipeline.execute()

print("Converting to EPSG:4326 ...")
gdal.Warp(args.destination_image, tempImage, dstSRS="EPSG:4326")
os.remove(tempImage)
