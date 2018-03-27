import argparse
import gdal

parser = argparse.ArgumentParser(
    description='Change the geotransform of an image')
parser.add_argument("image", help="Image file name")
parser.add_argument("--x_shift", type=float, help="The origin is shifted with this amount")
parser.add_argument("--y_shift", type=float, help="The origin is shifted with this amount")
args = parser.parse_args()

image = gdal.Open(args.image, gdal.GA_Update)
if not image:
    exit(1)

# check the default geotransform
oldgeo=image.GetGeoTransform()
geo = list(oldgeo)
if (args.x_shift):
    geo[0] = geo[0] + args.x_shift
if (args.y_shift):
    geo[3] = geo[3] + args.y_shift

print("Changing geotransform from {} to {}".format(oldgeo, geo))

image.SetGeoTransform(geo)
