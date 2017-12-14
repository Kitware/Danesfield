import argparse
import gdal
import subprocess
from subprocess import call

parser = argparse.ArgumentParser(
    description='Orthorectify an image given the DEM')
parser.add_argument("source_image", help="Source image file name")
parser.add_argument("dem", help="Digital elevation model (DEM) image file name")
parser.add_argument("rectified_image", help="Orthorectified image file name")
args = parser.parse_args()

rpc_arg = "RPC_DEM=%s" % args.dem
call_args = ["gdalwarp", "-rpc", "-to", rpc_arg,
             args.source_image, args.rectified_image]
subprocess.call(call_args)
