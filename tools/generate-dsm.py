import argparse
import json
import numpy
import subprocess
import sys


def getMinMax(json_string):
    j = json.loads(json_string)
    j = j["stats"]["statistic"]
    minX = j[0]["minimum"]
    maxX = j[0]["maximum"]
    minY = j[1]["minimum"]
    maxY = j[1]["maximum"]
    return minX, maxX, minY, maxY


parser = argparse.ArgumentParser(
    description='Generate a Digital Surface Model (DSM) from a point cloud')
parser.add_argument("-s", "--source_points", nargs="+", help="Source points file[s]")
parser.add_argument("destination_image", help="Destination image file name")
parser.add_argument("--bounds", nargs=4, type=float, action="store",
                    help="Destination image bounds (using the coordinate system "
                         "of the source_points file): minX, maxX, minY, maxY. "
                         "If not specified, it is computed from source_points files")
parser.add_argument("--gsd", help="Ground sample distance")
args = parser.parse_args()

if not args.gsd:
    args.gsd = 0.25
    print("Using gsd = 0.25 m")

if not args.source_points:
    print("error: At least one source_points file required")
    sys.exit(1)
if args.bounds:
    minX, maxX, minY, maxY = args.bounds
else:
    print("Computing the bounding box for {} point cloud files ...".format(
        len(args.source_points)))
    minX = numpy.inf
    maxX = - numpy.inf
    minY = numpy.inf
    maxY = - numpy.inf
    pdal_info_template = ["pdal", "info", "--stats", "--dimensions", "X,Y"]
    for i, s in enumerate(args.source_points):
        pdal_info_args = pdal_info_template + [s]
        out = subprocess.check_output(pdal_info_args)
        tempMinX, tempMaxX, tempMinY, tempMaxY = getMinMax(out)
        if tempMinX < minX:
            minX = tempMinX
        if tempMaxX > maxX:
            maxX = tempMaxX
        if tempMinY < minY:
            minY = tempMinY
        if tempMaxY > maxY:
            maxY = tempMaxY
        if i % 10 == 0:
            print("Iteration {}".format(i))
print("Bounds ({}, {}, {}, {})".format(minX, maxX, minY, maxY))

# compensate for PDAL expanding the extents by 1 pixel
maxX -= args.gsd
maxY -= args.gsd

# read the pdal file and project the points
jsonTemplate = """
{
  "pipeline": [
    %s,
    {
      "type": "filters.crop",
      "bounds": "([%s, %s], [%s, %s])"
    },
    {
      "resolution": %s,
      "data_type": "float",
      "filename":"%s",
      "output_type": "max",
      "window_size": "20",
      "bounds": "([%s, %s], [%s, %s])"
    }
  ]
}"""
print("Generating DSM ...")
all_sources = ",\n".join("\"" + str(e) + "\"" for e in args.source_points)
pipeline = jsonTemplate % (all_sources,
                           minX, maxX, minY, maxY,
                           args.gsd, args.destination_image,
                           minX, maxX, minY, maxY)
pdal_pipeline_args = ["pdal", "pipeline", "--stream", "--stdin"]
response = subprocess.run(pdal_pipeline_args, input=pipeline.encode(),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if response.returncode != 0:
    print("PDAL failed with error code {}", format(response.returncode))
    print("STDERR")
    print(response.stderr)
    print("STDOUT")
    print(response.stdout)
