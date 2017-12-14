import argparse
import pdal

parser = argparse.ArgumentParser(
    description='Orthorectify an image using '
                'an RPC projection and a point cloud')
parser.add_argument("source_image", help="Source image file name")
parser.add_argument("source_points", help="Source points file name")
parser.add_argument("destination_image", help="Destination image file name")
parser.add_argument("--raytheon-rpc", type=str,
                    help="Raytheon RPC file name. If not provided, "
                    "the RPC is read from the source_image")
args = parser.parse_args()


# read the pdal file and project the points
json = u"""
{
  "pipeline": [
    "%s",
    {
        "type":"filters.reprojection",
        "out_srs":"EPSG:4326"
    },
    {
      "type":"filters.python",
      "script":"tools/orthorectify-pdal-stage.py",
      "function":"store_gray",
      "add_dimension": "gray",
      "pdalargs":{"source_image":"%s"}
    },
    {
      "resolution": 0.000003,
      "filename":"%s",
      "output_type": "max",
      "window_size": "20",
      "dimension": "gray"
    }
  ]
}"""
print("Reproject points ...")
json = json % (args.source_points, args.source_image, args.destination_image)
pipeline = pdal.Pipeline(json)
pipeline.validate()  # check if our JSON and options were good
# this causes a segfault at the end of the program
# pipeline.loglevel = 8  # really noisy
count = pipeline.execute()
