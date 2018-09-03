#!/usr/bin/env python

import sys
import os
import argparse
import requests
import subprocess


def main(args):
    parser = argparse.ArgumentParser(description="Download Openstreetmap data \
    for a region and convert to shapefile")
    parser.add_argument(
        '--left',
        type=str,
        required=True,
        help='Longitude of left / westernmost side of bounding box')
    parser.add_argument(
        '--bottom',
        type=str,
        required=True,
        help='Lattitude of bottom / southernmost side of bounding box')
    parser.add_argument(
        '--right',
        type=str,
        required=True,
        help='Longitude of right / easternmost side of bounding box')
    parser.add_argument(
        '--top',
        type=str,
        required=True,
        help='Lattitude of top / northernmost side of bounding box')
    parser.add_argument(
        '--api-endpoint',
        type=str,
        default="https://api.openstreetmap.org/api/0.6/map")
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True)
    parser.add_argument(
        '--output-prefix',
        type=str,
        default="road_vector")

    args = parser.parse_args(args)

    request_params = {"bbox": ",".join([args.left,
                                        args.bottom,
                                        args.right,
                                        args.top])}
    response = requests.get(args.api_endpoint, request_params)

    output_osm_filepath = os.path.join(args.output_dir, "{}.osm".format(args.output_prefix))
    try:
        os.mkdir(args.output_dir)
    except FileExistsError as e:
        pass

    with open(output_osm_filepath, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=128):
            fd.write(chunk)

    subprocess.run(['ogr2ogr',
                    '-f',
                    'ESRI Shapefile',
                    args.output_dir,
                    output_osm_filepath,
                    'lines'], check=True)

    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
