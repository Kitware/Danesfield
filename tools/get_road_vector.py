#!/usr/bin/env python

import sys
import os
import argparse
import requests
import subprocess
import json
import re


def main(args):
    parser = argparse.ArgumentParser(description="Download Openstreetmap data \
    for a region and convert to GeoJSON")
    parser.add_argument(
        '--left',
        type=str,
        required=True,
        help='Longitude of left / westernmost side of bounding box')
    parser.add_argument(
        '--bottom',
        type=str,
        required=True,
        help='Latitude of bottom / southernmost side of bounding box')
    parser.add_argument(
        '--right',
        type=str,
        required=True,
        help='Longitude of right / easternmost side of bounding box')
    parser.add_argument(
        '--top',
        type=str,
        required=True,
        help='Latitude of top / northernmost side of bounding box')
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

    # Make the initial conversion from OSM to GeoJSON
    geojson_preform_output_filepath = os.path.join(
        args.output_dir, "{}.preformat.geojson".format(args.output_prefix))
    osm_sql_query = 'SELECT * FROM lines'
    subprocess.run(['ogr2ogr',
                    '-f',
                    'GeoJSON',
                    geojson_preform_output_filepath,
                    output_osm_filepath,
                    '-sql',
                    osm_sql_query],
                   check=True)

    geojson_output_filepath = os.path.join(
        args.output_dir, "{}.geojson".format(args.output_prefix))
    # Manually tweak GeoJSON to fit expected format
    with open(geojson_preform_output_filepath, 'r') as f:
        json_data = json.load(f)

    json_data["features"] = [feature_map(f) for f in json_data["features"]]

    with open(geojson_output_filepath, 'w') as f:
        f.write(json.dumps(json_data))

    return 0


def properties_map(in_properties):
    properties = in_properties.copy()
    other_tags = json.loads("{" +
                            re.sub("=>", ":", properties.get("other_tags", "")) +
                            "}")

    if "railway" in other_tags:
        properties["railway"] = other_tags["railway"]

    # Forcing bridge in output tags
    if other_tags.get("bridge") == "yes":
        bridge_val = 1
    else:
        bridge_val = 0

    properties["bridge"] = bridge_val

    # Only interested in these classes for now; in priority order
    class_level_properties = ["highway",
                              "railway"]

    for k in class_level_properties:
        if k in properties:
            properties["class"] = k
            properties["type"] = properties[k]
            break

    return properties


def feature_map(in_feature):
    feature = in_feature.copy()

    feature["properties"] = properties_map(feature["properties"])

    return feature


if __name__ == '__main__':
    main(sys.argv[1:])
