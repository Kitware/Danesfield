#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

import os
import sys
import argparse
import subprocess
from ply2txt import convert
from clean_point_cloud import clean

def main(args):
    parser = argparse.ArgumentParser(
        description='Generate LAS file from satellite images.')
    parser.add_argument('--config_file',
                        help='JSON file to configure VisSat', required=True)
    parser.add_argument('--work_dir',
                        help='Work directory for VisSat', required=True)
    parser.add_argument('--point_cloud',
                        help='Output LAS file', required=True)
    parser.add_argument('--utm', 
                        help='UTM zone that AOI is located in', required=True)
    parser.add_argument('--dtm', help='Provide a DTM to obtain min and max alt.')
    args = parser.parse_args(args)
    
    if args.dtm:
        from danesfield.gdal_utils import gdal_open
        import json
        src = gdal_open(args.dtm)
        data = src.GetRasterBand(1).ReadAsArray()
        min_val = min(data[data!=src.GetRasterBand(1).GetNoDataValue()])
        max_val = max(data[data!=src.GetRasterBand(1).GetNoDataValue()])
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        config['alt_min'] = float(min_val)
        config['alt_max'] = float(max_val)
        with open(args.config_file, 'w') as f:
            json.dump(config, f, indent=4)
    cmd_args = ['/bin/bash', '-c', 
                'python3 /VisSatSatelliteStereo/stereo_pipeline.py --config_file '
                +args.config_file]
    subprocess.run(cmd_args, check=True)
    
    convert([os.path.join(args.work_dir, 
            'mvs_results/aggregate_3d/aggregate_3d.ply'), 
            os.path.join(args.work_dir, 
            'mvs_results/aggregate_3d/aggregate_3d.txt')])
    clean(os.path.join(args.work_dir, 'mvs_results/aggregate_3d/aggregate_3d.txt'),
          os.path.join(args.work_dir, 'mvs_results/aggregate_3d/aggregate_3d_dense.txt'))

    subprocess.run(["/LAStools/bin/txt2las", 
                    "-i", os.path.join(args.work_dir,
                    'mvs_results/aggregate_3d/aggregate_3d_dense.txt'), 
                    "-parse", "xyz", 
                    "-o", args.point_cloud, 
                    "-utm", args.utm, 
                    "-target_utm", args.utm], check=True)


if __name__=="__main__":
    main(sys.argv[1:])
