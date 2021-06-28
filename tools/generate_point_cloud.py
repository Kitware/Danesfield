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
    args = parser.parse_args(args)

    cmd_args = ['/bin/bash', '-c', 
                'source /opt/conda/etc/profile.d/conda.sh && conda activate vissat \
                && python3 /VisSatSatelliteStereo/stereo_pipeline.py --config_file '
                +args.config_file]
    subprocess.run(cmd_args, check=True)
    
    convert([os.path.join(args.work_dir, 
            'mvs_results/aggregate_3d/aggregate_3d.ply'), 
            os.path.join(args.work_dir, 
            'mvs_results/aggregate_3d/aggregate_3d.txt')])

    clean(['--input_cloud', os.path.join(args.work_dir, 
           'mvs_results/aggregate_3d/aggregate_3d.txt'),
           '--output_cloud', os.path.join(args.work_dir,
            'mvs_results/aggregate_3d/aggregate_3d_dense.txt')])
    subprocess.run(["/LAStools/bin/txt2las", 
                    "-i", os.path.join(args.work_dir,
                    'mvs_results/aggregate_3d/aggregate_3d_dense.txt'), 
                    "-parse", "xyz", 
                    "-o", args.point_cloud, 
                    "-utm", args.utm, 
                    "-target_utm", args.utm], check=True)


if __name__=="__main__":
    main(sys.argv[1:])
