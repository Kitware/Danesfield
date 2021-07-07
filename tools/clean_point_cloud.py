#!/usr/bin/env python
import numpy as np
import sys
import os
import subprocess
import argparse
import random

def clean(args):
    parser = argparse.ArgumentParser(
        description='Apply filtering to a point cloud.')
    parser.add_argument('--input_cloud',
                        help='Point cloud text file to clean', required=True)
    parser.add_argument('--output_cloud',
                        help='Cleaned point cloud as a text file', required=True)
    args = parser.parse_args(args)

    subprocess.run(['bilateralfilter', args.input_cloud,
                    os.path.join(os.path.dirname(args.input_cloud), 
                    'filtered_cloud.txt'),
                    '-N', '1', '-r', '2', '-n', '2'], check=True)

    print('reading file...')
    hrad_meters = 1.0
    fid = open(os.path.join(os.path.dirname(args.input_cloud), 
                    'filtered_cloud.txt'), 'r')
    lines = fid.readlines()
    fid.close()
    num = len(lines)
    x = np.zeros(num)
    y = np.zeros(num)
    z = np.zeros(num)
    for k in range(num):
        xyz = lines[k].split()
        x[k] = float(xyz[0])
        y[k] = float(xyz[1])
        z[k] = float(xyz[2])
    xyz = None
    # write new ASCII XYZ file
    print('writing file...')
    random.seed(0)
    fid = open(args.output_cloud,'w')
    for k in range(len(x)):
        xx = x[k] + random.uniform(-hrad_meters, hrad_meters)
        yy = y[k] + random.uniform(-hrad_meters, hrad_meters)
        zz = z[k]
        if not (np.isnan(xx) or np.isnan(yy) or np.isnan(zz)):
            fid.write(str(xx) + ' ' + str(yy) + ' ' + str(zz) + '\n')
    fid.close()

if __name__=='__main__':
    clean(sys.argv[1:])
