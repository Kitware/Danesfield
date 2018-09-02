"""
Run multiple parameter with multiple GPUs and one python script 
Usage: python run_all.py

Author: Xu Zhang
Email: xu.zhang@columbia.edu.cn
"""

#! /usr/bin/env python2

import os
import sys
import subprocess

####################################################################
# Parse command line
####################################################################


def usage():
    print >> sys.stderr
    sys.exit(1)


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


parameter_set = [
    'D2',
    'D1',
    'D3', 'D4'
]

dtm_set = [
    'D2_WPAFB',
    'D1_WPAFB',
    'D3_UCSD', 'D4_Jacksonville'
]

process_set = []
for idx, parameter in enumerate(parameter_set):
    print('Test Parameter: {}'.format(parameter))
    # Dayton_20sqkm Jacksonville
    command = 'python geon_to_mesh.py \
            --input_geon=../out_geon/{}_Curve_Geon.npy \
            --input_dtm=/dvmm-filer2/projects/Core3D/{}/DTMs/{}_DTM.tif \
            --output_mesh=../out_geon/{}_Curve_Mesh.ply'.format(parameter, dtm_set[idx],
                                                                parameter, parameter)

    print(command)
    subprocess.call(command, shell=True)
