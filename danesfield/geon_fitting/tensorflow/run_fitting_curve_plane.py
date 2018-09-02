#! /usr/bin/env python2

import os
import sys
import subprocess

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
    'D2', 'D1', 'D3', 'D4'
]

process_set = []
for idx, parameter in enumerate(parameter_set):
    print('Test Parameter: {}'.format(parameter))

    command = 'python fitting_curved_plane.py \
            --input_pc=../outlas/out_{}.txt \
            --output_png=../segmentation_graph/fit_{}.png \
            --output_txt=../outlas/remain_{}.txt \
            --output_geon=../out_geon/{}_Curve_Geon.npy '.format(parameter,
                                                                 parameter, parameter, parameter)

    print(command)
    subprocess.call(command, shell=True)
    command = 'txt2las -i ../outlas/remain_{}.txt -o ../outlas/remain_{}.las -parse xyzc'.format(
        parameter, parameter)
    print(command)
    subprocess.call(command, shell=True)
