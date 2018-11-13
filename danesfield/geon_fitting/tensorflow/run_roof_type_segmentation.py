###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

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
import shlex

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

gpu_set = ['3']
parameter_set = [ 
        'D1','D2','D3','D4' 
        ]
#parameter_set = [ 
#        'D2'
#        ]

number_gpu = len(gpu_set)
process_set = []

for idx, parameter in enumerate(parameter_set):
    print('Test Parameter: {}'.format(parameter))
    #Dayton_20sqkm Jacksonville
    command = 'python roof_segmentation.py \
            --input_pc=/home/xuzhang/project/Core3D/core3d-columbia/data/{}.las_bd.txt --output_png=../segmentation_graph/out_{}.png \
            --text_output --output_txt=../outlas/out_{}.txt --gpu_id {} '.format(parameter, parameter, parameter, gpu_set[idx%number_gpu])

    print(command)
    p = subprocess.Popen(shlex.split(command))
    process_set.append(p)
    
    if (idx+1)%number_gpu == 0:
        print('Wait for process end')
        for sub_process in process_set:
            sub_process.wait()
    
        process_set = []

for sub_process in process_set:
    sub_process.wait()

