"""
Run multiple parameter with multiple GPUs and one python script
Usage: python run_all.py

Author: Xu Zhang
Email: xu.zhang@columbia.edu.cn
"""

# ! /usr/bin/env python2

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
# parameter_set = [' --shift --da ', ' --da ', ' ', ' --da --shift --flip ', '
# --da --shift --scale ', ' --da --shift --flip --scale ']# , ' --da '
parameter_set = [' --da --shift --scale ']  # , ' --da ',  , ' --da --shift --scale '
# parameter_set = [' --da ']# , ' --da '

number_gpu = len(gpu_set)
process_set = []

for idx, parameter in enumerate(parameter_set):
    print('Test Parameter: {}'.format(parameter))

    command = 'python test_roof_segmentation.py {}  --gpu_id={}'.format(
        parameter, gpu_set[idx % number_gpu])

    print(command)
    p = subprocess.Popen(shlex.split(command))
    process_set.append(p)

    if (idx+1) % number_gpu == 0:
        print('Wait for process end')
        for sub_process in process_set:
            sub_process.wait()

        process_set = []

for sub_process in process_set:
    sub_process.wait()
