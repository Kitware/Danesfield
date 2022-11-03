#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

'''
Take point cloud data and create a texture mapping to 'paint' that data on to
a mesh. Two methods are supported. 'Splatting' where each point is mapped to the
nearest mesh triangle or 'sampling' where each mesh triangle uses the value from
the nearest point in the point cloud.
'''

import argparse
import numpy as np
import sys

from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter

def main(args):
    parser = argparse.ArgumentParser(
        description="Diffuse texture image to remove edge artifacts.")
    parser.add_argument("input_image", help="path to input image.")
    parser.add_argument("output_image", help="path to output image.")
    parser.add_argument("--num_iterations", help="number of interations.",
                        default=100, type=int)
    args = parser.parse_args(args)

    # Read in input image to numpy array
    in_image = np.array(Image.open(args.input_image), np.float64)
    filtered_img = np.copy(in_image)

    # index of points in original texture map
    orig_indx = (in_image[:,:,0] > 0.)

    # Filter and replace in loop
    for i in range(args.num_iterations):
        for j in range(3):
            filtered_img[:,:,j] = gaussian_filter(filtered_img[:,:,j], sigma=1)
        filtered_img[orig_indx] = in_image[orig_indx]

    pil_img = Image.fromarray(filtered_img.astype(np.uint8))
    pil_img.save(args.output_image)

if __name__ == '__main__':
    main(sys.argv[1:])
