#!/usr/bin/env python

"""
Coordinate system utilities for CORE3D metrics.
"""

import json
import logging
import os
import shutil
import subprocess


def get_coordinate_system(image):
    """
    Get PROJ.4 coordinate system string from an image file.
    """
    result = subprocess.run(
        ['gdalinfo', '-proj4', '-json', image],
        stdout=subprocess.PIPE,
        encoding='utf-8',
        check=True)
    info = json.loads(result.stdout)
    proj4 = info['coordinateSystem']['proj4']

    return proj4


def convert_coordinate_system(image, ref_proj4):
    """
    Convert image's coordinate system to ref_proj4, if necessary.
    """
    proj4 = get_coordinate_system(image)
    if proj4 == ref_proj4:
        logging.debug('Skipping coordinate system conversion for {}'.format(image))
        return

    logging.debug('Converting coordinate system for {}'.format(image))

    output_path, output_basename = os.path.split(image)
    output_name, output_ext = os.path.splitext(output_basename)
    output_name += '-gdalwarp'
    output_image = os.path.join(output_path, output_name + output_ext)

    subprocess.run(['gdalwarp', '-t_srs', ref_proj4, image, output_image], check=True)
    shutil.copyfile(output_image, image)
    os.remove(output_image)
