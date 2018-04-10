#!/usr/bin/env python

"""
Datatype utilities for CORE3D metrics.
"""

import json
import logging
import os
import shutil
import subprocess


def is_float64(image):
    """
    Check whether an image has bands with the Float64 data type.
    """
    result = subprocess.run(
        ['gdalinfo', '-json', image],
        stdout=subprocess.PIPE,
        encoding='utf-8',
        check=True)
    info = json.loads(result.stdout)
    bands = info['bands']
    return any(band['type'] == 'Float64' for band in bands)


def convert_float32(image):
    """
    Convert an image to the Float32 data type, if necessary.
    """
    if not is_float64(image):
        logging.debug('Skipping data type conversion for {}'.format(image))
        return

    logging.debug('Converting data type to Float32 for {}'.format(image))

    output_path, output_basename = os.path.split(image)
    output_name, output_ext = os.path.splitext(output_basename)
    output_name += '-gdal_translate'
    output_image = os.path.join(output_path, output_name + output_ext)

    subprocess.run(['gdal_translate', '-ot', 'Float32', image, output_image], check=True)
    shutil.copyfile(output_image, image)
    os.remove(output_image)
