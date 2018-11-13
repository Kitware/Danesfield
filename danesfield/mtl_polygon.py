###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

import numpy as np
from PIL import Image, ImageDraw
import gdal
from scipy.stats import mode


def _get_label(image, x, y):
    labels = image[x, y]
    material_index = int(mode(labels)[0][0])
    materials = ['Undefined', 'Asphalt', 'Concrete', 'Glass', 'Tree',
                 'Non_Tree_Veg', 'Metal', 'Soil', 'Ceramic', 'Solar_Panel',
                 'Water', 'Polymer']
    return materials[material_index]


def assign_mtl_polygon_label(polygons, in_dataset, label_img_path):
    height = in_dataset.RasterXSize
    width = in_dataset.RasterYSize

    label_image = gdal.Open(
        label_img_path, gdal.GA_ReadOnly).ReadAsArray()

    polygon_labels = {}
    for i, polygon in polygons.items():
        polygon = [tuple(idx) for idx in polygon]

        mask = Image.new('L', (width, height), 0)
        ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
        mask = np.asarray(mask)

        x, y = np.where(mask == 1)
        if len(x) == 0:
            mtl_label = 'Undefined'
        else:
            mtl_label = _get_label(label_image, x, y)
        polygon_labels[i] = mtl_label

    return polygon_labels
