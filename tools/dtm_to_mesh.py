#!/usr/bin/env python

import argparse
import gdal
import gdalconst
import logging
import numpy as np
import scipy.misc
import scipy.ndimage
import sys
import cv2

# This script generates a mesh from a DTM.
# The DTM is downsampled by the parameter --downsample (40 by default)


def main(args):
    parser = argparse.ArgumentParser(
        description='Transform a DTM into a mesh')
    parser.add_argument("dtm_file", help="DTM image (.tif)")
    parser.add_argument("offset_file", help="Text file containing the offset used"
                                            "in x, y and z (one value per line)")
    parser.add_argument("output_file", help="Output mesh file (.obj)")
    parser.add_argument("--downsample", action="store", type=int, default=40,
                        help="Downsampling factor for the DTM")
    args = parser.parse_args()

    dtm_file = args.dtm_file
    offset_file = args.offset_file
    output_file = args.output_file
    reduction_factor = args.downsample
    print("Downsampling factor ", reduction_factor)

    # read the mesh offset
    with open(offset_file, "r") as offset_f:
        lines = offset_f.readlines()
    mesh_offset = np.array([float(lines[0]), float(lines[1]), float(lines[2])])

    # read DTM image
    dtm = cv2.imread(dtm_file, cv2.IMREAD_LOAD_GDAL)

    # smooth and downsample the DTM
    smooth_size = int(reduction_factor / 4)
    smooth_kernel = np.full((smooth_size, smooth_size),
                            1.0 / (smooth_size * smooth_size))
    dtm = scipy.ndimage.filters.convolve(dtm, smooth_kernel)
    nb_u_samples = int(dtm.shape[1] / reduction_factor)
    nb_v_samples = int(dtm.shape[0] / reduction_factor)
    downsampled_u = np.linspace(0, dtm.shape[1] - 1, nb_u_samples)
    downsampled_v = np.linspace(0, dtm.shape[0] - 1, nb_v_samples)
    dtm = dtm[downsampled_v.astype(int), :]
    dtm = dtm[:, downsampled_u.astype(int)]

    # read DTM origin and scale
    gdal.AllRegister()
    dataset = gdal.Open(dtm_file, gdalconst.GA_ReadOnly)
    geo_transform = dataset.GetGeoTransform()
    origin = np.array([geo_transform[0], geo_transform[3]])
    scale = np.array([geo_transform[1], geo_transform[5]])

    # build array of 3D points in utm world coordinates
    x = np.tile(downsampled_u, nb_v_samples) * scale[0] + origin[0]
    y = np.repeat(downsampled_v, nb_u_samples) * scale[1] + origin[1]
    z = dtm.ravel()
    xyz = np.vstack((x, y, z)).T

    # translate points with the same offset
    xyz -= mesh_offset

    # build list of faces
    faces = []
    for j in range(nb_v_samples-1):
        for i in range(nb_u_samples-1):
            id0 = j * nb_u_samples + i
            id1 = id0 + 1
            id2 = id0 + nb_u_samples
            id3 = id2 + 1
            faces.append([id0, id2, id1])
            faces.append([id2, id3, id1])

    # write DTM mesh to OBJ file
    with open(output_file, "w") as file:
        for pt in xyz:
            file.write("v " + str(pt[0]) + " " + str(pt[1]) + " " + str(pt[2]) + "\n")
        for f in faces:
            file.write("f " + str(f[0]+1) + " " + str(f[1]+1) + " " + str(f[2]+1) + "\n")
    print("Done.")


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit()
