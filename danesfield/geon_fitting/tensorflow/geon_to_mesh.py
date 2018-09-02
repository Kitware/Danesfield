import numpy as np

import utils
import pickle
import plyfile
import gdal

import argparse


def get_theta(length, r):
    if length < -0.8*r:
        return np.pi
    if length > 0.9*r:
        return 0
    if length < 0:
        return np.pi-np.arccos((-1*length)/r)
    return np.arccos(length/r)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_geon', default='../out_geon/D4_Curve_Geon.npy',
    help='input geon file.')
parser.add_argument(
    '--input_dtm', default='/dvmm-filer2/projects/Core3D/D4_Jacksonville/DTMs/D4_DTM.tif',
    help='Input labelled point cloud. The point cloud should has geon type label. ')  # D3_UCSD D4_Jacksonville
parser.add_argument(
    '--output_mesh', default='../out_geon/D4_Curve_Mesh.ply',
    help='Output txt result file.')
args = parser.parse_args()

original_dtm = gdal.Open(args.input_dtm, gdal.GA_ReadOnly)
gt = original_dtm.GetGeoTransform()  # captures origin and pixel size
left = gdal.ApplyGeoTransform(gt, 0, 0)[0]
top = gdal.ApplyGeoTransform(gt, 0, 0)[1]
right = gdal.ApplyGeoTransform(
    gt, original_dtm.RasterXSize, original_dtm.RasterYSize)[0]
bottom = gdal.ApplyGeoTransform(
    gt, original_dtm.RasterXSize, original_dtm.RasterYSize)[1]

dtm = original_dtm.ReadAsArray()

projection_model = {}
projection_model['corners'] = [left, top, right, bottom]
projection_model['project_model'] = gt
projection_model['scale'] = 1.0

geon_model = []
all_vertex = []
all_face = []

center_of_mess, geon_model = pickle.load(open(args.input_geon, "rb"))

for model in geon_model:
    if model['name'] == 'poly_cylinder':
        centroid, ex, ey, coefficients, min_axis_z,\
            max_axis_z, ortho_x_min, ortho_x_max, fitted_indices_length, mean_diff = model[
                'model']
        vertex, face = utils.get_poly_ply_volume(dtm, projection_model, centroid, ex, ey,
                                                 coefficients, min_axis_z, max_axis_z,
                                                 ortho_x_min, ortho_x_max, len(all_vertex), center_of_mess)
        if len(all_vertex) > 0:
            all_vertex.extend(vertex)
            all_face.extend(face)
        else:
            all_vertex = vertex
            all_face = face

    elif model['name'] == 'sphere':
        centroid, r, min_axis_z,\
            max_axis_z, fitted_indices_length = model['model']
        theta_max = get_theta(min_axis_z, r)
        theta_min = get_theta(max_axis_z, r)

        vertex, face = utils.get_sphere_volume(dtm, projection_model, centroid, r,
                                               theta_min, theta_max, len(all_vertex), center_of_mess)

        if len(all_vertex) > 0:
            all_vertex.extend(vertex)
            all_face.extend(face)
        else:
            all_vertex = vertex
            all_face = face

for idx in range(len(all_vertex)):
    all_vertex[idx] = (all_vertex[idx][0]+center_of_mess[0],
                       all_vertex[idx][1]+center_of_mess[1], all_vertex[idx][2]+center_of_mess[2])

all_vertex = np.array(
    all_vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
all_face = np.array(all_face, dtype=[(
    'vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

el_vertex = plyfile.PlyElement.describe(all_vertex, 'vertex')
el_face = plyfile.PlyElement.describe(all_face, 'face')
plyfile.PlyData([el_vertex, el_face]).write(args.output_mesh)
