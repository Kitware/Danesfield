# -*- coding: utf-8 -*-
# Euclidean Cluster Extraction
# http://pointclouds.org/documentation/tutorials/cluster_extraction.php#cluster-extraction
import numpy as np
import pcl
import os

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
#import plotly.plotly as py
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt
import two_D_fitting
import utils
import pickle
import plyfile
import gdal
import copy
import time

import argparse

def fit_cylinder(points, max_r=80, min_r=40):
    section_pc = pcl.PointCloud()
    section_pc.from_array(points)

    cylinder_seg = section_pc.make_segmenter_normals(ksearch=50)
    cylinder_seg.set_optimize_coefficients (True)
    cylinder_seg.set_model_type (pcl.SACMODEL_CYLINDER)
    cylinder_seg.set_normal_distance_weight (0.1)
    cylinder_seg.set_method_type (pcl.SAC_RANSAC)
    cylinder_seg.set_max_iterations(1000)
    cylinder_seg.set_distance_threshold(2)
    cylinder_seg.set_radius_limits(min_r, max_r)
    cylinder_indices, cylinder_coefficients = cylinder_seg.segment()

    return cylinder_indices, cylinder_coefficients

def fit_sphere(points):
    section_pc = pcl.PointCloud()
    section_pc.from_array(points)

    sphere_seg = section_pc.make_segmenter_normals(ksearch=50)
    sphere_seg.set_optimize_coefficients (True)
    sphere_seg.set_model_type (pcl.SACMODEL_SPHERE)
    sphere_seg.set_normal_distance_weight (0.1)
    sphere_seg.set_method_type (pcl.SAC_RANSAC)
    sphere_seg.set_max_iterations (1000)
    sphere_seg.set_distance_threshold(1)
    sphere_seg.set_radius_limits (5, 20)
    sphere_indices, sphere_coefficients = sphere_seg.segment()

    return sphere_indices, sphere_coefficients

def check_sphere(points, c, r):
    distance = points - c
    distance = distance * distance
    distance = np.sqrt(np.sum(distance, axis=1))
    error = distance-r
    sphere_indices = np.arange(points.shape[0])
    sphere_indices = sphere_indices[np.logical_and(error<2, error>-2)]
    return sphere_indices, error

def draw_sphere(ax, c, r):
    u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)*r
    y=np.sin(u)*np.sin(v)*r
    z=np.cos(v)*r
    x = x + c[0]
    y = y + c[1]
    z = z + c[2]
    ax.plot_wireframe(x, y, z, color='r', alpha=0.5)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_pc', default='/home/xuzhang/project/Core3D/danesfield_gitlab/danesfield/geon_fitting/outlas/out_D4.txt',
    help='Input labelled point cloud. The point cloud should has geon type label. ')
parser.add_argument(
    '--input_dtm', default='/dvmm-filer2/projects/Core3D/D4_Jacksonville/DTMs/D4_DTM.tif',
    help='Input labelled point cloud. The point cloud should has geon type label. ')#D3_UCSD D4_Jacksonville
parser.add_argument(
    '--output_png', default='../segmentation_graph/out.png',
    help='Output png result file.')
parser.add_argument("--text_output", action="store_true",
    help="Output full text result or not")
parser.add_argument(
    '--output_txt', default='../outlas/remain_D4.txt',
    help='Output txt result file.')
args = parser.parse_args()


original_dtm = gdal.Open(args.input_dtm, gdal.GA_ReadOnly)
gt = original_dtm.GetGeoTransform() # captures origin and pixel size
left = gdal.ApplyGeoTransform(gt,0,0)[0]
top = gdal.ApplyGeoTransform(gt,0,0)[1]
right = gdal.ApplyGeoTransform(gt,original_dtm.RasterXSize,original_dtm.RasterYSize)[0]
bottom = gdal.ApplyGeoTransform(gt,original_dtm.RasterXSize,original_dtm.RasterYSize)[1]

dtm = original_dtm.ReadAsArray()

projection_model = {}
projection_model['corners'] = [left, top, right, bottom]
projection_model['project_model'] = gt 
projection_model['scale'] = 1.0

point_list, building_label_list, geon_label_list = utils.read_geon_type_pc(args.input_pc)
center_of_mess = np.mean(point_list, axis = 0)
point_list = point_list - center_of_mess
point_list = point_list.astype(np.float32)
cloud_filtered = pcl.PointCloud()
cloud_filtered.from_array(point_list)

print(cloud_filtered.size)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

geon_model = []
all_vertex = [] 
all_face = []
all_remaining_index = []
index = 0

building_index_list = []
building_max_index = np.max(building_label_list)
total_list = np.arange(building_label_list.shape[0])
for i in range(building_max_index+1):
    current_list = total_list[building_label_list==i]
    building_index_list.append(current_list)

geon_type_number = 4
cylinder_index = 2
sphere_index = 3

point_number_scale = 1
c_index = 0
for indices in building_index_list:
    print('new building: {}'.format(len(indices)))
    if len(indices)<300:
        if len(all_remaining_index) == 0:
            all_remaining_index = copy.copy(indices)
        else:
            all_remaining_index = np.concatenate((all_remaining_index,indices), axis = None)
        continue
    
    current_model = []

    geon_index_list = []
    for i in range(geon_type_number):
        geon_index_list.append([])
    
    building_points = np.zeros((len(indices),3), dtype=np.float32)
    num_building_points = len(indices)

    for i, indice in enumerate(indices):
        building_points[i][0] = cloud_filtered[indice][0]
        building_points[i][1] = cloud_filtered[indice][1]
        building_points[i][2] = cloud_filtered[indice][2]

    for i, indice in enumerate(indices):
        geon_index_list[geon_label_list[indice]].append(indice)

    fitted_index = np.zeros(len(indices), dtype=np.int32)
    fitted_index = fitted_index==1

    if len(geon_index_list[cylinder_index])>0.1*len(indices):
        points = np.zeros((len(geon_index_list[cylinder_index]),3), dtype=np.float32)
        for i, indice in enumerate(geon_index_list[cylinder_index]):
            points[i][0] = cloud_filtered[indice][0]
            points[i][1] = cloud_filtered[indice][1]
            points[i][2] = cloud_filtered[indice][2]
        
        current_cloud = pcl.PointCloud()
        current_cloud.from_array(points)

        vg = current_cloud.make_voxel_grid_filter()
        vg.set_leaf_size(1, 1, 1)
        current_cloud = vg.filter()

        current_points = np.zeros((current_cloud.size,3), dtype=np.float32)
        for i in range(current_cloud.size):
            current_points[i] = current_cloud[i]

        max_r=80
        min_r=40
        if num_building_points>10000:
            max_r=80
            min_r=40
        else:
            max_r=30
            min_r=10

        while True:
            cylinder_indices, cylinder_coefficients = fit_cylinder(current_points, max_r, min_r)
            #if cylinder_coefficients[-2]<0
            #    cylinder_coefficients[-2] = -1*cylinder_coefficients[-2]
            print(cylinder_coefficients)
            if len(cylinder_indices)<500*point_number_scale:
                break

            cylinder_points = np.zeros((len(cylinder_indices), 3), dtype=np.float32)
            for i, indice in enumerate(cylinder_indices):
                cylinder_points[i][0] = current_points[indice][0]
                cylinder_points[i][1] = current_points[indice][1]
                cylinder_points[i][2] = current_points[indice][2]

            #ax.scatter(cylinder_points[:,0],cylinder_points[:,1],cylinder_points[:,2],\
            #                        zdir='z', s=1, c='C{}'.format(0), rasterized=True, alpha=0.5)
            centroid, ex, ey, ez, fitted_indices, coefficients, min_axis_z, max_axis_z, mean_diff \
                = two_D_fitting.fit_2D_curve(cylinder_coefficients[3:-1], cylinder_points, fit_type='poly2', dist_threshold=10)

            for i in range(len(fitted_indices)):
                if len(fitted_indices[i])<500*point_number_scale:
                    continue
                fitted_points = np.zeros((len(fitted_indices[i]), 3),np.float32)
                for j, tmp_idx in enumerate(fitted_indices[i]):
                    fitted_points[j,:] = cylinder_points[tmp_idx,:]

                fitted_wire = utils.draw_poly_curve(ax, centroid, ex, ey, fitted_points, coefficients, min_axis_z[i], max_axis_z[i], 'C{}'.format(i))
                c_index += 1
                print(c_index,i,len(fitted_indices[i]))
                #c_index = (c_index)%4

                current_model.append({'name': 'poly', 'model': [centroid + center_of_mess, ex, ey, len(fitted_indices[i]), coefficients, min_axis_z[i],\
                        max_axis_z[i], mean_diff]})

            all_fitted_indices, error = two_D_fitting.check_2D_curve(ez,
                    coefficients, centroid, building_points,
                    fit_type='poly2', dist_threshold=10)
        #    
        #    vertex, face, ortho_x_min, ortho_x_max, boundary_points = utils.get_poly_ply_volume(dtm, projection_model, centroid, ex, ey,\
        #            fitted_points, coefficients, min_axis_z[i], max_axis_z[i], len(all_vertex), center_of_mess)
        #    
        #    if len(all_vertex) > 0: 
        #        all_vertex.extend(vertex)
        #        all_face.extend(face)
        #    else:
        #        all_vertex = vertex
        #        all_face = face
            
            fitted_index[all_fitted_indices] = True
            print('Fitted Point: ')
            print(len(all_fitted_indices), np.sum(fitted_index))

            current_cloud = current_cloud.extract(cylinder_indices, True)
            if current_cloud.size<1000*point_number_scale:
                break

            current_points = np.zeros((current_cloud.size,3), dtype=np.float32)
            for i in range(current_cloud.size):
                current_points[i][0] = current_cloud[i][0]
                current_points[i][1] = current_cloud[i][1]
                current_points[i][2] = current_cloud[i][2]
    
    print([len(x) for x in geon_index_list])
    if len(geon_index_list[sphere_index])>0.3*len(indices):
        points = np.zeros((len(geon_index_list[sphere_index]),3), dtype=np.float32)
        for i, indice in enumerate(geon_index_list[sphere_index]):
            points[i][0] = cloud_filtered[indice][0]
            points[i][1] = cloud_filtered[indice][1]
            points[i][2] = cloud_filtered[indice][2]
        
        current_cloud = pcl.PointCloud()
        current_cloud.from_array(points)

        vg = current_cloud.make_voxel_grid_filter()
        vg.set_leaf_size(1, 1, 1)
        current_cloud = vg.filter()

        current_points = np.zeros((current_cloud.size,3), dtype=np.float32)
        for i in range(current_cloud.size):
            current_points[i] = current_cloud[i]

        while True:
            sphere_indices, sphere_coefficients = fit_sphere(current_points)
            if len(sphere_indices)<200*point_number_scale:
                break
            
            if sphere_coefficients[-1]>0:
                draw_sphere(ax, sphere_coefficients[0:3] , sphere_coefficients[-1])

            sphere_points = np.zeros((len(sphere_indices), 3), dtype=np.float32)
            for i, indice in enumerate(sphere_indices):
                sphere_points[i][0] = current_points[indice][0]
                sphere_points[i][1] = current_points[indice][1]
                sphere_points[i][2] = current_points[indice][2]
            
            ax.scatter(sphere_points[:,0],sphere_points[:,1],sphere_points[:,2],\
                                    zdir='z', s=1, c='C{}'.format(3), rasterized=True, alpha=0.5)
            
            all_fitted_indices, error = check_sphere(building_points, sphere_coefficients[0:3], sphere_coefficients[-1])
            print(len(all_fitted_indices)) 
            fitted_index[all_fitted_indices] = True

            current_cloud = current_cloud.extract(sphere_indices, True)
            if current_cloud.size<1000*point_number_scale:
                break

            current_points = np.zeros((current_cloud.size,3), dtype=np.float32)
            for i in range(current_cloud.size):
                current_points[i][0] = current_cloud[i][0]
                current_points[i][1] = current_cloud[i][1]
                current_points[i][2] = current_cloud[i][2]


    remaining_index_list = indices[fitted_index==False]
    
    if len(all_remaining_index) == 0:
        all_remaining_index = copy.copy(remaining_index_list)
        print(all_remaining_index.shape)
    else:
        all_remaining_index = np.concatenate((all_remaining_index, remaining_index_list), axis=None)
        print(all_remaining_index.shape, remaining_index_list.shape)

remaining_point_list = []
remaining_geon_list = []
for index in all_remaining_index:
    remaining_point_list.append(point_list[index,:])
    remaining_geon_list.append(geon_label_list[index])

remaining_point_list = np.asarray(remaining_point_list)

#ax.scatter(remaining_point_list[:,0],remaining_point_list[:,1],remaining_point_list[:,2],\
#                zdir='z', s=1, c='C{}'.format(4), rasterized=True, alpha=0.5)

remaining_point_list = remaining_point_list + center_of_mess

fout = open('{}'.format(args.output_txt), mode='w')

for point_idx in range(remaining_point_list.shape[0]):
    fout.write('{} {} {} {}\n'.format(remaining_point_list[point_idx, 0],
                                      remaining_point_list[point_idx, 1],
                                      remaining_point_list[point_idx, 2],
                                      remaining_geon_list[point_idx]))


utils.axisEqual3D(ax)
plt.savefig('./test.png', bbox_inches='tight')
plt.close()
