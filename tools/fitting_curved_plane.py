#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################


import sys
import pickle
import copy
import argparse

from danesfield.geon_fitting.tensorflow import two_D_fitting
from danesfield.geon_fitting.tensorflow import utils
import numpy as np
import pcl
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def fit_cylinder(points, max_r=80, min_r=40):
    section_pc = pcl.PointCloud()
    section_pc.from_array(points)

    cylinder_seg = section_pc.make_segmenter_normals(ksearch=50)
    cylinder_seg.set_optimize_coefficients(True)
    cylinder_seg.set_model_type(pcl.SACMODEL_CYLINDER)
    cylinder_seg.set_normal_distance_weight(0.1)
    cylinder_seg.set_method_type(pcl.SAC_RANSAC)
    cylinder_seg.set_max_iterations(1000)
    cylinder_seg.set_distance_threshold(3)
    cylinder_seg.set_radius_limits(min_r, max_r)
    cylinder_indices, cylinder_coefficients = cylinder_seg.segment()

    return cylinder_indices, cylinder_coefficients


def fit_sphere(points):
    section_pc = pcl.PointCloud()
    section_pc.from_array(points)

    sphere_seg = section_pc.make_segmenter_normals(ksearch=50)
    sphere_seg.set_optimize_coefficients(True)
    sphere_seg.set_model_type(pcl.SACMODEL_SPHERE)
    sphere_seg.set_normal_distance_weight(0.3)
    sphere_seg.set_method_type(pcl.SAC_RANSAC)
    sphere_seg.set_max_iterations(1000)
    sphere_seg.set_distance_threshold(2)
    sphere_seg.set_radius_limits(5, 20)
    sphere_indices, sphere_coefficients = sphere_seg.segment()

    min_lst = []
    fitted_indices = []
    max_lst = []
    if len(sphere_indices) > 100:
        sphere_points = points[sphere_indices, :]
        sphere_indices = np.asarray(sphere_indices)
        points_z = sphere_points[:, 2]-sphere_coefficients[2]
        if np.max(points_z)-np.min(points_z) < 8:
            sphere_indices = []
            return [], sphere_coefficients, min_lst, max_lst
        min_lst, max_lst, fitted_indices = two_D_fitting.get_z_length(
            points_z, sphere_indices)
        for i in range(len(max_lst)):
            if min_lst[i] < -0.8*sphere_coefficients[-1]:
                min_lst[i] = -1*sphere_coefficients[-1]
            if max_lst[i] > 0.9*sphere_coefficients[-1]:
                max_lst[i] = sphere_coefficients[-1]
    return sphere_indices, sphere_coefficients, min_lst, max_lst


def check_sphere(points, c, r):
    distance = points - c
    distance = distance * distance
    distance = np.sqrt(np.sum(distance, axis=1))
    error = distance-r
    sphere_indices = np.arange(points.shape[0])
    sphere_indices = sphere_indices[np.logical_and(error < 3, error > -3)]
    return sphere_indices, error


def get_theta(length, r):
    if length < -0.8*r:
        return np.pi
    if length > 0.9*r:
        return 0
    if length < 0:
        return np.pi-np.arccos((-1*length)/r)
    return np.arccos(length/r)


def draw_sphere(ax, c, r, z_min, z_max):
    theta_max = get_theta(z_min, r)
    theta_min = get_theta(z_max, r)
    u, v = np.mgrid[0:2*np.pi:10j, theta_min:theta_max:10j]
    x = np.cos(u)*np.sin(v)*r
    y = np.sin(u)*np.sin(v)*r
    z = np.cos(v)*r
    x = x + c[0]
    y = y + c[1]
    z = z + c[2]
    ax.plot_wireframe(x, y, z, color='r', alpha=0.5)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_pc',
        # default='/home/xuzhang/project/Core3D/danesfield_gitlab/danesfield/geon_fitting/outlas/out_D4.txt',
        type=str,
        help='Input labelled point cloud. The point cloud should has geon type label, \
        output txt from roof_segmentation.py. ')
    parser.add_argument(
        '--output_png',
        # default='../segmentation_graph/out.png',
        type=str,
        help='Output png result file.')
    parser.add_argument(
        '--output_txt',
        # default='../outlas/remain_D4.txt',
        type=str,
        help='Output txt result file includes all planar points for Purdue to process.')
    parser.add_argument(
        '--output_geon',
        # default='../out_geon/D4_Curve_Geon.npy',
        type=str,
        help='Output geon file.')
    args = parser.parse_args(args)

    point_list, building_label_list, geon_label_list = utils.read_geon_type_pc(
        args.input_pc)
    center_of_mess = np.mean(point_list, axis=0)
    point_list = point_list - center_of_mess
    point_list = point_list.astype(np.float32)
    cloud_filtered = pcl.PointCloud()
    cloud_filtered.from_array(point_list)

    print(cloud_filtered.size)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    geon_model = []
    # all_vertex = []
    # all_face = []
    all_remaining_index = []
    index = 0

    building_index_list = []
    building_max_index = np.max(building_label_list)
    total_list = np.arange(building_label_list.shape[0])
    for i in range(building_max_index+1):
        current_list = total_list[building_label_list == i]
        building_index_list.append(current_list)

    geon_type_number = 4
    cylinder_index = 2
    sphere_index = 3

    point_number_scale = 1
    # c_index = 0
    for indices in building_index_list:
        if len(indices) < 300:
            if len(all_remaining_index) == 0:
                all_remaining_index = copy.copy(indices)
            else:
                all_remaining_index = np.concatenate(
                    (all_remaining_index, indices), axis=None)
            continue

        geon_index_list = []
        for i in range(geon_type_number):
            geon_index_list.append([])

        building_points = np.zeros((len(indices), 3), dtype=np.float32)
        num_building_points = len(indices)

        for i, indice in enumerate(indices):
            building_points[i][0] = cloud_filtered[indice][0]
            building_points[i][1] = cloud_filtered[indice][1]
            building_points[i][2] = cloud_filtered[indice][2]

        for i, indice in enumerate(indices):
            geon_index_list[geon_label_list[indice]].append(indice)

        fitted_index = np.zeros(len(indices), dtype=np.int32)
        fitted_index = fitted_index == 1

        if len(geon_index_list[cylinder_index]) > 0.1*len(indices):
            points = np.zeros(
                (len(geon_index_list[cylinder_index]), 3), dtype=np.float32)
            for i, indice in enumerate(geon_index_list[cylinder_index]):
                points[i][0] = cloud_filtered[indice][0]
                points[i][1] = cloud_filtered[indice][1]
                points[i][2] = cloud_filtered[indice][2]

            current_cloud = pcl.PointCloud()
            current_cloud.from_array(points)

            num_current_cylinder_point = current_cloud.size
            if num_building_points > 15000:
                vg = current_cloud.make_voxel_grid_filter()
                vg.set_leaf_size(1, 1, 1)
                current_cloud = vg.filter()

            num_filtered_building_points = current_cloud.size

            current_points = np.zeros((current_cloud.size, 3), dtype=np.float32)
            for i in range(current_cloud.size):
                current_points[i] = current_cloud[i]

            max_r = 80
            min_r = 40
            if num_current_cylinder_point > 10000:
                max_r = 80
                min_r = 40
            else:
                max_r = 30
                min_r = 10

            while True:
                cylinder_indices, cylinder_coefficients = fit_cylinder(
                    current_points, max_r, min_r)

                if len(cylinder_indices) < 1000*point_number_scale:
                    break

                cylinder_points = np.zeros(
                    (len(cylinder_indices), 3), dtype=np.float32)
                for i, indice in enumerate(cylinder_indices):
                    cylinder_points[i][0] = current_points[indice][0]
                    cylinder_points[i][1] = current_points[indice][1]
                    cylinder_points[i][2] = current_points[indice][2]

                (centroid,
                 ex,
                 ey,
                 ez,
                 fitted_indices,
                 coefficients,
                 min_axis_z,
                 max_axis_z,
                 mean_diff) = two_D_fitting.fit_2D_curve(cylinder_coefficients[3:-1],
                                                         cylinder_points,
                                                         fit_type='poly2',
                                                         dist_threshold=10)

                for i in range(len(fitted_indices)):

                    if len(fitted_indices[i]) < max(500, 0.05*num_filtered_building_points):
                        continue

                    fitted_points = np.zeros(
                        (len(fitted_indices[i]), 3), np.float32)
                    for j, tmp_idx in enumerate(fitted_indices[i]):
                        fitted_points[j, :] = cylinder_points[tmp_idx, :]

                    # fitted_wire = utils.draw_poly_curve(
                    #     ax,
                    #     centroid,
                    #     ex,
                    #     ey,
                    #     fitted_points,
                    #     coefficients,
                    #     min_axis_z[i],
                    #     max_axis_z[i],
                    #     'C{}'.format(2))
                    ax.scatter(fitted_points[:, 0], fitted_points[:, 1], fitted_points[:, 2],
                               zdir='z', s=1, c='C{}'.format(2), rasterized=True, alpha=0.5)

                    (all_fitted_indices,
                     ortho_x_max,
                     ortho_x_min,
                     error) = two_D_fitting.check_2D_curve(ex,
                                                           ey,
                                                           ez,
                                                           coefficients,
                                                           centroid,
                                                           building_points,
                                                           min_axis_z[i],
                                                           max_axis_z[i],
                                                           fit_type='poly2')
                    fitted_index[all_fitted_indices] = True

                    # ortho_x = np.matmul(fitted_points - centroid, ex)
                    geon_model.append({'name': 'poly_cylinder', 'model':
                                       [centroid, ex, ey, coefficients, min_axis_z[i],
                                        max_axis_z[i], ortho_x_min, ortho_x_max,
                                        len(fitted_indices[i]), mean_diff]})

                current_cloud = current_cloud.extract(cylinder_indices, True)
                if current_cloud.size < max(500, 0.1*num_filtered_building_points):
                    break

                current_points = np.zeros(
                    (current_cloud.size, 3), dtype=np.float32)
                for i in range(current_cloud.size):
                    current_points[i][0] = current_cloud[i][0]
                    current_points[i][1] = current_cloud[i][1]
                    current_points[i][2] = current_cloud[i][2]

        # print([len(x) for x in geon_index_list])
        if len(geon_index_list[sphere_index]) > 0.3*len(indices):
            points = np.zeros(
                (len(geon_index_list[sphere_index]), 3), dtype=np.float32)
            for i, indice in enumerate(geon_index_list[sphere_index]):
                points[i][0] = cloud_filtered[indice][0]
                points[i][1] = cloud_filtered[indice][1]
                points[i][2] = cloud_filtered[indice][2]

            current_cloud = pcl.PointCloud()
            current_cloud.from_array(points)

            if num_building_points > 10000:
                vg = current_cloud.make_voxel_grid_filter()
                vg.set_leaf_size(1, 1, 1)
                current_cloud = vg.filter()

            current_points = np.zeros((current_cloud.size, 3), dtype=np.float32)
            for i in range(current_cloud.size):
                current_points[i] = current_cloud[i]

            while True:
                sphere_indices, sphere_coefficients, min_lst, max_lst = fit_sphere(
                    current_points)
                if len(sphere_indices) < 200*point_number_scale:
                    break

                if sphere_coefficients[-1] > 0:
                    draw_sphere(
                        ax,
                        sphere_coefficients[0:3],
                        sphere_coefficients[-1],
                        min_lst[0],
                        max_lst[0])

                sphere_points = np.zeros(
                    (len(sphere_indices), 3), dtype=np.float32)
                for i, indice in enumerate(sphere_indices):
                    sphere_points[i][0] = current_points[indice][0]
                    sphere_points[i][1] = current_points[indice][1]
                    sphere_points[i][2] = current_points[indice][2]

                ax.scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2],
                           zdir='z', s=1, c='C{}'.format(3), rasterized=True, alpha=0.5)

                geon_model.append({'name': 'sphere', 'model': [sphere_coefficients[0:3],
                                                               sphere_coefficients[-1],
                                                               min_lst[0],
                                                               max_lst[0],
                                                               len(sphere_indices)]})

                all_fitted_indices, error = check_sphere(
                    building_points, sphere_coefficients[0:3], sphere_coefficients[-1])
                fitted_index[all_fitted_indices] = True

                current_cloud = current_cloud.extract(sphere_indices, True)
                if current_cloud.size < 1000*point_number_scale:
                    break

                current_points = np.zeros(
                    (current_cloud.size, 3), dtype=np.float32)
                for i in range(current_cloud.size):
                    current_points[i][0] = current_cloud[i][0]
                    current_points[i][1] = current_cloud[i][1]
                    current_points[i][2] = current_cloud[i][2]

        remaining_index_list = indices[fitted_index == False]  # noqa: E712

        if len(all_remaining_index) == 0:
            all_remaining_index = copy.copy(remaining_index_list)
        else:
            all_remaining_index = np.concatenate(
                (all_remaining_index, remaining_index_list), axis=None)

    remaining_point_list = []
    remaining_geon_list = []
    for index in all_remaining_index:
        remaining_point_list.append(point_list[index, :])
        remaining_geon_list.append(geon_label_list[index])

    remaining_point_list = np.asarray(remaining_point_list)

    show_cloud = pcl.PointCloud()
    show_cloud.from_array(remaining_point_list)

    vg = show_cloud.make_voxel_grid_filter()
    vg.set_leaf_size(2, 2, 2)
    show_cloud = vg.filter()
    show_points = np.zeros((show_cloud.size, 3), dtype=np.float32)
    for i in range(show_cloud.size):
        show_points[i, :] = show_cloud[i]

    ax.scatter(show_points[:, 0], show_points[:, 1], show_points[:, 2],
               zdir='z', s=1, c='C{}'.format(9), alpha=0.01)

    remaining_point_list = remaining_point_list + center_of_mess

    fout = open('{}'.format(args.output_txt), mode='w')

    for point_idx in range(remaining_point_list.shape[0]):
        fout.write('{} {} {} {}\n'.format(remaining_point_list[point_idx, 0],
                                          remaining_point_list[point_idx, 1],
                                          remaining_point_list[point_idx, 2],
                                          remaining_geon_list[point_idx]))

    utils.axisEqual3D(ax)
    plt.savefig(args.output_png, bbox_inches='tight')
    plt.close()

    pickle.dump([center_of_mess, geon_model], open(args.output_geon, "wb"))


if __name__ == "__main__":
    main(sys.argv[1:])
