#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################


import numpy as np
from .poly_functions import Polygon, fit_plane
from .MinimumBoundingBox import MinimumBoundingBox as mbr
from shapely.geometry import Point


def plane_intersect(p1, p2):
    '''
    Calculate intersected line between two planes
    :param p1:
    :param p2:
    :return: a line cross[x,y,0] and the normal vector[e,f,g]
    '''
    [a1, b1, c1, d1] = p1
    [a2, b2, c2, d2] = p2
    return [
        (b1*d2-b2*d1)/(a1*b2-a2*b1),
        (a1*d2-a2*d1)/(a2*b1-a1*b2),
        b1*c2-b2*c1,
        a2*c1-a1*c2,
        a1*b2-a2*b1
    ]


def point_in_plane(point, planes):
    '''
    Get plane index where point locates
    :param point:
    :param planes:
    :return: plane index
    '''
    p = Point(point)
    surfs = [surf[:, 0:2] for surf in planes]
    surfs = [Polygon(plane) for plane in surfs]

    for i in range(len(surfs)):
        if surfs[i].contains(p):
            return i

    return False


def get_z_from_plane(flag, polycenter, planes):
    '''
    Give a 2D coordinate(polycenter) and get the Z coordinate from the plane
    :param flag: plane index
    :param polycenter: 2D coordinate
    :param planes: 3D plane
    :return: Z coordinate
    '''
    if flag is not False:
        [x, y] = polycenter
        [a, b, c, d] = fit_plane(planes[flag])
        return (-d - a*x - b*y)/c
    else:
        return np.mean([plane[:, 2].max() for plane in planes])


def get_z_from_bottom(planes):
    '''
    Get average Z coordinate from planes
    :param planes:
    :return:
    '''
    return np.mean([plane[:, 2].min() for plane in planes])


def get_roof_line_theta(surfs):
    '''
    Get roof line angle from surfaces
    :param surfs:
    :return:
    '''
    roof_thetas = []
    for surf1, surf2 in zip(surfs[0::2], surfs[1::2]):
        p1 = fit_plane(surf1)
        p2 = fit_plane(surf2)
        intersect_line = plane_intersect(p1, p2)

        pn_2d = [intersect_line[2], intersect_line[3]]
        pn_2d = np.array(
            [pn_2d[0] / np.sqrt(pn_2d[0] ** 2 + pn_2d[1] ** 2),
             pn_2d[1] / np.sqrt(pn_2d[0] ** 2 + pn_2d[1] ** 2)])
        roof_thetas.append(np.arctan(pn_2d[1]/pn_2d[0]))
    return np.mean(roof_thetas)


def point_dist(point, fit_polygon):
    '''
    Get minimum distance from a point to a fit polygon
    :param point:
    :param fit_polygon:
    :return: distance
    '''
    area = []
    length = []
    for i in range(fit_polygon.shape[0] - 1):
        area.append(Polygon([point, fit_polygon[i], fit_polygon[i + 1]]).area)
        length.append(np.linalg.norm(fit_polygon[i] - fit_polygon[i + 1]))
    area = np.array(area)
    length = np.array(length)
    dist = area*2 / length
    return dist.min()


def get_error(points, poly, polygon_height):
    '''
    Get mean distance error from points to their fit polygon
    :param points:
    :param poly:
    :param polygon_height:
    :return:
    '''
    fit_polygon = poly
    fit_polygon.append(fit_polygon[0])
    last_colum = np.full(5, polygon_height)
    fit_polygon = np.c_[np.array(fit_polygon), last_colum]
    error = 0
    for point in points:
        error += point_dist(point, fit_polygon)

    return error/points.shape[0]


def add_box_geon(id, topsurf, bottomsurf, offset):
    '''
    Get box geon dictionary
    :param id: geon ID
    :param topsurf:
    :param bottomsurf:
    :param offset:
    :return: box geon dictionary
    '''

    surf2d = topsurf[:, 0:2]
    surf_mbb = mbr(surf2d)
    poly = surf_mbb.corner_points
    error = get_error(topsurf, poly, bottomsurf[:, 2].mean())
    surf_mbb_rm = np.array([
        [np.cos(surf_mbb.unit_vector_angle), -np.sin(surf_mbb.unit_vector_angle), 0],
        [np.sin(surf_mbb.unit_vector_angle), np.cos(surf_mbb.unit_vector_angle), 0],
        [0, 0, 1]])
    affine_m = surf_mbb_rm
    center = np.array([
        surf_mbb.corner_points[2][0], surf_mbb.corner_points[2][1], topsurf[:, 2].mean()])

    last_row = np.zeros(4)
    last_row[-1] = 1
    length = surf_mbb.length_parallel
    width = surf_mbb.length_orthogonal
    height = center[2] - bottomsurf[:, 2].mean()
    center[2] = center[2] - height
    affine_m = np.c_[affine_m, center - offset]
    affine_m = np.vstack((affine_m, last_row))
    return dict(type='rect_prism', id='box_' + str(id),
                transform=dict(affine_matrix=affine_m.tolist()), width=width,
                length=length, height=height), error


def add_mesh_geon(id, top_surf, bottom_surf, offset):
    '''
    Get mesh geon dictionary
    :param id: geon ID
    :param top_surf:
    :param bottom_surf:
    :param offset:
    :return: mesh geon dictionary
    '''
    top_surf -= offset
    bottom_surf -= offset
    pn = top_surf.shape[0]
    top_index = list(range(pn))
    bottom_index = list(range(pn, 2 * pn))
    wall_index = []
    point_cor = np.r_[top_surf, bottom_surf]
    wall_index.append(top_index)
    wall_index.append(bottom_index)
    for wi in range(pn):
        if wi == pn - 1:
            wall_index.append([top_index[wi], bottom_index[wi], bottom_index[0], top_index[0]])
        else:
            wall_index.append([top_index[wi], bottom_index[wi], bottom_index[wi + 1],
                               top_index[wi + 1]])

    return dict(type='mesh', id='mesh_' + str(id),
                transform=dict(affine_matrix=np.identity(4).tolist()),
                vertices_3d=point_cor.tolist(), faces=wall_index), 0


def add_shed_geon(id, surf, body_Z, offset):
    '''
    Get shed geon dictionary
    :param id: geon ID
    :param surf:
    :param body_Z: geon bottom Z coordinate
    :param offset:
    :return: shed geon dictionary
    '''
    pn = np.array(fit_plane(surf)[0:3])
    pn[2] = abs(pn[2])
    s_n = np.array([0, 0, 1])
    x_n = np.array([0, -1])
    surf2d = surf[:, 0:2]
    surf2d_mbr = mbr(surf2d)
    poly = surf2d_mbr.corner_points
    error = get_error(surf, poly, body_Z)
    pn_2d = np.array(
        [pn[0] / np.sqrt(pn[0] ** 2 + pn[1] ** 2), pn[1] / np.sqrt(pn[0] ** 2 + pn[1] ** 2)])
    theta = np.arccos(np.dot(pn, s_n) / (np.linalg.norm(pn) * np.linalg.norm(s_n)))
    ztheta = np.arccos(np.dot(x_n, pn_2d) / (np.linalg.norm(x_n) * np.linalg.norm(pn_2d)))
    if pn_2d[0] > 0 and pn_2d[1] > 0:
        ztheta = ztheta - np.pi / 2
    elif pn_2d[0] < 0:
        ztheta = 3 * np.pi / 2 - ztheta
    else:
        ztheta = 3 * np.pi / 2 + ztheta
    rm = np.array([[np.cos(ztheta), -np.sin(ztheta), 0],
                   [np.sin(ztheta), np.cos(ztheta), 0],
                   [0, 0, 1]])

    if abs(np.dot(pn_2d, surf2d_mbr.unit_vector)) > 1 - abs(np.dot(pn_2d, surf2d_mbr.unit_vector)):
        roof_width = surf2d_mbr.length_parallel
        roof_length = surf2d_mbr.length_orthogonal
    else:
        roof_length = surf2d_mbr.length_parallel
        roof_width = surf2d_mbr.length_orthogonal

    roof_width = roof_width/np.cos(theta)
    poly_center = list(surf2d_mbr.rectangle_center)
    poly_center.append(get_z_from_plane(0, poly_center, [surf]))
    poly_center = np.array(poly_center)
    height = poly_center[2] - body_Z
    center = poly_center - offset
    last_row = np.zeros(4)
    last_row[-1] = 1
    affine_m = np.c_[rm, center]
    affine_m = np.vstack((affine_m, last_row))

    return dict(type='shed', id='shed_' + str(id), transform=dict(affine_matrix=affine_m.tolist()),
                width=roof_width, length=roof_length, height=height, theta=theta), error


def add_gable_geon(id, surfs, body_Z, offset):
    '''
    Get gable geon dictionary
    :param id:
    :param surfs: gable roofs
    :param body_Z: gable roofs bottom Z coordinate
    :param offset:
    :return: gable geon dictionary
    '''
    theta = get_roof_line_theta(surfs)
    pn_2d = np.array([np.cos(theta), np.sin(theta)])
    # Merge all gable roofs
    surf2d = np.vstack(tuple(surfs))[:, 0:2]
    surf2d_mbr = mbr(surf2d)
    poly = surf2d_mbr.corner_points
    error = get_error(surf2d, poly, body_Z)
    if abs(np.dot(pn_2d, surf2d_mbr.unit_vector)) > 1 - abs(np.dot(pn_2d, surf2d_mbr.unit_vector)):
        roof_length = surf2d_mbr.length_parallel
        roof_width = surf2d_mbr.length_orthogonal
        theta = surf2d_mbr.unit_vector_angle
    else:
        roof_width = surf2d_mbr.length_parallel
        roof_length = surf2d_mbr.length_orthogonal
        theta = surf2d_mbr.unit_vector_angle + np.pi/2

    poly_center = list(surf2d_mbr.rectangle_center)
    flag = point_in_plane(poly_center, surfs)
    poly_center.append(get_z_from_plane(flag, poly_center, surfs))
    roof_bottom_height = get_z_from_bottom(surfs)
    body_height = roof_bottom_height - body_Z
    gabel_roof_height = poly_center[2] - roof_bottom_height
    center = np.array(poly_center) - offset
    rm = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    last_row = np.zeros(4)
    last_row[-1] = 1
    affine_m = np.c_[rm, center]
    affine_m = np.vstack((affine_m, last_row))
    return dict(type='gable', id='gable_' + str(id),
                transform=dict(affine_matrix=affine_m.tolist()), width=roof_width,
                length=roof_length, roof_height=gabel_roof_height, body_height=body_height), error
