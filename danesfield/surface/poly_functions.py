#!/usr/bin/env python

import re
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union


def list_intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))


def list_union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))


def ply_parser(fp):
    '''
    :param fp: PLY file path
    :return: Surface coordinates and surface index
    '''
    tf = open(fp)
    lines = tf.readlines()
    flag = 0
    for l in lines:
        if re.search("\s*element\s*vertex\s*\d*", l) is not None:
            vertex_num = int(re.findall("\d+\.?\d*", l)[0])
        if re.search("\s*element\s*face\s*\d*", l) is not None:
            face_num = int(re.findall("\d+\.?\d*", l)[0])
        if re.search("end_header", l) is not None:
            begin_num = flag + 1
        flag += 1

    x = [float(re.findall("-*\d+\.?\d*", l)[0]) for l in lines[begin_num:begin_num + vertex_num]]
    y = [float(re.findall("-*\d+\.?\d*", l)[1]) for l in lines[begin_num:begin_num + vertex_num]]
    z = [float(re.findall("-*\d+\.?\d*", l)[2]) for l in lines[begin_num:begin_num + vertex_num]]

    cor = [[x[i], y[i], z[i]] for i in range(0, len(x))]
    cor = np.asarray(cor)
    f = [re.findall("\d+\.?\d*", l)
         for l in lines[begin_num + vertex_num:begin_num + vertex_num + face_num]]

    return cor, f


def check_relation(plane1, plane2):
    '''
    Checking spatial relationship between planes.
    :param plane1:
    :param plane2:
    :return: spatial relationship tag
    '''
    p1 = Polygon(plane1)
    p2 = Polygon(plane2)

    try:
        if p1.intersects(p2):
            if p1.contains(p2):
                flag = 1
            else:
                if p1.area >= p2.area:
                    flag = 2
                else:
                    flag = 3
        else:
            flag = 4
        return flag
    except:
        return 4


def get_height_from_dem(cor, dem_parameter):
    '''
    Get Z coordinate from DEM based on given XY coordinate.
    r1-r4 represent the image boundaries for coordinates outside.
    :param cor: XY coordinate
    :param dem: DEM object
    :return: Z coordinate
    '''
    xOrigin = dem_parameter[0]
    yOrigin = dem_parameter[1]
    pixelWidth = dem_parameter[2]
    pixelHeight = dem_parameter[3]
    data = dem_parameter[4]
    r = dem_parameter[5]
    base_height = []
    for i in range(cor.shape[0]):
        x = cor[i, 0]
        y = cor[i, 1]
        xOffset = int((x - xOrigin) / pixelWidth)
        yOffset = int((y - yOrigin) / pixelHeight)
        try:
            value = data[yOffset][xOffset]
            base_height.append(value)
        except:
            dist_2 = np.sum((r - np.array([yOffset, xOffset])) ** 2, axis=1)
            index = np.argmin(dist_2)
            value = data[r[index, 0]][r[index, 1]]
            base_height.append(value)
    return np.array(base_height)


def get_height_from_lower_surface(plane1, plane2):
    '''
    :param plane1: Higher surface
    :param plane2: Lower surface
    :return: Z coordinate on lower surface
    '''
    [a, b, c, d] = fit_plane(plane1)

    def z(x):
        return -(a * x[0] + b * x[1] + d) / c
    return z([plane2[:, 0], plane2[:, 1]])


def get_difference_plane(plane1, plane2):
    '''
    Get difference and intersection part for two planes
    :param plane1:
    :param plane2:
    :return:
    '''
    try:
        p1 = Polygon(plane1)
        p2 = Polygon(plane2)
        pd = p2.difference(p1)
        pi = p2.intersection(p1)
        flag = True
        p3 = np.array(pd.exterior.coords[:])
        p4 = np.array(pi.exterior.coords[:])
        return [flag, p3, p4]
    except:
        flag = False
        p3 = None
        p4 = None
        return [flag, p3, p4]


def fit_plane(point):
    '''
    Using normal vector and distance to origin to represent a plane.
    :param point: Plane coordinates
    :return: Plane parameters
    '''
    xyz_mean = np.array([point[:, 0].mean(), point[:, 1].mean(), point[:, 2].mean()])
    xyz_m = np.array(
        [point[:, 0] - xyz_mean[0], point[:, 1] - xyz_mean[1], point[:, 2] - xyz_mean[2]])
    [U, S, V] = np.linalg.svd(xyz_m)
    v = np.array([U[0, 2], U[1, 2], U[2, 2]])
    a = v[0]
    b = v[1]
    c = v[2]
    d = - np.dot(v, xyz_mean.transpose())
    # normal vector of plane
    return [a, b, c, d]


def rotate_plane(plane):
    '''
    Rotate a 3D plane into 2D plane.
    :param plane:
    :return: [2D plane coordinates, rotate tag(whether or not), rotation matrix, plane center]
    '''
    temp_cor = plane
    p_n = fit_plane(temp_cor)
    p_n = np.array(p_n[0:3])
    s_n = np.array([0, 0, 1])
    [rx, ry, rz] = np.cross(p_n, s_n)
    ra = np.arccos(np.dot(p_n, s_n) / (np.linalg.norm(p_n) * np.linalg.norm(s_n)))
    rotate_flag = False
    rm = None
    center = None
    if abs(ra) > 0.001:
        norm = np.linalg.norm(np.cross(p_n, s_n))
        [rx, ry, rz] = [rx / norm, ry / norm, rz / norm]
        r1 = [np.cos(ra) + rx ** 2 * (1 - np.cos(ra)), rx * ry * (1 - np.cos(ra)) - rz * np.sin(ra),
              ry * np.sin(ra) + rx * rz * (1 - np.cos(ra))]
        r2 = [rz * np.sin(ra) + rx * ry * (1 - np.cos(ra)), np.cos(ra) + ry ** 2 * (1 - np.cos(ra)),
              -rx * np.sin(ra) + ry * rz * (1 - np.cos(ra))]
        r3 = [-ry * np.sin(ra) + rx * rz * (1 - np.cos(ra)),
              rx * np.sin(ra) + ry * rz * (1 - np.cos(ra)),
              np.cos(ra) + rz ** 2 * (1 - np.cos(ra))]
        rm = np.array([r1, r2, r3])
        center = [np.mean(temp_cor[:, 0]), np.mean(temp_cor[:, 1]), np.mean(temp_cor[:, 2])]
        cor_2d = np.dot(rm, (temp_cor - center).transpose()).transpose()
        rotate_flag = True
    else:
        cor_2d = temp_cor

    return [cor_2d, rotate_flag, rm, center]


def remove_close_point(plane, T=1e-6):
    '''
    Remove close points in a surface
    :param plane:
    :param T: Threshold
    :return: New plane coordinates
    '''
    origin_plane = plane
    test_plane = plane[:, 0:2]
    del_list = []
    for i in range(0, test_plane.shape[0]):
        for j in range(i+1, test_plane.shape[0]):
            dist = np.linalg.norm(test_plane[i] - test_plane[j])
            if dist <= T:
                del_list.append(i)
    plane = np.delete(plane, del_list, axis=0)
    if plane.shape[0] < 3:
        return origin_plane
    else:
        return plane


def fix_intersection(plane):
    '''
    Solve self-intersection issue
    :param plane: plane coordinates
    :return: None self-intersection plane coordinates
    '''
    if plane.shape[0] <= 4:
        return plane, False
    temp_cor = plane
    p_n = fit_plane(temp_cor)
    p_n = np.array(p_n[0:3])
    s_n = np.array([0, 0, 1])
    [rx, ry, rz] = np.cross(p_n, s_n)
    ra = np.arccos(np.dot(p_n, s_n) / (np.linalg.norm(p_n) * np.linalg.norm(s_n)))
    rotate_flag = False
    if abs(ra) > 0.001:
        norm = np.linalg.norm(np.cross(p_n, s_n))
        [rx, ry, rz] = [rx / norm, ry / norm, rz / norm]
        r1 = [np.cos(ra) + rx ** 2 * (1 - np.cos(ra)), rx * ry * (1 - np.cos(ra)) - rz * np.sin(ra),
              ry * np.sin(ra) + rx * rz * (1 - np.cos(ra))]
        r2 = [rz * np.sin(ra) + rx * ry * (1 - np.cos(ra)), np.cos(ra) + ry ** 2 * (1 - np.cos(ra)),
              -rx * np.sin(ra) + ry * rz * (1 - np.cos(ra))]
        r3 = [-ry * np.sin(ra) + rx * rz * (1 - np.cos(ra)),
              rx * np.sin(ra) + ry * rz * (1 - np.cos(ra)),
              np.cos(ra) + rz ** 2 * (1 - np.cos(ra))]
        rm = np.array([r1, r2, r3])
        center = [np.mean(temp_cor[:, 0]), np.mean(temp_cor[:, 1]), np.mean(temp_cor[:, 2])]
        cor_2d = np.dot(rm, (temp_cor - center).transpose()).transpose()
        rotate_flag = True
    else:
        cor_2d = temp_cor
    poly_cor = Polygon(cor_2d[:, 0:2])
    if poly_cor.is_valid:
        return plane, False
    else:
        try:
            ls = LineString(np.array(poly_cor.exterior.coords[:]))
            lr = LineString(ls.coords[:] + ls.coords[0:1])
            mls = unary_union(lr)
            t_cor = polygonize(mls)

            fixed_polys = []
            for polygon in t_cor:
                temp_cor = np.array(polygon.exterior.coords[:])
                temp_cor = np.c_[temp_cor, np.zeros(temp_cor.shape[0])]
                if rotate_flag:
                    temp_cor = np.dot(np.linalg.inv(rm), temp_cor.transpose()).transpose() + center
                else:
                    temp_cor[:, 2] = temp_cor[:, 2] + np.mean(plane[:, 2])

                fixed_polys.append(temp_cor)
            return fixed_polys, True
        except:
            return plane


def fix_height(plane, new_cor):
    '''
    Fit a un-flat plane by given coordinates and plane parameters
    :param plane: Plane parameters
    :param new_cor: Given plane XY coordinates
    :return: Fixed plane coordinates
    '''
    [a, b, c, d] = fit_plane(plane)

    def z(x):
        return -(a * x[0] + b * x[1] + d) / c
    height = z([new_cor[:, 0], new_cor[:, 1]])
    new_plane = np.c_[new_cor[:, 0], new_cor[:, 1], height]
    return new_plane


def counterClockwiseCheck(vertList):
    '''
    Check to see if a list of 2D vertices are clockwise
    :param vertList:
    :return:
    '''
    sum = 0
    for x in range(1, len(vertList)):
        v1 = vertList[x-1]
        v2 = vertList[x]
        t = (v2[0]-v1[0])*(v2[1]+v1[1])
        sum = sum + t
    v1 = vertList[len(vertList)-1]
    v2 = vertList[0]
    t = (v2[0]-v1[0])*(v2[1]+v1[1])

    sum = sum + t
    return sum < 0
