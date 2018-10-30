#!/usr/bin/env python

import copy
import numpy as np

from shapely.geometry import Polygon

from .poly_functions import (
    check_relation,
    counterClockwiseCheck,
    fix_height,
    fix_intersection,
    get_difference_plane,
    get_height_from_dem,
    get_height_from_lower_surface,
    rotate_plane,
)


class Surface:
    '''
    Surface class
    '''

    def __init__(self, cor):
        self.point_cor = cor


class Building:

    def __init__(self):
        self.surface_num = 0
        self.topsurface = []
        self.bottomsurface = []
        self.vertex_num = 0
        self.wall_num = 0
        self.edge_num = 0
        self.surface_info = []
        self.flatsurface = []
        self.scene_name = ''

    def get_flatsurface(self):
        '''
        Rotate 3D plane to 2D
        '''
        for i in range(self.surface_num):
            surf = self.topsurface[i]
            height = self.bottomsurface[i].point_cor[:, 2].mean()
            flat = rotate_plane(surf.point_cor)
            flat.append(height)
            self.flatsurface.append(flat)

    def add_topsurface(self, plane):
        '''
        Add building roof one by one
        '''
        temp_cor = plane.point_cor
        [fixed_polys, flag] = fix_intersection(temp_cor)
        if flag:
            for poly in fixed_polys:
                self.topsurface.append(Surface(poly))
                self.surface_num += 1
        else:
            self.topsurface.append(Surface(temp_cor))
            self.surface_num += 1

    def split_surface(self):
        '''
        Check planes spatial relationship and split intersected planes
        '''
        for i in range(0, self.surface_num):
            for j in range(0, self.surface_num):
                if i != j:
                    relationship_flag = check_relation(
                        self.topsurface[i].point_cor[:, 0:2], self.topsurface[j].point_cor[:, 0:2])
                    if relationship_flag == 2:
                        try:
                            rst = get_difference_plane(
                                self.topsurface[i].point_cor, self.topsurface[j].point_cor)
                            if rst[0]:
                                self.topsurface[j].point_cor = \
                                    fix_height(self.topsurface[j].point_cor, rst[1])
                                self.topsurface.append(
                                    Surface(fix_height(self.topsurface[j].point_cor,
                                                       rst[2])))
                                self.surface_num += 1
                        except Exception as e:
                            print(e)

    def get_bottomsurface(self, dem_parameter):
        '''
        Get bottom surface for each roof
        :param dem: DEM object
        '''
        self.bottomsurface = copy.deepcopy(self.topsurface)
        for i in range(0, self.surface_num):
            base_height = get_height_from_dem(self.bottomsurface[i].point_cor,
                                              dem_parameter)
            self.bottomsurface[i].point_cor[:, 2] = base_height
        for i in range(0, self.surface_num):
            for j in range(0, self.surface_num):
                if i != j:
                    relationship_flag = check_relation(
                        self.topsurface[i].point_cor, self.topsurface[j].point_cor)
                    if relationship_flag == 1:
                        base_height1 = get_height_from_lower_surface(
                            self.topsurface[i].point_cor, self.topsurface[j].point_cor)
                        self.bottomsurface[j].point_cor[:, 2] = base_height1

    def get_obj_string(self, offset):
        '''
        Generate obj file strings
        :param offset: offset for whole area
        '''
        objs = []
        point_flag = 1

        for i in range(self.surface_num):
            topstring = []
            bottomstring = []
            pn = self.topsurface[i].point_cor.shape[0]

            poly_check = self.topsurface[i].point_cor[:, 0:2]
            if not counterClockwiseCheck(poly_check):
                self.topsurface[i].point_cor = np.copy(np.flip(self.topsurface[i].point_cor, 0))

            poly_check = self.bottomsurface[i].point_cor[:, 0:2]
            if not counterClockwiseCheck(poly_check):
                self.bottomsurface[i].point_cor = \
                    np.copy(np.flip(self.bottomsurface[i].point_cor, 0))
            temp_surf = Polygon(self.topsurface[i].point_cor)
            # surface info: vertex num, edge num, area
            try:
                area = temp_surf.area
            except:  # noqa: E722
                area = 0
            self.surface_info.append([pn, pn, area])
            self.vertex_num += 2*pn
            self.edge_num += 3*pn
            top_index = [str(j) for j in range(point_flag, point_flag + pn)]
            top_string = 'f ' + ' '.join(top_index) + "\n"
            bottom_index = [str(j) for j in range(point_flag + pn, point_flag + 2*pn)]
            bottom_string = 'f ' + ' '.join(bottom_index) + "\n"
            wallstring = []
            for j in range(pn):
                if j == pn - 1:
                    wallstring.append('f ' + ' '.join([
                        top_index[j], bottom_index[j], bottom_index[0], top_index[0]
                    ]) + '\n')
                    self.wall_num += 1
                else:
                    wallstring.append('f ' + ' '.join([
                        top_index[j], bottom_index[j], bottom_index[j + 1], top_index[j + 1]
                    ]) + '\n')
                    self.wall_num += 1
            point_flag = point_flag + 2*pn
            for v in self.topsurface[i].point_cor:
                v = v - offset
                tl = v.tolist()
                tl = [str(t) for t in tl]
                topstring.append('v ' + ' '.join(tl) + '\n')
            for v in self.bottomsurface[i].point_cor:
                v = v - offset
                tl = v.tolist()
                tl = [str(t) for t in tl]
                bottomstring.append('v ' + ' '.join(tl) + '\n')

            s = "o Mesh" + str(i) + "\ng Mesh" + str(i) + "\n" +\
                ''.join(topstring) + ''.join(bottomstring) + top_string + bottom_string + \
                ''.join(wallstring)
            objs.append(s)

        return objs

    def get_top_string(self, offset):
        '''
        Generate top surface obj file strings
        '''
        objs = []
        point_flag = 1
        for i in range(self.surface_num):
            topstring = []
            pn = self.topsurface[i].point_cor.shape[0]
            top_index = [str(j) for j in range(point_flag, point_flag + pn)]
            top_string = 'f ' + ' '.join(top_index) + "\n"
            point_flag = point_flag + pn
            for v in self.topsurface[i].point_cor:
                v = v - offset
                tl = v.tolist()
                tl = [str(t) for t in tl]
                topstring.append('v ' + ' '.join(tl) + '\n')

            s = ''.join(topstring) + top_string
            objs.append(s)

        return objs
