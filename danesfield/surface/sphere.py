#!/usr/bin/env python

import copy
from .poly_functions import *
from .base_surface import Building
from shapely.geometry import Point
from shapely.geometry import MultiPoint


class Sphere_building(Building):

    def __init__(self):
        Building.__init__(self)
        self.surface_num = 0
        self.top_curved_surface = []
        self.bottom_curved_surface = []
        self.top_curved_surface_index = []
        self.top_boundary_index = []
        self.bottom_boundary_index = []
        self.wall_num = 0
        self.geon_type = []

    def get_bottomsurface(self, dem):
        self.bottom_curved_surface = copy.deepcopy(self.top_curved_surface)
        for i in range(0, self.surface_num):
            temp_cor = self.bottom_curved_surface[i]
            temp_cor = temp_cor[self.top_boundary_index[i]]
            self.bottom_curved_surface[i] = temp_cor
            self.bottom_curved_surface[i][:, 2] = get_height_from_dem(temp_cor, dem)

    def add_topsurface(self, top_surface, index, geon_type='sphere'):
        self.top_curved_surface.append(top_surface)
        self.top_curved_surface_index.append(index)
        hull = MultiPoint(top_surface[:, 0:2]).convex_hull
        boundary_index = []
        for i in range(top_surface.shape[0]):
            point = Point(top_surface[i][0:2])
            if not hull.contains(point):
                boundary_index.append(i)
        self.top_boundary_index.append(boundary_index)
        self.surface_num += 1
        self.wall_num += 1
        self.geon_type.append(geon_type)

    def get_obj_string(self, offset):

        objs = []
        point_flag = 0
        for i in range(0, self.surface_num):
            curved_surface_index = []
            curved_surface_index_str = []
            curved_surface_cor_str = []
            bottom_surface_cor_str = []
            wall_str = []
            boundary_num = len(self.top_boundary_index[i])

            pn = self.top_curved_surface[i].shape[0]
            # surface info: vertex num, edge num, area
            self.surface_info.append([pn + boundary_num, boundary_num * 3, 0])

            for l in self.top_curved_surface[i]:
                nl = l - offset
                tl = nl.tolist()
                tl = [str(t) for t in tl]
                curved_surface_cor_str.append('v ' + ' '.join(tl) + '\n')

            for l in self.bottom_curved_surface[i]:
                nl = l - offset
                tl = nl.tolist()
                tl = [str(t) for t in tl]
                bottom_surface_cor_str.append('v ' + ' '.join(tl) + '\n')

            for l in self.top_curved_surface_index[i]:
                curved_surface_index.append([si + point_flag for si in l.tolist()])
                curved_surface_index_str.append('f ' + 
                ' '.join([str(si + point_flag) for si in l.tolist()]) + "\n")

            surf_boundary_index = [str(si + point_flag) for si in self.top_boundary_index[i]]
            bottom_surface_index = [str(si + point_flag + boundary_num) for si in self.top_boundary_index[i]]
            bottom_surface_index_str = 'f ' + ' '.join([str(si + point_flag + boundary_num) for si in self.top_boundary_index[i]]) + "\n"

            for j in range(boundary_num):
                if j == boundary_num - 1:
                    wall_str.append(
                        'f ' + ' '.join([surf_boundary_index[j], bottom_surface_index[j], bottom_surface_index[0], surf_boundary_index[0]]) + '\n')
                else:
                    wall_str.append(
                        'f ' + ' '.join([surf_boundary_index[j], bottom_surface_index[j], bottom_surface_index[j + 1], surf_boundary_index[j + 1]]) + '\n')

            point_flag += pn + boundary_num

            s = "o Mesh" + str(i) + "\ng Mesh" + str(i) + "\n" + ''.join(curved_surface_cor_str) + ''.join(bottom_surface_cor_str) + ''.join(curved_surface_index_str) + ''.join(bottom_surface_index_str) + ''.join(wall_str)
            objs.append(s)

        return objs

    def get_top_string(self, offset):
        objs = []
        point_flag = 0
        for i in range(0, self.surface_num):
            curved_surface_index_str = []
            curved_surface_cor_str = []

            pn = self.top_curved_surface[i].shape[0]

            for l in self.top_curved_surface[i]:
                nl = l - offset
                tl = nl.tolist()
                tl = [str(t) for t in tl]
                curved_surface_cor_str.append('v ' + ' '.join(tl) + '\n')

            for l in self.top_curved_surface_index[i]:
                curved_surface_index_str.append('f ' + ' '.join([str(si) for si in l.tolist()]) + "\n")

            point_flag += pn
            self.vertex_num += point_flag

            s = "o Mesh" + str(i) + "\ng Mesh" + str(i) + "\n" + ''.join(curved_surface_cor_str) + ''.join(curved_surface_index_str)
            objs.append(s)

        return objs

    def get_flatsurface(self):
        return "get_flatsurface"

    def split_surface(self):
        return "split_surface"
