#!/usr/bin/env python

import copy

from .poly_functions import get_height_from_dem, list_intersect
from .base_surface import Building


class Curved_building(Building):

    def __init__(self):
        Building.__init__(self)
        self.body_num = 0
        self.top_curved_surface = []
        self.bottom_curved_surface = []
        self.top_curved_surface_index = []
        self.top_boundary_index = []
        self.bottom_boundary_index = []
        self.surface_num = 0
        self.wall_num = 0
        self.geon_type = []

    def get_bottomsurface(self, dem):
        self.bottom_curved_surface = copy.deepcopy(self.top_curved_surface)
        for i in range(0, self.body_num):
            temp_cor = self.bottom_curved_surface[i]
            self.bottom_curved_surface[i][:, 2] = get_height_from_dem(temp_cor, dem)

    def add_topsurface(self, top_surface, index, geon_type='curve'):
        self.top_curved_surface.append(top_surface)
        self.top_curved_surface_index.append(index)
        self.body_num += 1
        self.surface_num += 1
        self.wall_num += 4
        self.geon_type.append(geon_type)

    def get_obj_string(self, offset):

        objs = []
        point_flag = 0
        for i in range(0, self.body_num):
            curved_surface_index = []
            curved_surface_index_str = []
            curved_surface_cor_str = []
            bottom_surface_index = []
            bottom_surface_index_str = []
            bottom_surface_cor_str = []
            intersect_line = []
            wall_str = []

            pn = self.top_curved_surface[i].shape[0]
            # surface info: vertex num, edge num, area
            self.surface_info.append([pn * 2, pn * 3, 0])

            for v in self.top_curved_surface[i]:
                v = v - offset
                tl = v.tolist()
                tl = [str(t) for t in tl]
                curved_surface_cor_str.append('v ' + ' '.join(tl) + '\n')

            for v in self.bottom_curved_surface[i]:
                v = v - offset
                tl = v.tolist()
                tl = [str(t) for t in tl]
                bottom_surface_cor_str.append('v ' + ' '.join(tl) + '\n')

            for l in self.top_curved_surface_index[i]:
                curved_surface_index.append([si + point_flag for si in l.tolist()])
                bottom_surface_index.append([si + point_flag + pn for si in l.tolist()])
                curved_surface_index_str.append(
                    'f ' + ' '.join([str(si + point_flag) for si in l.tolist()]) + "\n")
                bottom_surface_index_str.append(
                    'f ' + ' '.join([str(si + point_flag + pn) for si in l.tolist()]) + "\n")

            for i1 in range(0, self.top_curved_surface_index[i].shape[0]):
                for i2 in range(i1, self.top_curved_surface_index[i].shape[0]):
                    if len(list_intersect(self.top_curved_surface_index[i][i1],
                                          self.top_curved_surface_index[i][i2])) == 2:
                        intersect_line.append(
                            list_intersect(self.top_curved_surface_index[i][i1],
                                           self.top_curved_surface_index[i][i2]))

            for j in range(0, len(self.top_curved_surface_index[i])):
                surface_pn = len(curved_surface_index[j])
                for k in range(0, surface_pn):
                    flag = 0
                    if k == surface_pn - 1:
                        temp_wall_str = [
                            curved_surface_index[j][k] - point_flag,
                            curved_surface_index[j][0] - point_flag
                        ]
                        for l in intersect_line:
                            if len(list_intersect(temp_wall_str, l)) == 2:
                                flag = 1
                                break
                            else:
                                flag = 0
                        if flag == 0:
                            temp_wall_index = [curved_surface_index[j][k],
                                               bottom_surface_index[j][k],
                                               bottom_surface_index[j][0],
                                               curved_surface_index[j][0]]
                            wall_str.append(
                                'f ' + ' '.join([str(l) for l in temp_wall_index]) + '\n')
                    else:
                        temp_wall_str = [
                            curved_surface_index[j][k] - point_flag,
                            curved_surface_index[j][k + 1] - point_flag
                        ]
                        for l in intersect_line:
                            if len(list_intersect(temp_wall_str, l)) == 2:
                                flag = 1
                                break
                            else:
                                flag = 0
                        if flag == 0:
                            temp_wall_index = [curved_surface_index[j][k],
                                               bottom_surface_index[j][k],
                                               bottom_surface_index[j][k + 1],
                                               curved_surface_index[j][k + 1]]
                            wall_str.append(
                                'f ' + ' '.join([str(l) for l in temp_wall_index]) + '\n')
            point_flag += pn

            s = "o Mesh" + str(i) + "\ng Mesh" + str(i) + "\n" + ''.join(curved_surface_cor_str) + \
                ''.join(bottom_surface_cor_str) + ''.join(curved_surface_index_str) + \
                ''.join(bottom_surface_index_str) + ''.join(wall_str)
            objs.append(s)

        return objs

    def get_top_string(self, offset):
        objs = []
        point_flag = 0
        for i in range(0, self.body_num):
            curved_surface_index_str = []
            curved_surface_cor_str = []

            pn = self.top_curved_surface[i].shape[0]

            for v in self.top_curved_surface[i]:
                v = v - offset
                tl = v.tolist()
                tl = [str(t) for t in tl]
                curved_surface_cor_str.append('v ' + ' '.join(tl) + '\n')

            for l in self.top_curved_surface_index[i]:
                curved_surface_index_str.append(
                    'f ' + ' '.join([str(si) for si in l.tolist()]) + "\n")

            point_flag += pn

            s = "o Mesh" + str(i) + "\ng Mesh" + str(i) + "\n" + ''.join(curved_surface_cor_str) + \
                ''.join(curved_surface_index_str)
            objs.append(s)

        return objs

    def get_flatsurface(self):
        return "get_flatsurface"

    def split_surface(self):
        return "split_surface"
