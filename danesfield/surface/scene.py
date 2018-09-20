#!/usr/bin/env python

import os
import sys
import time
import numpy as np
from osgeo import gdal
from pathlib import Path
from plyfile import PlyData
from .poly_functions import list_intersect, list_union, ply_parser
from .base_surface import Building
from .base_surface import Surface
from .curve_surface import Curved_building


class Model(object):
    def __init__(self):
        self.dem = None
        self.building_num = 0
        self.buildings = []
        self.building_name = []
        self.ply_path = ""
        self.obj_path = ""
        self.surface_path = ""
        self.x_offset = None
        self.y_offset = None
        self.z_offset = None
        self.surface_info_str = ''
        self.surface_num_total = 0
        self.top_num_total = 0
        self.bottom_num_total = 0
        self.wall_num_total = 0
        self.vertex_num_total = 0
        self.edge_num_total = 0
        self.offset_flag = True

    def get_offset(self, fp):
        try:
            plydata = PlyData.read(fp)
            if plydata['vertex'].count == 0:
                return

            cor = np.vstack((plydata['vertex']['x'],
                             plydata['vertex']['y'],
                             plydata['vertex']['z'])).transpose()

            if self.x_offset is None:
                self.x_offset = min(cor[:, 0])
                self.y_offset = min(cor[:, 1])
                self.z_offset = min(cor[:, 2])
            else:
                self.x_offset = min(self.x_offset, min(cor[:, 0]))
                self.y_offset = min(self.y_offset, min(cor[:, 1]))
                self.z_offset = min(self.z_offset, min(cor[:, 2]))

        except:
            cor, f = ply_parser(fp)

            for i in range(0, len(f)):
                for j in range(0, len(f[i])):
                    f[i][j] = int(f[i][j])
                del f[i][0]

            for face_index in f:
                face_cor = cor[face_index]
                if self.x_offset == None:
                    self.x_offset = min(face_cor[:, 0])
                    self.y_offset = min(face_cor[:, 1])
                    self.z_offset = min(face_cor[:, 2])
                else:
                    self.x_offset = min(self.x_offset, min(face_cor[:, 0]))
                    self.y_offset = min(self.y_offset, min(face_cor[:, 1]))
                    self.z_offset = min(self.z_offset, min(face_cor[:, 2]))


    def load_from_ply(self, fp):
        scene_name = Path(fp).with_suffix('').name

        try:
            plydata = PlyData.read(fp)
            if plydata['vertex'].count == 0:
                return Building()
            cor = np.vstack((plydata['vertex']['x'],
                             plydata['vertex']['y'],
                             plydata['vertex']['z'])).transpose()
            building_model = Building()
            building_model.scene_name = scene_name
            face_name = plydata['face'].data.dtype.names[0]

            for face_index in plydata['face'].data[face_name]:
                face_cor = cor[face_index]
                building_model.add_topsurface(Surface(face_cor))

            return building_model
        except:
            cor, f = ply_parser(fp)

            for i in range(0, len(f)):
                for j in range(0, len(f[i])):
                    f[i][j] = int(f[i][j])
                del f[i][0]

            building_model = Building()
            building_model.scene_name = scene_name

            for face_index in range(len(f)):
                face_cor = cor[f[face_index]]
                building_model.add_topsurface(Surface(face_cor))

            return building_model



    def load_from_curved_ply(self, fp):
        scene_name = Path(fp).with_suffix('').name
        try:
            plydata = PlyData.read(fp)
            if plydata['vertex'].count == 0:
                return Curved_building()

            cor = np.vstack((plydata['vertex']['x'],
                             plydata['vertex']['y'],
                             plydata['vertex']['z'])).transpose()

            building_model = Curved_building()
            building_model.scene_name = scene_name
            face_name = plydata['face'].data.dtype.names[0]
            fi = np.array([face_index for face_index
                           in plydata['face'].data[face_name]])
        except:
            cor, f = ply_parser(fp)

            for i in range(0, len(f)):
                for j in range(0, len(f[i])):
                    f[i][j] = int(f[i][j])
                del f[i][0]

            building_model = Curved_building()
            building_model.scene_name = scene_name
            fi = np.array(f)

        c_cor = []
        c_cor_index = []
        t_fi = fi
        while (len(t_fi) > 0):
            t = t_fi[0]
            del_i = []

            for i in range(0, len(t_fi)):
                if len(list_intersect(t, t_fi[i])) > 0:
                    t = list_union(t, t_fi[i])
                    del_i.append(i)
            t_fi = np.delete(np.array(t_fi), tuple(del_i), 0)
            c_cor.append(t)

            si = []
            for i in range(0, len(fi)):
                if len(list_intersect(t, fi[i])) > 0:
                    si.append(i)
            c_cor_index.append(si)

        pn = 0
        for i in range(0, len(c_cor_index)):
            si = c_cor_index[i]
            surface_index = fi[si]
            triangle_index = []
            unique_index = np.unique(surface_index).tolist()
            for index in surface_index:
                triangle_index.append([unique_index.index(x) + pn + 1
                                       for x in index])
            pn = np.max(triangle_index)
            building_model.add_topsurface(cor[unique_index], np.array(triangle_index))

        return building_model

    def initialize(self, ply_path, dem_path, offset=True):
        self.dem = gdal.Open(dem_path)
        self.ply_path = ply_path
        self.obj_path = ply_path + "_obj"
        self.surface_path = ply_path + "_surface"
        self.offset_flag = offset

        if not os.path.exists(self.obj_path):
            os.makedirs(self.obj_path)
        if not os.path.exists(self.surface_path):
            os.makedirs(self.surface_path)

        file_name = os.listdir(self.ply_path)
        log_file = open(self.obj_path + "_model_log.txt", 'w')
        log_file.write(time.asctime() + '\n######\n')
        start_time = time.time()

        self.building_name = [fp.replace('.ply', '') for fp in file_name]

        for fp in file_name:
            self.get_offset(os.path.join(self.ply_path, fp))

        for fp in file_name:
            process = ''.join(['Now loading the PLY: ' + fp + '\n'])
            sys.stdout.write(process)
            if 'curve' in fp:
                self.buildings.append(self.load_from_curved_ply(os.path.join(self.ply_path, fp)))
            else:
                self.buildings.append(self.load_from_ply(os.path.join(self.ply_path, fp)))
            self.building_num += 1
        sys.stdout.write('Loading PLY finished!         \n')

        for i in range(0, self.building_num):
            process = ''.join(['Now processing intersected surfaces: ' + str(i) + '\r'])
            sys.stdout.write(process)
            self.buildings[i].split_surface()
            sys.stdout.flush()
        sys.stdout.write('Processing intersected surfaces finished!\n')

        for i in range(0, self.building_num):
            process = ''.join(['Now generating bottom surfaces: ' + str(i) + '\r'])
            sys.stdout.write(process)
            self.buildings[i].get_bottomsurface(self.dem)
            self.buildings[i].get_flatsurface()
            sys.stdout.flush()
        sys.stdout.write('Generating bottom surfaces finished!\n')

        generate_model_time = time.time()
        log_file.write('Generate model time:')
        log_file.write(str(generate_model_time - start_time) + 's' + '\n######\n')
        log_file.close()

    def write_model(self, offset=True):
        log_file = open(os.path.join(self.obj_path, "model_log.txt"), 'a+')
        start_time = time.time()
        if not offset:
            model_offset = [0, 0, 0]
        else:
            model_offset = [self.x_offset, self.y_offset, self.z_offset]

        for bi in range(0, self.building_num):
            write_path = os.path.join(self.obj_path, self.building_name[bi] + ".obj")
            out_file = open(write_path, 'w')
            s = self.buildings[bi].get_obj_string(model_offset)
            self.surface_info_str += self.building_name[bi] + '\n'
            for si in range(0, self.buildings[bi].surface_num):
                self.surface_info_str += 'surface #' + str(si) + '\nVertex num: ' +\
                                         str(self.buildings[bi].surface_info[si][0]) + \
                                         '\nEdge num: ' + \
                                         str(self.buildings[bi].surface_info[si][1]) + \
                                         '\nArea: ' + \
                                         str(self.buildings[bi].surface_info[si][2]) + \
                                         '\n'
            out_file.write('#x offset: ' + str(model_offset[0]) + '\n')
            out_file.write('#y offset: ' + str(model_offset[1]) + '\n')
            out_file.write('#z offset: ' + str(model_offset[2]) + '\n')
            out_file.write('#top surface num: ' + str(self.buildings[bi].surface_num) + '\n')
            self.top_num_total += self.buildings[bi].surface_num
            out_file.write('#bottom surface num: ' + str(self.buildings[bi].surface_num) + '\n')
            self.bottom_num_total += self.buildings[bi].surface_num
            out_file.write('#wall surface num: ' + str(self.buildings[bi].wall_num) + '\n')
            self.wall_num_total += self.buildings[bi].wall_num
            out_file.write('#edge num: ' + str(self.buildings[bi].edge_num) + '\n')
            self.edge_num_total += self.buildings[bi].edge_num
            out_file.write('#vertex num: ' + str(self.buildings[bi].vertex_num) + '\n')
            self.vertex_num_total += self.buildings[bi].vertex_num
            out_file.write(''.join(s))
            del s
            out_file.close()
        write_model_time = time.time()
        self.surface_num_total = self.top_num_total + self.bottom_num_total + self.wall_num_total
        log_file.write('Write model time:')
        log_file.write(str(write_model_time - start_time) + 's' + '\n######\n')
        log_file.write('Total top surface num: ')
        log_file.write(str(self.top_num_total) + '\n')
        log_file.write('Total bottom surface num: ')
        log_file.write(str(self.bottom_num_total) + '\n')
        log_file.write('Total wall surface num: ')
        log_file.write(str(self.wall_num_total) + '\n')
        log_file.write('Total surface num: ')
        log_file.write(str(self.surface_num_total) + '\n')
        log_file.write('Total edge num: ')
        log_file.write(str(self.edge_num_total) + '\n')
        log_file.write('Total vertex num: ')
        log_file.write(str(self.vertex_num_total) + '\n######\n')
        log_file.write('Surface info: \n')
        log_file.write(self.surface_info_str + '\n######\n')
        log_file.close()

    def write_surface(self, offset=True):
        if not offset:
            model_offset = [0, 0, 0]
        else:
            model_offset = [self.x_offset, self.y_offset, self.z_offset]
        for bi in range(0, self.building_num):
            write_path = os.path.join(self.surface_path, self.building_name[bi] + ".obj")
            out_file = open(write_path, 'w')
            s = self.buildings[bi].get_top_string(model_offset)
            out_file.write(''.join(s))
            del s
            out_file.close()
