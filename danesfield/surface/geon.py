#!/usr/bin/env python

import os
import re
import sys
import time
import json as js
import numpy as np
from osgeo import gdal
from pathlib import Path
from .scene import Model
from .MinimumBoundingBox import MinimumBoundingBox as mbr
from .geon_functions import (
    add_box_geon,
    add_gable_geon,
    add_mesh_geon,
    add_shed_geon,
)
from .poly_functions import Polygon, list_intersect


class Geon(Model):
    def __init__(self):
        super(Geon, self).__init__()
        self.geon_type = {}
        self.geon_parameter = {}
        self.geon_json = []
        self.error = []

    def load_geon(self, fp, building_index):
        '''
        Load geon info from ply file
        :param fp: Ply file path
        :param building_index: building index
        :return: Geon dictionary
        '''
        tf = open(fp)
        lines = tf.readlines()
        scene_name = Path(fp).with_suffix('').name
        flag = 0
        geon = {}
        gable = []

        for l in lines:
            if 'Planes ID' in l:
                ID_string = re.findall('Planes ID: .*', l)[0].replace('Planes ID: ', '')
                ID = ID_string.split(' ')
                geon_type = 'Flat'
                if 'Gable' in l:
                    geon_type = 'Gable ' + ID_string
                elif 'Flats' in l:
                    geon_type = 'Flat'
                elif 'Shelds' in l:
                    geon_type = 'Shed'

                for geo_ID in ID:
                    if geo_ID != '':
                        if 'Gable' in geon_type:
                            '''
                            Merge Gable ID
                            '''
                            temp_gable = [int(ID[0]), int(ID[1])]
                            gable.append(temp_gable)
                            for j in range(len(gable) - 1):
                                if len(list_intersect(gable[j], temp_gable)) >= 1:
                                    gable[j].append(temp_gable[0])
                                    gable[j].append(temp_gable[1])
                                    del gable[-1]
                            break
                        else:
                            geon[int(geo_ID)] = geon_type
            flag += 1
        for i in range(len(gable)):
            geon[gable[i][0]] = 'Gable ' + ' '.join([str(index) for index in gable[i]])
        self.geon_type[building_index] = dict(scene_name=scene_name, geon=geon)

    def initialize(self, ply_path, dem_path, offset=True):
        '''
        Initialize whole area data
        :param ply_path: Ply files director
        :param dem_path: DEM file path
        :param offset:
        :return:
        '''
        self.dem = gdal.Open(dem_path)
        self.ply_path = ply_path
        self.offset_flag = offset
        self.geonjson_path = ply_path + "_json"

        if not os.path.exists(self.geonjson_path):
            os.makedirs(self.geonjson_path)

        file_name = os.listdir(self.ply_path)
        self.building_name = [fp.replace('.ply', '') for fp in file_name]
        self.building_num = len(file_name)
        for fp in file_name:
            self.get_offset(os.path.join(self.ply_path, fp))

        for i in range(self.building_num):
            fp = file_name[i]
            process = ''.join(['Now loading the PLY: ' + fp + '\n'])
            sys.stdout.write(process)
            self.buildings.append(self.load_from_ply(os.path.join(self.ply_path, fp)))
            self.load_geon(os.path.join(self.ply_path, fp), i)
        sys.stdout.write('Loading PLY finished!           \n')

        for i in range(self.building_num):
            process = ''.join(['Now processing intersected surfaces: ' + str(i) + '\r'])
            sys.stdout.write(process)
            self.buildings[i].get_bottomsurface(self.dem)
            self.buildings[i].get_flatsurface()
        sys.stdout.write('Generating bottom surfaces finished!\n')

    def get_geons(self):
        '''
        Get geons parameters
        :return:
        '''
        model_offset = np.array([self.x_offset, self.y_offset, self.z_offset])
        for i in self.geon_type.keys():
            building_error = {}
            id = 0
            geon_info = self.geon_type[i]
            building = self.buildings[i]
            geon_parameter = []
            for si in geon_info['geon'].keys():
                if geon_info['geon'][si] == 'Flat':
                    top_surf = building.topsurface[si].point_cor
                    bottom_surf = building.bottomsurface[si].point_cor
                    poly_area = Polygon(top_surf[:, 0:2]).area
                    mbr_area = mbr(top_surf[:, 0:2]).area
                    if (mbr_area - poly_area) / mbr_area >= 0.35:
                        geon_rst = add_mesh_geon(id, top_surf, bottom_surf, model_offset)
                        geon_parameter.append(geon_rst[0])
                        building_error['mesh_' + str(id)] = geon_rst[1]
                    else:
                        geon_rst = add_box_geon(id, top_surf, bottom_surf, model_offset)
                        geon_parameter.append(geon_rst[0])
                        building_error['box_' + str(id)] = geon_rst[1]

                elif 'Gable' in geon_info['geon'][si]:
                    surf_ID = geon_info['geon'][si].split(' ')[1:]
                    surf_ID = [int(ID) for ID in surf_ID]
                    top_surfs = [building.topsurface[ID].point_cor for ID in surf_ID]
                    bottom_surfs = [building.bottomsurface[ID] for ID in surf_ID]
                    body_Z = np.mean([pc.point_cor[:, 2].mean() for pc in bottom_surfs])
                    geon_rst = add_gable_geon(id, top_surfs, body_Z, model_offset)
                    geon_parameter.append(geon_rst[0])
                    building_error['gable_' + str(id)] = geon_rst[1]

                elif geon_info['geon'][si] == 'Shed':
                    top_surf = building.topsurface[si].point_cor
                    bottom_surf = building.bottomsurface[si].point_cor
                    poly_area = Polygon(top_surf[:, 0:2]).area
                    mbr_area = mbr(top_surf[:, 0:2]).area
                    if (mbr_area - poly_area) / mbr_area >= 0.35:
                        body_Z = bottom_surf[:, 2].mean()
                        geon_rst = add_shed_geon(id, top_surf, body_Z, model_offset)
                        geon_parameter.append(geon_rst[0])
                        building_error['shed_' + str(id)] = geon_rst[1]
                    else:
                        geon_rst = add_box_geon(id, top_surf, bottom_surf, model_offset)
                        geon_parameter.append(geon_rst[0])
                        building_error['box_' + str(id)] = geon_rst[1]

                id += 1
            self.geon_parameter[i] = geon_parameter
            self.error.append(building_error)

    def geons_to_json(self):
        '''
        Transer geon parameters to json string
        :return:
        '''
        offset = np.array([self.x_offset, self.y_offset, self.z_offset])
        for i in range(len(self.geon_parameter)):
            geon_obj = []
            for geon in self.geon_parameter[i]:
                geon_obj.append(geon)
            coordinate = dict(type='EPSG',
                              parameters=['wgs84', 'UTM zone 16N', offset[0], offset[1], offset[2],
                                          0, 0, 0, 0, 0, ])
            scene = dict(id=self.geon_type[i]['scene_name'],
                         coordinate_system=coordinate, objects=geon_obj)
            json_obj = dict(name='Geon Json', producer_id='Purdue-Zhixin', team='CORE3D',
                            timestamp=time.ctime(), scenes=[scene])
            self.geon_json.append(js.dumps(json_obj, sort_keys=True, indent=2,
                                  separators=(',', ': ')))

    def write_geonjson(self):
        '''
        Write geon json and error assessment result
        :return:
        '''
        area_error = 0
        for bi in range(0, self.building_num):
            write_path = os.path.join(self.geonjson_path, self.building_name[bi] + ".json")
            error_path = os.path.join(self.geonjson_path, self.geon_type[bi]['scene_name'] + ".txt")
            out_file = open(write_path, 'w')
            out_file.write(self.geon_json[bi])
            error_file = open(error_path, 'a+')
            error_file.write('Error for each geon(unit: meter)\n')
            building_error = 0
            for geon_id in self.error[bi].keys():
                error_info = geon_id + ': ' + str(self.error[bi][geon_id]) + '\n'
                error_file.write(error_info)
                building_error += self.error[bi][geon_id]
            building_error = building_error/len(self.error[bi].keys())
            error_file.write('Building error: ' + str(building_error))
            error_file.close()
            area_error += building_error
        area_error = area_error / self.building_num
        area_error_path = os.path.join(self.geonjson_path, "scene_error.txt")
        area_error_file = open(area_error_path, 'w')
        area_error_file.write('Area error(unit: meter): ' + str(area_error))
        area_error_file.close()
