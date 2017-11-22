import gdal, ogr, os, osr
import numpy as np

def ProjectPoint(model, pt):
    px = int((pt[0]-model['corners'][0])/model['project_model'][1]*model['scale'])
    py = int((pt[1]-model['corners'][1])/model['project_model'][5]*model['scale'])
    return [px,py]

def GetBuildingCluster(building_list, model, cluster_thres = 200):
    building_cluster_list = []
    for building in building_list:
        if len(building_cluster_list) == 0:
            building_cluster_list.append([building])
        else:
            max_index = -1
            max_similarity = 0
            for cluster_idx, building_cluster in enumerate(building_cluster_list):
                for exist_building in building_cluster:
                    similarity = BuildingSimilarity(building, exist_building, model, cluster_thres)
                    if similarity > max_similarity:
                        max_index = cluster_idx
                        max_similarity = similarity
            if max_index>=0 and max_similarity>0.1:
                building_cluster_list[max_index].append(building)
            else:
                building_cluster_list.append([building])

    return building_cluster_list

def BuildingSimilarity(building_1, building_2, model, cluster_thres = 200):
    geom_1 = building_1.GetGeometryRef()
    g_1 = geom_1.GetGeometryRef(0)
    first_polygon = g_1.GetGeometryRef(0)
    pt_1 = first_polygon.GetPoint(0)

    geom_2 = building_2.GetGeometryRef()
    g_2 = geom_2.GetGeometryRef(0)
    first_polygon = g_2.GetGeometryRef(0)
    pt_2 = first_polygon.GetPoint(0)
    
    similarity = np.sqrt(float((pt_1[0]-pt_2[0])*(pt_1[0]-pt_2[0])) + \
            float((pt_1[1]-pt_2[1])*(pt_1[1]-pt_2[1])))\
            /model['project_model'][1]*model['scale']/cluster_thres
    return max(1-similarity,0)
