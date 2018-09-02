import numpy as np
import os

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
#import plotly.plotly as py
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import pcl

from scipy.spatial import ConvexHull

def ProjectPoint(model, pt):
    #simplest projection model
    px = int((pt[0]-model['corners'][0])/model['project_model'][1]*model['scale'])
    py = int((pt[1]-model['corners'][1])/model['project_model'][5]*model['scale'])
    return [px,py]

def BackProjectPoint(model, pt):
    #simplest projection model
    px = float(pt[0])/model['scale']*model['project_model'][1] + model['corners'][0]
    py = float(pt[1])/model['scale']*model['project_model'][5] + model['corners'][1]
    return [px,py]

def label_point_shape(model, image, pc):
    label = np.zeros((pc.shape[0]), dtype = np.int32)
    for idx in range(pc.shape[0]):
        projected_point = ProjectPoint(model, [pc[idx,0],pc[idx,1]])
        projected_point[0] = int(projected_point[0]+0.5)
        projected_point[1] = int(projected_point[1]+0.5)
        label[idx] = image[projected_point[1], projected_point[0]]
    return label

def read_txt_pc(filename):
    point_list = []
    with open(filename, 'r') as pc_file:
        print('opened')
        for line in pc_file:
            point_coordinate = line.split(',')
            point_list.append([float(point_coordinate[0]), float(point_coordinate[1]), float(point_coordinate[2])])
    return np.array(point_list)

def read_geon_type_pc(filename):
    point_list = []
    geon_label = []
    building_label = []
    with open(filename, 'r') as pc_file:
        print('opened')
        for line in pc_file:
            point_coordinate = line.split(' ')
            point_list.append([float(point_coordinate[0]), float(point_coordinate[1]), float(point_coordinate[2])])
            building_label.append(int(point_coordinate[3]))
            geon_label.append(int(point_coordinate[4]))
    return np.array(point_list), np.array(building_label), np.array(geon_label)

def write_txt_pc(filename, pc):
    with open(filename, 'w') as pc_file:
        for point in pc:
            pc_file.write('{},{},{}\n'.format(point[0],point[1],point[2]))

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def draw_poly_curve(ax, centroid, ex, ey, fitted_points, coefficients, min_axis_z, max_axis_z, color = 'y'):
    origin = np.array([0, 0, 0])
    ez = np.cross(ex,ey) 
    ortho_x = np.matmul(fitted_points - centroid,ex)
    ortho_x_max = np.max(ortho_x)
    ortho_x_min = np.min(ortho_x)
    
    inverse_matrix = np.zeros((3,3),np.float32)
    inverse_matrix[0,:] = ex 
    inverse_matrix[1,:] = ez 
    inverse_matrix[2,:] = ey 

    ortho_grid_x = np.arange(ortho_x_min, ortho_x_max, 10.0)
    ortho_grid_z = np.arange(min_axis_z, max_axis_z,  10.0)
    ortho_grid_y = coefficients[0]*ortho_grid_x*ortho_grid_x + coefficients[1]*ortho_grid_x + coefficients[2]
    grid_point = np.zeros((ortho_grid_x.shape[0] * ortho_grid_z.shape[0], 3),dtype = np.float32)
    
    for i in range(ortho_grid_x.shape[0]):
        for j in range(ortho_grid_z.shape[0]):
            grid_point[i*ortho_grid_z.shape[0] + j,:] = np.array([ortho_grid_x[i], ortho_grid_z[j], ortho_grid_y[i]])

    original_grid_point = np.matmul(grid_point, inverse_matrix) + centroid
    original_grid_point = np.reshape(original_grid_point,(ortho_grid_x.shape[0],ortho_grid_z.shape[0],3))
    #print(grid_point.shape)
    ax.plot_wireframe(original_grid_point[:,:,0], original_grid_point[:,:,1], original_grid_point[:,:,2], color=color, alpha=0.8)
    return original_grid_point
    #ax.scatter(original_grid_point[:,0], original_grid_point[:,1], original_grid_point[:,2], color='y', alpha=0.1)

def get_poly_hull(centroid, ex, ey, fitted_points, coefficients):
    origin = np.array([0, 0, 0])
    ez = np.cross(ex,ey) 
    
    inverse_matrix = np.zeros((3,3),np.float32)
    inverse_matrix[0,:] = ex 
    inverse_matrix[1,:] = ez 
    inverse_matrix[2,:] = ey 

    forward_matrix = np.zeros((3,3),np.float32)
    forward_matrix[:,0] = ex 
    forward_matrix[:,1] = ey 
    forward_matrix[:,2] = ez 

    projected_points = np.matmul(fitted_points - centroid,forward_matrix)
    projected_points[:,1] = 0
    projected_points_2d = np.zeros((projected_points.shape[0], 2), dtype=np.float32)
    projected_points_2d[:,0] = projected_points[:,0]
    projected_points_2d[:,1] = projected_points[:,2]
    hull = ConvexHull(projected_points_2d)

    hull_points = np.zeros((len(hull.vertices), 3), dtype=np.float32)
    hull_edges = []
    for i in range(len(hull.vertices)):
        hull_points[i,0] = projected_points_2d[hull.vertices[i],0]
        hull_points[i,1] = projected_points_2d[hull.vertices[i],1]
        hull_points[i,2] = coefficients[0]*hull_points[i,0]*hull_points[i,0] + coefficients[1]*hull_points[i,0] + coefficients[2] 

    for simplex in hull.simplices:
        hull_points_1 = [0,0]
        hull_points_1[0] = projected_points_2d[simplex[0],0]
        hull_points_1[1] = projected_points_2d[simplex[0],1]

        hull_points_2 = [0,0]
        hull_points_2[0] = projected_points_2d[simplex[1],0]
        hull_points_2[1] = projected_points_2d[simplex[1],1]
        
        hull_edges.append((hull_points_1,hull_points_2))

    return hull_points, hull_edges

def get_poly_ply_volume_with_hull(dtm, projection_model, centroid, ex, ey, coefficients,\
     hull_points, hull_edges, start_point, center_of_mess):

    origin = np.array([0, 0, 0])
    ez = np.cross(ex,ey) 
    ortho_x_max, max_axis_z, _  = np.max(hull_points, axis = 0)
    ortho_x_min, min_axis_z, _  = np.min(hull_points, axis = 0)

    inverse_matrix = np.zeros((3,3),np.float32)
    inverse_matrix[0,:] = ex 
    inverse_matrix[1,:] = ez 
    inverse_matrix[2,:] = ey 

    ortho_grid_x = np.linspace(ortho_x_min+0.001, ortho_x_max-0.001, 30)
    ortho_grid_y = coefficients[0]*ortho_grid_x*ortho_grid_x + coefficients[1]*ortho_grid_x + coefficients[2]
    
    vertex = []
    face = []
    grid_point = np.zeros((ortho_grid_x.shape[0] * 2, 3),dtype = np.float32)
    
    #for edge in hull_edges:
    #    print(edge)
    for i in range(ortho_grid_x.shape[0]):
        valid_z = []
        for edge in hull_edges:
            if (edge[0][0]<=ortho_grid_x[i] and edge[1][0]>ortho_grid_x[i]):
                otho_z = edge[0][1] + (ortho_grid_x[i] - edge[0][0])/(edge[1][0] - edge[0][0])*(edge[1][1] - edge[0][1])
                valid_z.append(otho_z)
            if (edge[0][0]>ortho_grid_x[i] and edge[1][0]<=ortho_grid_x[i]):
                otho_z = edge[1][1] + (ortho_grid_x[i] - edge[1][0])/(edge[0][0] - edge[1][0])*(edge[0][1] - edge[1][1])
                valid_z.append(otho_z)
        if len(valid_z)==1:
            valid_z.append(valid_z[0])
        if len(valid_z)==2:
            grid_point[i*2,:] = np.array([ortho_grid_x[i], min(valid_z[0],valid_z[1]), ortho_grid_y[i]])
            grid_point[i*2 + 1,:] = np.array([ortho_grid_x[i], max(valid_z[1],valid_z[0]), ortho_grid_y[i]])
    #print(grid_point) 
    original_grid_point = np.matmul(grid_point, inverse_matrix) + centroid 

    original_grid_point = np.reshape(original_grid_point,(ortho_grid_x.shape[0],2,3))

    for i in range(ortho_grid_x.shape[0]):
        for j in range(2):
            vertex.append((original_grid_point[i,j,0], original_grid_point[i,j,1], original_grid_point[i,j,2]))
            image_point = ProjectPoint(projection_model, \
                    [original_grid_point[i,j,0] + center_of_mess[0],\
                    original_grid_point[i,j,1] + center_of_mess[1]])
            height = dtm[image_point[1],image_point[0]] - center_of_mess[2]
            vertex.append((original_grid_point[i,j,0], original_grid_point[i,j,1], height))
        if i != 0:
            face.append(([start_point+4*i-4, start_point+4*i-2 , start_point+4*i], 255, 255, 255)) 
            face.append(([start_point+4*i, start_point+4*i+2, start_point+4*i-2], 255, 255, 255)) 
            face.append(([start_point+4*i-3, start_point+4*i-1 , start_point+4*i+3], 255, 255, 255)) 
            face.append(([start_point+4*i+3, start_point+4*i+1, start_point+4*i-3], 255, 255, 255)) 

            face.append(([start_point+4*i-4, start_point+4*i-3, start_point+4*i], 255, 255, 255)) 
            face.append(([start_point+4*i, start_point+4*i+1, start_point+4*i-3], 255, 255, 255)) 
            face.append(([start_point+4*i-2, start_point+4*i-1, start_point+4*i+2], 255, 255, 255)) 
            face.append(([start_point+4*i+2, start_point+4*i+3, start_point+4*i-1], 255, 255, 255)) 

    face.append(([start_point+0, start_point+1 , start_point+3], 255, 255, 255))
    face.append(([start_point+3, start_point+2 , start_point+0], 255, 255, 255))

    final = len(vertex)-1

    face.append(([start_point+final-3, start_point+final-2 , start_point+final], 255, 255, 255))
    face.append(([start_point+final, start_point+final-1 , start_point+final-3], 255, 255, 255))

    return vertex, face

def get_poly_ply_with_hull(centroid, ex, ey, coefficients, hull_points, hull_edges, start_point):
    origin = np.array([0, 0, 0])
    ez = np.cross(ex,ey)
    ortho_x_max, max_axis_z, _  = np.max(hull_points, axis = 0)
    ortho_x_min, min_axis_z, _  = np.min(hull_points, axis = 0)

    inverse_matrix = np.zeros((3,3),np.float32)
    inverse_matrix[0,:] = ex 
    inverse_matrix[1,:] = ez 
    inverse_matrix[2,:] = ey 

    ortho_grid_x = np.linspace(ortho_x_min, ortho_x_max, 30)
    ortho_grid_y = coefficients[0]*ortho_grid_x*ortho_grid_x + coefficients[1]*ortho_grid_x + coefficients[2]
    
    vertex = []
    face = []
    grid_point = np.zeros((ortho_grid_x.shape[0] * 2, 3),dtype = np.float32)#valid_z

    for i in range(ortho_grid_x.shape[0]):
        valid_z = []
        for edge in hull_edges:
            if (edge[0][0]<=ortho_grid_x[i] and edge[1][0]>ortho_grid_x[i]):
                otho_z = edge[0][1] + (ortho_grid_x[i] - edge[0][0])/(edge[1][0] - edge[0][0])*(edge[1][1] - edge[0][1])
                valid_z.append(otho_z)
            if (edge[0][0]>ortho_grid_x[i] and edge[1][0]<=ortho_grid_x[i]):
                otho_z = edge[1][1] + (ortho_grid_x[i] - edge[1][0])/(edge[0][0] - edge[1][0])*(edge[0][1] - edge[1][1])
                valid_z.append(otho_z)
        if len(valid_z)==1:
            valid_z.append(valid_z[0])
        if len(valid_z)==2:
            grid_point[i*2,:] = np.array([ortho_grid_x[i], min(valid_z[0],valid_z[1]), ortho_grid_y[i]])
            grid_point[i*2 + 1,:] = np.array([ortho_grid_x[i], max(valid_z[1],valid_z[0]), ortho_grid_y[i]])
    
    original_grid_point = np.matmul(grid_point, inverse_matrix) + centroid 
    original_grid_point = np.reshape(original_grid_point,(ortho_grid_x.shape[0],2,3))

    for i in range(ortho_grid_x.shape[0]):
        for j in range(2):
            vertex.append((original_grid_point[i,j,0], original_grid_point[i,j,1], original_grid_point[i,j,2]))
        if i != 0:
            face.append(([start_point+(i-1)*2, start_point+2*i-1 , start_point+2*i], 255, 255, 255)) 
            face.append(([start_point+2*i+1, start_point+2*i-1, start_point+2*i], 255, 255, 255)) 
   
    print(grid_point.shape)
    grid_points_show = np.matmul(grid_point, inverse_matrix) + centroid
    print(grid_points_show.shape)

    return vertex, face, grid_points_show

def get_poly_ply(centroid, ex, ey, fitted_points, coefficients, min_axis_z, max_axis_z, start_point):
    origin = np.array([0, 0, 0])
    ez = np.cross(ex,ey) 
    ortho_x = np.matmul(fitted_points - centroid,ex)
    ortho_x_max = np.max(ortho_x)
    ortho_x_min = np.min(ortho_x)
    
    inverse_matrix = np.zeros((3,3),np.float32)
    inverse_matrix[0,:] = ex 
    inverse_matrix[1,:] = ez 
    inverse_matrix[2,:] = ey 

    ortho_grid_x = np.linspace(ortho_x_min, ortho_x_max, 30)
    ortho_grid_z = np.array([min_axis_z, max_axis_z])
    ortho_grid_y = coefficients[0]*ortho_grid_x*ortho_grid_x + coefficients[1]*ortho_grid_x + coefficients[2]
    
    vertex = []
    face = []#setting an array element with a sequence
    grid_point = np.zeros((ortho_grid_x.shape[0] * ortho_grid_z.shape[0], 3),dtype = np.float32)

    for i in range(ortho_grid_x.shape[0]):
        for j in range(ortho_grid_z.shape[0]):
            grid_point[i*ortho_grid_z.shape[0] + j,:] = np.array([ortho_grid_x[i], ortho_grid_z[j], ortho_grid_y[i]])
    
    original_grid_point = np.matmul(grid_point, inverse_matrix) + centroid 

    original_grid_point = np.reshape(original_grid_point,(ortho_grid_x.shape[0],ortho_grid_z.shape[0],3))

    for i in range(ortho_grid_x.shape[0]):
        for j in range(ortho_grid_z.shape[0]):
            vertex.append((original_grid_point[i,j,0], original_grid_point[i,j,1], original_grid_point[i,j,2]))
        if i != 0:
            face.append(([start_point+(i-1)*2, start_point+2*i-1 , start_point+2*i], 255, 255, 255)) 
            face.append(([start_point+2*i+1, start_point+2*i-1, start_point+2*i], 255, 255, 255)) 

    return vertex, face, ortho_x_min, ortho_x_max

def get_poly_ply_volume(dtm, projection_model, centroid, ex, ey, fitted_points, coefficients,\
        min_axis_z, max_axis_z, start_point, center_of_mess):

    origin = np.array([0, 0, 0])
    ez = np.cross(ex,ey) 
    ortho_x = np.matmul(fitted_points - centroid,ex)
    ortho_x_max = np.max(ortho_x)
    ortho_x_min = np.min(ortho_x)
    
    inverse_matrix = np.zeros((3,3),np.float32)
    inverse_matrix[0,:] = ex 
    inverse_matrix[1,:] = ez 
    inverse_matrix[2,:] = ey 

    ortho_grid_x = np.linspace(ortho_x_min, ortho_x_max, 30)
    ortho_grid_z = np.array([min_axis_z, max_axis_z])
    ortho_grid_y = coefficients[0]*ortho_grid_x*ortho_grid_x + coefficients[1]*ortho_grid_x + coefficients[2]
    
    vertex = []
    face = []#setting an array element with a sequence
    grid_point = np.zeros((ortho_grid_x.shape[0] * ortho_grid_z.shape[0], 3),dtype = np.float32)

    for i in range(ortho_grid_x.shape[0]):
        for j in range(ortho_grid_z.shape[0]):
            grid_point[i*ortho_grid_z.shape[0] + j,:] = np.array([ortho_grid_x[i], ortho_grid_z[j], ortho_grid_y[i]])
    
    original_grid_point = np.matmul(grid_point, inverse_matrix) + centroid 

    boundary_points = [] 
    boundary_points.append((original_grid_point[0]+center_of_mess).tolist())
    boundary_points.append((original_grid_point[ortho_grid_z.shape[0]-1]+center_of_mess).tolist())
    boundary_points.append((original_grid_point[-1]+center_of_mess).tolist())
    boundary_points.append((original_grid_point[-ortho_grid_z.shape[0]]+center_of_mess).tolist())

    original_grid_point = np.reshape(original_grid_point,(ortho_grid_x.shape[0],ortho_grid_z.shape[0],3))

    for i in range(ortho_grid_x.shape[0]):
        for j in range(ortho_grid_z.shape[0]):
            vertex.append((original_grid_point[i,j,0], original_grid_point[i,j,1], original_grid_point[i,j,2]))
            image_point = ProjectPoint(projection_model, \
                    [original_grid_point[i,j,0] + center_of_mess[0],\
                    original_grid_point[i,j,1] + center_of_mess[1]])
            height = dtm[image_point[1],image_point[0]] - center_of_mess[2]
            vertex.append((original_grid_point[i,j,0], original_grid_point[i,j,1], height))
        if i != 0:
            face.append(([start_point+4*i-4, start_point+4*i-2 , start_point+4*i], 255, 255, 255)) 
            face.append(([start_point+4*i, start_point+4*i+2, start_point+4*i-2], 255, 255, 255)) 
            face.append(([start_point+4*i-3, start_point+4*i-1 , start_point+4*i+3], 255, 255, 255)) 
            face.append(([start_point+4*i+3, start_point+4*i+1, start_point+4*i-3], 255, 255, 255)) 

            face.append(([start_point+4*i-4, start_point+4*i-3, start_point+4*i], 255, 255, 255)) 
            face.append(([start_point+4*i, start_point+4*i+1, start_point+4*i-3], 255, 255, 255)) 
            face.append(([start_point+4*i-2, start_point+4*i-1, start_point+4*i+2], 255, 255, 255)) 
            face.append(([start_point+4*i+2, start_point+4*i+3, start_point+4*i-1], 255, 255, 255)) 

    face.append(([start_point+0, start_point+1 , start_point+3], 255, 255, 255))
    face.append(([start_point+3, start_point+2 , start_point+0], 255, 255, 255))

    final = len(vertex)-1

    face.append(([start_point+final-3, start_point+final-2 , start_point+final], 255, 255, 255))
    face.append(([start_point+final, start_point+final-1 , start_point+final-3], 255, 255, 255))

    return vertex, face, ortho_x_min, ortho_x_max, boundary_points

def check_poly_point(points, centroid, ex, ey, coefficients,\
        min_axis_z, max_axis_z, ortho_x_min, ortho_x_max,  \
        boundary_points):
    
    label = np.zeros((points.shape[0],),dtype = np.int32)
    origin = np.array([0, 0, 0])
    ez = np.cross(ex,ey) 

    ortho_x = np.matmul(points-centroid,ex)
    ortho_z = np.matmul(points-centroid,ez)

    flag_x = np.logical_and(ortho_x>ortho_x_min, ortho_x<ortho_x_max)
    flag_z = np.logical_and(ortho_z>min_axis_z, ortho_z<max_axis_z)
    flag = np.logical_and(flag_x, flag_z)
    print(np.sum(flag))
    if np.sum(flag) == 0:
        return label
    left_x = ortho_x[flag]
    left_z = points[flag,2] - centroid[2]
    y = coefficients[0]*left_x*left_x + coefficients[1]*left_x + coefficients[2]
    label[flag] = np.abs(y-left_z)<1.5
    return label

def get_poly_dsm(dsm, centroid, ex, ey, coefficients,\
        min_axis_z, max_axis_z, ortho_x_min, ortho_x_max,  \
        projection_model, boundary_points):
    origin = np.array([0, 0, 0])
    ez = np.cross(ex,ey) 
    
    inverse_matrix = np.zeros((3,3),np.float32)
    inverse_matrix[0,:] = ex 
    inverse_matrix[1,:] = ez 
    inverse_matrix[2,:] = ey

    #print(ortho_x_min, 0, min_axis_z)
    #print(ortho_x_max, 0, max_axis_z)
    #boundary_set = np.array([[ortho_x_min, min_axis_z, 0], \
    #        [ortho_x_max, max_axis_z, 0],\
    #        [ortho_x_max, min_axis_z, 0],\
    #        [ortho_x_min, max_axis_z, 0]])

    #new_boundary_set = np.matmul(boundary_set,inverse_matrix)
    #new_boundary_set[:,0] =  new_boundary_set[:,0] + centroid[0]
    #new_boundary_set[:,1] =  new_boundary_set[:,1] + centroid[1]
    #new_boundary_set[:,2] =  new_boundary_set[:,2] + centroid[2]
    world_min_x, world_min_y, world_min_z = np.amin(boundary_points, axis = 0)
    world_max_x, world_max_y, world_max_z = np.amax(boundary_points, axis = 0)
    image_points = []
    print(boundary_points)
    for i in range(len(boundary_points)):
        image_points.append(ProjectPoint(projection_model, boundary_points[i]))
    image_points = np.array(image_points)
    image_min_x, image_min_y = np.amin(image_points,axis = 0)
    image_max_x, image_max_y = np.amax(image_points,axis = 0)

    print([world_min_x,world_min_y])
    print([world_max_x,world_max_y])
    print([world_min_z,world_min_z])
    print([image_min_x,image_min_y])
    print([image_max_x,image_max_y])

    for i in range(image_min_x, image_max_x):
        for j in range(image_min_y, image_max_y):
            world_point_x, world_point_y = BackProjectPoint(projection_model, [i,j])
            ortho_x = np.matmul(np.array([world_point_x - centroid[0],\
                    world_point_y - centroid[1], world_min_z - centroid[2]]),ex)
            ortho_z = np.matmul(np.array([world_point_x - centroid[0],\
                    world_point_y - centroid[1], world_min_z - centroid[2]]),ez)
            if i == image_min_x and j == image_min_y:
                print(world_point_x, world_point_y)
                print(ortho_x, ortho_z)
            if ortho_x<ortho_x_min or ortho_x>ortho_x_max:
                continue
            if ortho_z<min_axis_z or ortho_z>max_axis_z:
                continue
            ortho_y = coefficients[0]*ortho_x*ortho_x + coefficients[1]*ortho_x + coefficients[2]
            dsm[j,i] = ortho_y + centroid[2]

    return dsm

def get_cylinder_ply(ax, fitted_points, coefficients, start_point):

    origin = np.array(coefficients[0:3])
    n =  np.array(coefficients[3:6])
    r = coefficients[6]
    
    new_points = fitted_points - origin
    ortho_z = np.matmul(fitted_points - origin,n)
    max_z = np.max(ortho_z)
    min_z = np.min(ortho_z)

    p1 = origin + n*max_z 
    p0 = origin + n*min_z 

    #axis and radius
    v = p1 - p0
    #find magnitude of vector
    mag = np.sqrt(np.sum(v*v)) 
    #unit vector in direction of axis
    v = v / mag
    #make some vector not in the same direction as v

    not_v = np.array([1, 0, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    #make vector perpendicular to v
    n1 = np.cross(v, not_v)
    #normalize n1
    n1 /= np.sqrt(np.sum(n1*n1))
    #make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    #surface ranges over t from 0 to length of axis and 0 to 2*pi
    old_t = np.array([0, mag])
    old_theta = np.linspace(0, 2 * np.pi, 10)
    t, theta = np.meshgrid(old_t, old_theta)

    #generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + r * np.sin(theta) * n1[i] + r * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    
    vertex = []
    face = []
    print(X.shape)
    print(Z.shape)
    print(old_t.shape)
    print(old_theta.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            vertex.append((X[i,j], Y[i,j], Z[i,j]))
        if i != 0:
            face.append(([start_point+(i-1)*2, start_point+2*i-1 , start_point+2*i], 255, 255, 255)) 
            face.append(([start_point+2*i+1, start_point+2*i-1, start_point+2*i], 255, 255, 255))
    return vertex, face

def draw_cylinder(ax, fitted_points, coefficients):

    origin = np.array(coefficients[0:3])
    n =  np.array(coefficients[3:6])
    r = coefficients[6]
    
    new_points = fitted_points - origin
    ortho_z = np.matmul(fitted_points - origin,n)
    max_z = np.max(ortho_z)
    min_z = np.min(ortho_z)

    print(origin) 
    print(n) 
    print(max_z) 

    p1 = origin + n*max_z 
    p0 = origin + n*min_z 

    #axis and radius
    v = p1 - p0
    #find magnitude of vector
    mag = np.sqrt(np.sum(v*v)) 
    #unit vector in direction of axis
    v = v / mag
    #make some vector not in the same direction as v

    not_v = np.array([1, 0, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    #make vector perpendicular to v
    n1 = np.cross(v, not_v)
    #normalize n1
    n1 /= np.sqrt(np.sum(n1*n1))
    #make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    #surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(0, mag, 10)
    theta = np.linspace(0, 2 * np.pi, 10)
    t, theta = np.meshgrid(t, theta)
    #generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + r * np.sin(theta) * n1[i] + r * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    ax.plot_wireframe(X, Y, Z, color='y', alpha=0.8)

def draw_hull(ax, hull_points):
    x = hull_points[:,0].tolist()
    print(len(x))
    y = hull_points[:,1].tolist()
    z = hull_points[:,2].tolist()
    verts = [list(zip(x, y, z))]
    ax.add_collection3d(Poly3DCollection(verts), zs='z')
