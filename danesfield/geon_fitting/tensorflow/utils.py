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

def get_poly_ply_volume(dtm, projection_model, centroid, ex, ey, coefficients,\
        min_axis_z, max_axis_z, ortho_x_min, ortho_x_max, start_point, center_of_mess):

    origin = np.array([0, 0, 0])
    ez = np.cross(ex,ey) 
    
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
            image_point = ProjectPoint(projection_model, \
                    [original_grid_point[i,j,0] + center_of_mess[0],\
                    original_grid_point[i,j,1] + center_of_mess[1]])
            height = dtm[image_point[1],image_point[0]] - center_of_mess[2]
            vertex.append((original_grid_point[i,j,0], original_grid_point[i,j,1], height))
        if i != 0:
            face.append(([start_point+4*i, start_point+4*i-4, start_point+4*i-2], 255, 255, 255)) 
            face.append(([start_point+4*i+2, start_point+4*i, start_point+4*i-2], 255, 255, 255)) 

            face.append(([start_point+4*i-1, start_point+4*i-3, start_point+4*i+3], 255, 255, 255)) 
            face.append(([start_point+4*i+1, start_point+4*i+3, start_point+4*i-3], 255, 255, 255)) 

            face.append(([start_point+4*i-3, start_point+4*i-4, start_point+4*i], 255, 255, 255)) 
            face.append(([start_point+4*i-3, start_point+4*i, start_point+4*i+1], 255, 255, 255)) 
            face.append(([start_point+4*i+2, start_point+4*i-2, start_point+4*i-1], 255, 255, 255)) 
            face.append(([start_point+4*i+3, start_point+4*i+2, start_point+4*i-1], 255, 255, 255)) 

    face.append(([start_point+3, start_point+0, start_point+1], 255, 255, 255))
    face.append(([start_point+0, start_point+3, start_point+2], 255, 255, 255))

    final = len(vertex)-1

    face.append(([start_point+final-3, start_point+final, start_point+final-2], 255, 255, 255))
    face.append(([start_point+final, start_point+final-3, start_point+final-1], 255, 255, 255))

    return vertex, face#, ortho_x_min, ortho_x_max, boundary_points

def get_sphere_volume(dtm, projection_model, centroid, r, \
        theta_min, theta_max, start_point, center_of_mess):

    origin = np.array([0, 0, 0])

    u, v = np.mgrid[0:2*np.pi:20j, theta_min:theta_max:10j]
    x=np.cos(u)*np.sin(v)*r + centroid[0]
    y=np.sin(u)*np.sin(v)*r + centroid[1]
    z=np.cos(v)*r + centroid[2]
    vertex = []
    face = []

    for i in range(z.shape[0]):
        vertex.append((x[i,0], y[i,0], z[i,0]))

    for j in range(1, z.shape[1]):
        for i in range(z.shape[0]):
            vertex.append((x[i,j], y[i,j], z[i,j]))
            if i>0:
                face.append(([start_point+(j-1)*z.shape[0]+i-1, start_point+j*z.shape[0]+i-1, start_point+j*z.shape[0]+i], 255, 255, 255)) 
                face.append(([start_point+j*z.shape[0]+i, start_point+(j-1)*z.shape[0]+i, start_point+(j-1)*z.shape[0]+i-1], 255, 255, 255)) 
    
    if theta_max> -0.9*np.pi:
        j = z.shape[1]
        for i in range(z.shape[0]):
            image_point = ProjectPoint(projection_model, \
                    [x[i,j-1] + center_of_mess[0],\
                    y[i,j-1] + center_of_mess[1]])
            height = dtm[image_point[1],image_point[0]] - center_of_mess[2]
            vertex.append((x[i,j-1], y[i,j-1], height))
            if i>0:
                face.append(([start_point+(j-1)*z.shape[0]+i-1, start_point+j*z.shape[0]+i-1, start_point+j*z.shape[0]+i], 255, 255, 255)) 
                face.append(([start_point+j*z.shape[0]+i, start_point+(j-1)*z.shape[0]+i, start_point+(j-1)*z.shape[0]+i-1], 255, 255, 255))

        image_point = ProjectPoint(projection_model, \
                 [centroid[0] + center_of_mess[0],\
                 centroid[1] + center_of_mess[1]])
        height = dtm[image_point[1],image_point[0]] - center_of_mess[2]
        vertex.append((centroid[0], centroid[1], height))
        final_index = len(vertex)-1
        j = z.shape[1]
        for i in range(z.shape[0]):
            face.append(([start_point+j*z.shape[0]+(i+1)%z.shape[0], 
                start_point+j*z.shape[0]+i, 
                start_point+final_index], 255, 255, 255))

    return vertex, face

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

