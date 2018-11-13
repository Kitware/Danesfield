###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

from . import ellipses as el
import numpy as np
from scipy.optimize import fmin_cobyla

'''
args: n is principal axis direction,
      points are all 3d points to be fitted
      fit_type can have poly2 or ellipse
'''


def fit_2D_curve(n, points, fit_type='poly2', dist_threshold=0.05):
    centroid = get_centroid(points)
    # points_2d has been move to centroid and e1, e2 as axes
    points_2d, e1, e2 = project2plane(points, centroid, n)
    ex = e1
    ey = e2
    ez = np.cross(ex, ey)
    fitted_indices, coefficients, mean_diff = fit2Dshapes(
        points_2d, fit_type=fit_type,  dist_threshold=dist_threshold)
    if fit_type == 'ellipse':
        e1, e2, coefficients = transform_ellipse(e1, e2, coefficients, ez)
    points_z = get_z_along_axis(points[fitted_indices], centroid, ez)
    fitted_indices = np.asarray(fitted_indices)
    min_lst, max_lst, fitted_indices_lst = get_z_length(
        points_z, fitted_indices)
    return centroid, ex, ey, ez, fitted_indices_lst, coefficients, min_lst, max_lst, mean_diff


def get_z_length(points_z, fitted_indices):
    hist, bin_edges = np.histogram(points_z, bins=range(
        int(np.floor(min(points_z))), int(np.ceil(max(points_z))), 5))
    points_z_survivor_indices = np.asarray([i for i in range(len(points_z))
                                            if survive(points_z[i], hist, bin_edges, int(np.floor(min(points_z))), 5, cut_threshold_ratio=0.1)])
    indices_lst = interval_cluster_1d(points_z[points_z_survivor_indices], 10)
    min_lst = []
    max_lst = []
    fitted_indices_lst = []

    for i in range(len(indices_lst)):
        indices = indices_lst[i]
        min_lst.append(min(points_z[points_z_survivor_indices[indices]]))
        max_lst.append(max(points_z[points_z_survivor_indices[indices]]))
        fitted_indices_lst.append(
            fitted_indices[points_z_survivor_indices[indices]])
    return min_lst, max_lst, fitted_indices_lst


def check_2D_curve(ex, ey, ez, coefficients, centroid, points, min_axis_z, max_axis_z, fit_type='poly2', dist_threshold=0.05):
    #centroid = get_centroid(points)
    # points_2d has been move to centroid and e1, e2 as axes
    projection_matrix = np.zeros((3, 3), dtype=np.float)
    projection_matrix[:, 0] = ex
    projection_matrix[:, 1] = ey
    projection_matrix[:, 2] = ez

    new_points = np.matmul(points - centroid, projection_matrix)

    fitted_indices, error = check2Dshapes(
        new_points[:, 0:2], coefficients, fit_type=fit_type,  dist_threshold=dist_threshold)

    z_flag = np.logical_and(
        new_points[:, 2] > min_axis_z, new_points[:, 2] < max_axis_z)

    fitted_indices = np.arange(new_points.shape[0])

    final_flag = np.logical_and(z_flag, error < 3)

    fitted_indices = fitted_indices[final_flag]
    x_val = new_points[fitted_indices, 0]

    ortho_x_max = np.max(x_val)
    ortho_x_min = np.min(x_val)
    bin_num = 50
    hist, bin_edges = np.histogram(x_val, bins = bin_num)
    max_val = np.max(hist)

    min_index = 0
    for i in range(hist.shape[0]):
        if hist[i]>0.2*max_val:
            min_index = i
            break

    max_index = hist.shape[0]
    for i in range(hist.shape[0]-1, 0, -1):
        if hist[i]>0.2*max_val:
            max_index = i
            break

    real_x_max = ortho_x_min+ (ortho_x_max-ortho_x_min)/bin_num*max_index
    real_x_min = ortho_x_min+ (ortho_x_max-ortho_x_min)/bin_num*min_index

    return fitted_indices, real_x_max, real_x_min, error


def transform_ellipse(e1, e2, coefficients, n):
    nx, ny, nz = n
    ellipse_center, ellipse_width, ellipse_height, ellipse_phi = coefficients
    cos_phi = np.cos(ellipse_phi)
    sin_phi = np.sin(ellipse_phi)
    R_matrix = [[cos_phi + nx**2*(1 - cos_phi), nx * ny * (1 - cos_phi) - nz * sin_phi, nx * nz * (1 - cos_phi) + ny * sin_phi],
                [nx * ny * (1 - cos_phi) + nz * sin_phi, cos_phi + ny **
                 2*(1 - cos_phi), ny * nz * (1 - cos_phi) - nx * sin_phi],
                [nx * nz * (1 - cos_phi) - ny * sin_phi, ny * nz * (1 - cos_phi) + nx * sin_phi, cos_phi + nz**2*(1 - cos_phi)]]
    e1_transformed = np.matmul(R_matrix, e1)
    e2_transformed = np.matmul(R_matrix, e2)
    R_counter = [[cos_phi, sin_phi],
                 [-1.0 * sin_phi, cos_phi]]
    transformed_x, transformed_y = np.matmul(R_counter, ellipse_center)
    return e1_transformed, e2_transformed, [[transformed_x, transformed_y], ellipse_width, ellipse_height]


'''
    create histogram of points z value along principal axis,
    cut off the bin which smaller than cut_threshold_ratio of the largest bin
'''


def survive(point_z, hist, edges, start, bin_size, cut_threshold_ratio=0.1):
    bin_num = min(int(np.floor(point_z - start)) // bin_size, len(hist) - 1)
    cut_threshold = cut_threshold_ratio * max(hist)
    return hist[bin_num] > cut_threshold


def interval_cluster_1d(points_1d, max_interval):
    sorted_index = np.argsort(points_1d)
    indices_list = []
    start = 0
    for i in range(1, sorted_index.shape[0]):
        if abs(points_1d[sorted_index[i]] - points_1d[sorted_index[i-1]]) > max_interval:
            indices_list.append(sorted_index[start:i])
            start = i
    indices_list.append(sorted_index[start:len(points_1d)])
    return indices_list


def get_z_along_axis(points, centroid, n):
    return np.dot(points - centroid, n)


def get_centroid(points):
    return np.mean(points, axis=0)


def project2plane(points_3d, centroid, n, x=None):
    if x is None:
        e2x = 1.0
        e2y = 0.0
        e2z = (-1.0 * e2x * n[0]) / n[2]
        e2 = np.asarray([e2x, e2y, e2z], dtype=np.float32)
        e2 = e2 / np.linalg.norm(e2)

        e1x = 1.0
        e1z = -1.0 / e2z
        e1y = -1.0 * (n[0] + e1z * n[2]) / n[1]
        e1 = np.asarray([e1x, e1y, e1z], dtype=np.float32)
        e1 = e1 / np.linalg.norm(e1)
    else:
        e2 = x
        e2x, e2y, e2z = e2

        e1x = 1.0
        e1y = (e2z * n[0] - e2x * n[2]) / (n[2] * e2y - e2z * n[1])
        e1z = -1.0 * (e1y * n[1] + n[0]) / n[2]
        e1 = np.asarray([e1x, e1y, e1z], dtype=np.float32)
        e1 = e1 / np.linalg.norm(e1)
    # print e1, e2, n, np.dot(e1,e2), np.dot(e1,n), np.dot(e2,n)
    assert (np.dot(e1, e2) < 1e-3 and np.dot(e1, n) < 1e-3 and np.dot(e2, n) < 1e-3), \
        "e1,e2 and n not orthonormal!"
    t_1 = np.matmul(points_3d - centroid, e1)
    t_2 = np.matmul(points_3d - centroid, e2)
    points_2d = np.concatenate(
        [np.expand_dims(t_1, axis=1), np.expand_dims(t_2, axis=1)], axis=1)
    # print points_2d.shape
    # points_2d has been move to centroid and e1, e2 as axes
    return points_2d, e1, e2


def project2plane(points_3d, centroid, n, x=None):
    if x is None:
        e2x = 1.0
        e2y = 0.0
        e2z = (-1.0 * e2x * n[0]) / n[2]
        e2 = np.asarray([e2x, e2y, e2z], dtype=np.float32)
        e2 = e2 / np.linalg.norm(e2)

        e1x = 1.0
        e1z = -1.0 / e2z
        e1y = -1.0 * (n[0] + e1z * n[2]) / n[1]
        e1 = np.asarray([e1x, e1y, e1z], dtype=np.float32)
        e1 = e1 / np.linalg.norm(e1)
    else:
        e2 = x
        e2x, e2y, e2z = e2

        e1x = 1.0
        e1y = (e2z * n[0] - e2x * n[2]) / (n[2] * e2y - e2z * n[1])
        e1z = -1.0 * (e1y * n[1] + n[0]) / n[2]
        e1 = np.asarray([e1x, e1y, e1z], dtype=np.float32)
        e1 = e1 / np.linalg.norm(e1)
    # print e1, e2, n, np.dot(e1,e2), np.dot(e1,n), np.dot(e2,n)
    assert (np.dot(e1, e2) < 1e-3 and np.dot(e1, n) < 1e-3 and np.dot(e2, n) < 1e-3), \
        "e1,e2 and n not orthonormal!"
    t_1 = np.matmul(points_3d - centroid, e1)
    t_2 = np.matmul(points_3d - centroid, e2)
    points_2d = np.concatenate(
        [np.expand_dims(t_1, axis=1), np.expand_dims(t_2, axis=1)], axis=1)
    # print points_2d.shape
    # points_2d has been move to centroid and e1, e2 as axes
    return points_2d, e1, e2


'''
 fit_type include poly2 and ellipse
 ellipse return: ellipse_center, ellipse_width, ellipse_height, ellipse_phi
 poly2 return: square coefficient, linear coefficient and constant.
'''


def fit2Dshapes(points_2d, fit_type="poly2", dist_threshold=0.05):
    # points_2d = remove_bridge(points_2d)
    data = [points_2d[:, 0], points_2d[:, 1]]
    # ellipse
    if fit_type == "ellipse":
        lsqe = el.LSqEllipse()
        lsqe.fit(data)
        # width and height are half real width and height
        ellipse_center, ellipse_width, ellipse_height, ellipse_phi = lsqe.parameters()
        # print ellipse_center, ellipse_width, ellipse_height, ellipse_phi

        def elip(X, P):
            P_cond = (P[0] - ellipse_center[0]) ** 2 / ellipse_width ** 2 + \
                (P[1] - ellipse_center[1]) ** 2 / ellipse_height ** 2 - 1
            X_cond = (X[0] - ellipse_center[0]) ** 2 / ellipse_width ** 2 + \
                (X[1] - ellipse_center[1]) ** 2 / ellipse_height ** 2 - 1
            if P_cond >= 0:
                return -1 * X_cond
            return X_cond
        residuals, fitted_indices = find_min_dist_residual(
            elip, points_2d, dist_threshold)
        return fitted_indices, [ellipse_center, ellipse_width, ellipse_height, ellipse_phi], residuals
    # poly curve
    if fit_type == "poly2":
        poly_coefficients, residuals, _, _, _ = np.polyfit(
            data[0], data[1], 2, full=True)
        fitted_indices = np.arange(data[0].shape[0])
        error = poly_coefficients[0] * data[0]**2 + \
            poly_coefficients[1] * data[0] + poly_coefficients[2] - data[1]
        error = error**2
        fitted_indices = fitted_indices[error < 3]
        # def poly(X, P):
        #    P_cond = poly_coefficients[0] * P[0]**2 +poly_coefficients[1] * P[0] + poly_coefficients[2] - P[1]
        #    X_cond = poly_coefficients[0] * X[0]**2 +poly_coefficients[1] * X[0] + poly_coefficients[2] - X[1]
        #    if P_cond >= 0:
        #        return -1 * X_cond
        #    return X_cond
        #residuals, fitted_indices = find_min_dist_residual(poly, points_2d, dist_threshold)
        # print(residuals)
        # print(residuals/len(fitted_indices))
        return fitted_indices, poly_coefficients, error


def check2Dshapes(points_2d, coefficients, fit_type="poly2",  dist_threshold=0.05):
    # points_2d = remove_bridge(points_2d)
    data = [points_2d[:, 0], points_2d[:, 1]]
    # ellipse
    if fit_type == "ellipse":
        # width and height are half real width and height
        ellipse_center, ellipse_width, ellipse_height, ellipse_phi = coefficients
        # print ellipse_center, ellipse_width, ellipse_height, ellipse_phi

        def elip(X, P):
            P_cond = (P[0] - ellipse_center[0]) ** 2 / ellipse_width ** 2 + \
                (P[1] - ellipse_center[1]) ** 2 / ellipse_height ** 2 - 1
            X_cond = (X[0] - ellipse_center[0]) ** 2 / ellipse_width ** 2 + \
                (X[1] - ellipse_center[1]) ** 2 / ellipse_height ** 2 - 1
            if P_cond >= 0:
                return -1 * X_cond
            return X_cond
        residuals, fitted_indices = find_min_dist_residual(
            elip, points_2d, dist_threshold)
        return fitted_indices, residuals
    # poly curve
    if fit_type == "poly2":
        poly_coefficients = coefficients

        error = poly_coefficients[0] * data[0]**2 + \
            poly_coefficients[1] * data[0] + poly_coefficients[2] - data[1]
        error = np.sqrt(error**2)
        fitted_indices = []  # fitted_indices[error < 3]
        return fitted_indices, error


''' find every points min dist to curve and calculate the avg of square diff of all points'''


def find_min_dist_residual(f, points, dis_threshold):
    def objective(X, P):
        # print "objective:",X,P
        # P is point and X is a point on curve
        x, y = np.ndarray.tolist(np.asarray(X).reshape((2)))
        return np.sqrt((x - P[0]) ** 2 + (y - P[1]) ** 2)

    def c1(X, P):
        X = np.asarray(X).reshape((2))
        P = np.asarray(P).reshape((2))
        return f(X, P)

    summation = 0
    indices = []
    for i in range(points.shape[0]):
        P = points[i, :]
        # print "point of cloud:", P
        X = fmin_cobyla(objective, x0=[P], cons=[
                        c1], args=([P]), consargs=([P]))
        # print objective(X, P)
        dist = objective(X, P)
        if dist <= dis_threshold:
            summation += objective(X, P)
            indices.append(i)
    return summation / points.shape[0], indices
