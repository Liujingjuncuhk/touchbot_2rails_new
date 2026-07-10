import numpy as np
from lemkelcp.lemkelcp import lemketableau
# import quantecon as qe
from scipy.interpolate import BSpline, make_interp_spline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
import open3d as o3d
from scipy.ndimage import gaussian_filter
from scipy import interpolate
from scipy.optimize import minimize
# import lemkelcp as lcp
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from scipy.linalg import solve
import quantecon as qe
from typing import Tuple, Optional
from compecon import LCP


def fit_bspline_surface(point_cloud, u_degree=3, v_degree=3, num_control_points=(10, 12)):
    """
    Fit a B-spline surface to point cloud data using x,z as u,v coordinates.
    
    Parameters:
    point_cloud : ndarray
        Nx3 array of [x, y, z] coordinates
    u_degree : int
        Degree of the spline in u direction (default 3 for cubic)
    v_degree : int
        Degree of the spline in v direction (default 3 for cubic)
    num_control_points : tuple
        Number of control points in (u, v) directions
    
    Returns:
    surface : interpolate.BSpline
        Fitted B-spline surface object
    """
    # Extract coordinates
    x = point_cloud[:, 0]  # u coordinates
    y = point_cloud[:, 1]  # height values to fit
    z = point_cloud[:, 2]  # v coordinates
    
    # Create parameter arrays based on x and z coords
    u = x.copy()
    v = z.copy()
    
    # Normalize u,v to [0,1] range for better numerical stability
    u = (u - u.min()) / (u.max() - u.min())
    v = (v - v.min()) / (v.max() - v.min())
    
    # Create knot vectors
    nu, nv = num_control_points
    u_knots = np.linspace(0, 1, nu + u_degree + 1)
    v_knots = np.linspace(0, 1, nv + v_degree + 1)
    
    # Fit the surface using bisplrep
    # s parameter controls smoothness (smaller = closer fit, larger = smoother)
    surface = interpolate.bisplrep(u, v, y, 
                                  kx=u_degree, ky=v_degree,
                                  s=len(point_cloud))  # smoothing factor
    
    return surface

def visualize_bspline_surface(point_cloud, fitted_surface, grid_size=20):
    """
    Visualize a fitted B-spline surface along with original point cloud,
    ensuring correct orientation.
    
    Parameters:
    point_cloud : ndarray
        Nx3 array of [x, y, z] coordinates
    fitted_surface : scipy.interpolate.BSpline
        Fitted B-spline surface from fit_bspline_surface
    grid_size : int
        Number of points in each direction for surface grid
    """
    # Extract original coordinates
    x = point_cloud[:, 0]  # u direction
    y = point_cloud[:, 1]  # height
    z = point_cloud[:, 2]  # v direction
    
    # Create normalized u,v coordinates matching the fitting process
    u = (x - x.min()) / (x.max() - x.min())
    v = (z - z.min()) / (z.max() - z.min())
    
    # Create grid for surface evaluation
    u_grid = np.linspace(0, 1, grid_size)
    v_grid = np.linspace(0, 1, grid_size)
    U, V = np.meshgrid(u_grid, v_grid)
    
    # Evaluate surface (this gives us the height values)
    Y_fitted = interpolate.bisplev(u_grid, v_grid, fitted_surface)
    # Y_fitted = Y_fitted.flatten()
    # print("Y_fitted size", Y_fitted.shape)
    
    # Scale back to original coordinates
    X_grid = U * (x.max() - x.min()) + x.min()
    Z_grid = V * (z.max() - z.min()) + z.min()
    X_grid = X_grid.T
    Z_grid = Z_grid.T
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface with correct orientation
    # X_grid (x) as X-axis, Z_grid (z) as Y-axis, Y_fitted as Z-axis
    surf = ax.plot_surface(X_grid, Y_fitted, Z_grid, 
                          cmap='viridis',
                          alpha=0.8,
                          rstride=1,
                          cstride=1,
                          linewidth=0.1,
                          antialiased=True)
    
    # Plot original points with matching orientation
    ax.scatter(x, y, z,
              c='red',
              s=20,
              alpha=0.6,
              label='Original Points')
    
    # Labels matching our coordinate system
    ax.set_xlabel('X (u)')
    ax.set_ylabel('Y (height)')
    ax.set_zlabel('Z (v)')
    # axis equal
    max_dist = max([x.max() - x.min(), z.max() - z.min(), y.max() - y.min()])
    # ax.set_box_aspect([x.min() + max_dist, y.min()+ max_dist, z.min()+ max_dist])
    # ax.set_box_aspect([1, 1, 1])
    # plot each point's normal vector

    ax.set_xlim([x.min(), x.min() + max_dist])
    ax.set_ylim([y.min(), y.min() + max_dist])
    ax.set_zlim([z.min(), z.min() + max_dist])
    # ax.set_axis_on()
    ax.set_title('B-spline Surface Fit')
    ax.legend()
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, label='Height')
    
    # Adjust view angle
    ax.view_init(elev=30, azim=90)
    
    plt.tight_layout()
    return fig


def find_normal_vec(pcd, center_line):
    # compute the normal vectgor for each point in pts
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # visualize the point cloud with normals
    normals = np.asarray(pcd.normals)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    points = np.asarray(pcd.points)
    minz = np.min(points[:, 2])
    max_z = np.max(points[:, 2])
    seg_length = (max_z - minz) / 100
    normals = np.asarray(pcd.normals)
    for i in range(len(normals)):
        # check with the center line's point
        seg_num = int((points[i, 2] - minz) / (seg_length+1))
        center_line_point = center_line[seg_num]
        # compute the direction vector from the center line point to the point
        direction_vector = points[i,:2] - center_line_point[:2]
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        norm_xy = normals[i, :2]
        if np.dot(norm_xy, direction_vector) < 0:
            normals[i] = -normals[i]
    num_points = len(points)
    normal_length = 1e-2
    normal_endpoints = points + normal_length * normals
    normal_endpoints = points + normal_length * normals
    lines = [[i, i + num_points] for i in range(num_points)]
    all_points = np.vstack((points, normal_endpoints))
    
    # Create LineSet for normals
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(num_points)])  # Red normals
    
    # Visualize point cloud and normals
    # o3d.visualization.draw_geometries([pcd, line_set], window_name="Point Cloud with Normals")

    return normals

def find_center_line(pcd, n_seg=100):
    # Compute the center line of the arm contour point cloud
    # n_seg = 100
    # split the z range into n_seg segments
    z_min = np.min(np.asarray(pcd.points)[:, 2])
    z_max = np.max(np.asarray(pcd.points)[:, 2])
    z_range = np.linspace(z_min, z_max, n_seg+1)
    center_line = []
    for i in range(n_seg):
        zrange_min = z_range[i]
        zrange_max = z_range[i+1]
        # find the points in the z range
        points_in_range = np.asarray(pcd.points)[(np.asarray(pcd.points)[:, 2] >= zrange_min) & (np.asarray(pcd.points)[:, 2] <= zrange_max)]
        if len(points_in_range) > 0:
            # fit the point using a circle
            point_xy = points_in_range[:, :2]
            xc, yc, r = fit_circle(point_xy)
            center_line.append([xc, yc, zrange_min + (zrange_max - zrange_min) / 2])
    if len(center_line) == 0:
        print("No points found in the z range.")
        return None
    center_line = np.array(center_line)
    # visualize the center line
    center_line_pcd = o3d.geometry.PointCloud()
    center_line_pcd.points = o3d.utility.Vector3dVector(center_line)
    # o3d.visualization.draw_geometries([pcd, center_line_pcd], window_name="Center Line of Arm Contour")
    return center_line

def fit_circle(points):
    # Fit a circle to the points in the xy plane
    x = points[:, 0]
    y = points[:, 1]
    A = np.c_[x, y, np.ones_like(x)]
    B = x**2 + y**2
    C, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc, r = C[0]/2, C[1]/2, np.sqrt(C[2] + (C[0]**2 + C[1]**2)/4)
    return xc, yc, r


def min_distance_and_direction(kdtree,point_cloud, normals, query_point):
    
    # Build a KDTree for efficient nearest neighbor search
    # kdtree = cKDTree(point_cloud)
    # find the closest point in the point cloud
    distance, index = kdtree.query(query_point)
    closest_point = point_cloud[index]
    distance = np.linalg.norm(closest_point - query_point)
    normal = normals[index]
    unit_vec = (closest_point - query_point) / distance
    is_inside = False
    if np.dot(unit_vec, normal) > 0:
        is_inside = True
    return is_inside, distance, unit_vec, closest_point

def min_distance_and_direction_v2(kdtree, point_cloud, B_spline_surface, query_point, xzranges):
    x_min, x_max, z_min, z_max = xzranges
    u_query = (query_point[0] - x_min) / (x_max - x_min)
    v_query = (query_point[2] - z_min) / (z_max - z_min)
    Y_fitted = interpolate.bisplev(u_query, v_query, B_spline_surface)
    def distance_to_surface(uv):
        u, v = uv
        y_surface = interpolate.bisplev(u, v, B_spline_surface)
        # Convert u,v back to x,z (assuming same normalization as in fitting)
        x = u * (x_max - x_min) + x_min
        z = v * (z_max - z_min) + z_min
        surface_point = np.array([x, y_surface, z])
        return np.sum((surface_point - query_point) ** 2)
    if query_point[1] < Y_fitted:
        is_inside = False
        distance, index = kdtree.query(query_point)
        closest_point = point_cloud[index]
        u_ig, v_ig = (closest_point[0]-x_min)/(x_max-x_min), (closest_point[2]-z_min)/(z_max-z_min)
        u_offset = 0.01 # 1% of the range
        v_range = 0.01
        u_range = [u_ig - u_offset, u_ig + u_offset]
        v_range = [v_ig - v_range, v_ig + v_range]
        result = minimize(distance_to_surface,
                        [u_ig, v_ig],  # Initial guess in middle of parameter space
                        bounds=[u_range, v_range])
        u_closest, v_closest = result.x
        y_closest = interpolate.bisplev(u_closest, v_closest, B_spline_surface)
        x_closest = u_closest * (x_max - x_min) + x_min
        z_closest = v_closest * (z_max - z_min) + z_min
        surface_point = np.array([x_closest, y_closest, z_closest])
        dist_final = np.linalg.norm(surface_point - query_point)
        return False, dist_final, 0, 0
    else:
        is_inside = True
        distance, index = kdtree.query(query_point)
        closest_point = point_cloud[index]
        u_ig, v_ig = (closest_point[0]-x_min)/(x_max-x_min), (closest_point[2]-z_min)/(z_max-z_min)
        u_offset = 0.01 # 1% of the range
        v_range = 0.01
        u_range = [u_ig - u_offset, u_ig + u_offset]
        v_range = [v_ig - v_range, v_ig + v_range]
        
        result = minimize(distance_to_surface, 
                        [u_ig, v_ig],  # Initial guess in middle of parameter space
                        bounds=[u_range, v_range])
        
        u_closest, v_closest = result.x
        y_closest = interpolate.bisplev(u_closest, v_closest, B_spline_surface)
        x_closest = u_closest * (x_max - x_min) + x_min
        z_closest = v_closest * (z_max - z_min) + z_min
        surface_point = np.array([x_closest, y_closest, z_closest])
        
        delta_forNormal = 1e-5
        y_u_p = interpolate.bisplev(u_closest + delta_forNormal, v_closest, B_spline_surface)
        y_u_m = interpolate.bisplev(u_closest - delta_forNormal, v_closest, B_spline_surface)
        y_v_p = interpolate.bisplev(u_closest, v_closest + delta_forNormal, B_spline_surface)
        y_v_m = interpolate.bisplev(u_closest, v_closest - delta_forNormal, B_spline_surface)
        du_point = np.array([(u_closest + delta_forNormal) * (x_max - x_min) + x_min, y_u_p, 
                            v_closest * (z_max - z_min) + z_min]) - np.array([(u_closest - delta_forNormal) * (x_max - x_min) + x_min, y_u_m, 
                            v_closest * (z_max - z_min) + z_min])

        dv_point = np.array([u_closest * (x_max - x_min) + x_min, y_v_p,
                            (v_closest + delta_forNormal) * (z_max - z_min) + z_min]) - np.array([u_closest * (x_max - x_min) + x_min, y_v_m,
                            (v_closest - delta_forNormal) * (z_max - z_min) + z_min])

        normal = np.cross(du_point, dv_point)
        normal = normal / np.linalg.norm(normal)
        if normal[1] > 0:
            normal = -normal

        unit_direction = (surface_point - query_point) / np.linalg.norm(surface_point - query_point)
        
        return is_inside, distance, unit_direction, closest_point

def augument_point_cloud(point_cloud, B_spline_surface):
    minx = point_cloud[:, 0].min()
    maxx = point_cloud[:, 0].max() 
    minz = point_cloud[:, 2].min()
    maxz = point_cloud[:, 2].max()
    u_new = np.linspace(0,1, 40)
    v_new = np.linspace(0,1, 40)
    U, V = np.meshgrid(u_new, v_new)
    Y_fitted = interpolate.bisplev(u_new, v_new, B_spline_surface)
    X = U * (maxx - minx) + minx
    Z = V * (maxz - minz) + minz
    X = X.T
    Z = Z.T
    Y = Y_fitted.flatten()
    X = X.flatten()
    Z = Z.flatten()
    augumented_point_cloud = np.vstack((X, Y, Z)).T
    return augumented_point_cloud

def visuliaze_point_cloud_closest_point(point_cloud, B_spline_surface, query_point, unit_direction, closest_point):
    x = point_cloud[:, 0]  # u direction
    y = point_cloud[:, 1]  # height
    z = point_cloud[:, 2]  # v direction
    grid_size = 20
    # Create normalized u,v coordinates matching the fitting process
    u = (x - x.min()) / (x.max() - x.min())
    v = (z - z.min()) / (z.max() - z.min())
    
    # Create grid for surface evaluation
    u_grid = np.linspace(0, 1, grid_size)
    v_grid = np.linspace(0, 1, grid_size)
    U, V = np.meshgrid(u_grid, v_grid)
    
    # Evaluate surface (this gives us the height values)
    Y_fitted = interpolate.bisplev(u_grid, v_grid, B_spline_surface)
    X_grid = U * (x.max() - x.min()) + x.min()
    Z_grid = V * (z.max() - z.min()) + z.min()
    X_grid = X_grid.T
    Z_grid = Z_grid.T

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface with correct orientation
    # X_grid (x) as X-axis, Z_grid (z) as Y-axis, Y_fitted as Z-axis
    surf = ax.plot_surface(X_grid, Y_fitted, Z_grid, 
                          cmap='viridis',
                          alpha=0.8,
                          rstride=1,
                          cstride=1,
                          linewidth=0.1,
                          antialiased=True)
    # print all points
    ax.scatter(x, y, z)

    ax.set_xlabel('X (u)')
    ax.set_ylabel('Y (height)')
    ax.set_zlabel('Z (v)')
    # Plot query_point with direction vector
    ax.scatter(query_point[0], query_point[1], query_point[2],
              c='red',
              s=20,
              alpha=0.6,
              label='Query Point')
    ax.quiver(query_point[0], query_point[1], query_point[2], unit_direction[0], unit_direction[1], unit_direction[2], color='red', label='Unit Direction')

    # plot the closest point
    ax.scatter(closest_point[0], closest_point[1], closest_point[2],
              c='blue',
                s=20,
                alpha=0.6,
                label='Closest Point')
    max_dist = max([x.max() - x.min(), z.max() - z.min(), y.max() - y.min()])
    # ax.set_box_aspect([x.min() + max_dist, y.min()+ max_dist, z.min()+ max_dist])
    # ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([x.min(), x.min() + max_dist])
    ax.set_ylim([y.min(), y.min() + max_dist])
    ax.set_zlim([z.min(), z.min() + max_dist])
    plt.show()

def visuliaze_augmented_original_pts(point_cloud, augumented_point_cloud, B_spline_surface):
    x = point_cloud[:, 0]  # u direction
    y = point_cloud[:, 1]  # height
    z = point_cloud[:, 2]  # v direction
    grid_size = 20
    # Create normalized u,v coordinates matching the fitting process
    u = (x - x.min()) / (x.max() - x.min())
    v = (z - z.min()) / (z.max() - z.min())
    
    # Create grid for surface evaluation
    u_grid = np.linspace(0, 1, grid_size)
    v_grid = np.linspace(0, 1, grid_size)
    U, V = np.meshgrid(u_grid, v_grid)
    
    # Evaluate surface (this gives us the height values)
    Y_fitted = interpolate.bisplev(u_grid, v_grid, B_spline_surface)
    X_grid = U * (x.max() - x.min()) + x.min()
    Z_grid = V * (z.max() - z.min()) + z.min()
    X_grid = X_grid.T
    Z_grid = Z_grid.T

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface with correct orientation
    # X_grid (x) as X-axis, Z_grid (z) as Y-axis, Y_fitted as Z-axis
    surf = ax.plot_surface(X_grid, Y_fitted, Z_grid, 
                          cmap='viridis',
                          alpha=0.8,
                          rstride=1,
                          cstride=1,
                          linewidth=0.1,
                          antialiased=True)
    # print all points
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='red', label='Original Point Cloud')
    ax.scatter(augumented_point_cloud[:, 0], augumented_point_cloud[:, 1], augumented_point_cloud[:, 2], c='blue', label='Augumented Point Cloud')
    ax.set_xlabel('X (u)')
    ax.set_ylabel('Y (height)')
    ax.set_zlabel('Z (v)')
    max_dist = max([x.max() - x.min(), z.max() - z.min(), y.max() - y.min()])
    ax.set_xlim([x.min(), x.min() + max_dist])
    ax.set_ylim([y.min(), y.min() + max_dist])
    ax.set_zlim([z.min(), z.min() + max_dist])
    plt.show()

def cal_vol_tet(tet):
    """
    Calculate the volume of a tetrahedron given its vertices.

    Parameters:
    - tet: A 4x3 numpy array containing the vertices of the tetrahedron.

    Returns:
    - volume: The volume of the tetrahedron.
    """
    # Extract the vertices
    a, b, c, d = tet

    # Calculate the volume using the determinant formula
    volume = np.abs(np.linalg.det(np.array([a - d, b - d, c - d])) / 6)

    return volume

def R33_2_R1212(R_list_33):
    nR = len(R_list_33)
    R_list_1212 = [np.eye(12) for _ in range(nR)]   
    for i in range(nR):
        R_list_1212[i][0:3, 0:3] = R_list_33[i].copy()
        R_list_1212[i][3:6, 3:6] = R_list_33[i].copy()
        R_list_1212[i][6:9, 6:9] = R_list_33[i].copy()
        R_list_1212[i][9:12, 9:12] = R_list_33[i].copy()
    return R_list_1212

def myLCP(M, q, max_iter=1000, tol=1e-10):
    """
    Solve the Linear Complementarity Problem (LCP):
        w = Mz + q, w >= 0, z >= 0, w^T z = 0

    Uses Lemke's algorithm (simplified version).

    Args:
        M: (n,n) matrix
        q: (n,) vector
        max_iter: maximum iterations
        tol: tolerance for termination

    Returns:
        z: solution vector
        w: w = Mz + q
    """
    n = len(q)
    z = np.zeros(n)
    w = M @ z + q

    if np.all(w >= 0):
        return z, w

    # Artificial variable
    t = n
    tableau = np.hstack((M, -np.eye(n)))
    tableau = np.vstack((tableau, np.hstack((-np.eye(n), np.zeros((n, n))))))

    rhs = np.hstack((-q, np.zeros(n)))
    basis = list(range(n, 2*n))

    leaving = np.argmin(rhs[:n])
    if rhs[leaving] >= 0:
        return z, w

    entering = t

    z = np.zeros(n)
    for _ in range(max_iter):
        col = np.zeros(2*n)
        col[leaving] = 1

        d = np.linalg.solve(tableau, col)
        ratios = []
        for i in range(2*n):
            if d[i] > tol:
                ratios.append(rhs[i]/d[i])
            else:
                ratios.append(np.inf)

        min_ratio = np.min(ratios)
        if min_ratio == np.inf:
            raise ValueError("LCP has no solution.")

        leaving = np.argmin(ratios)
        basis[leaving] = entering

        rhs = rhs - min_ratio * d

        if entering == t:
            entering = basis[leaving]
        else:
            entering = t

        if entering < n:
            break

    z = np.maximum(rhs[:n], 0)
    w = M @ z + q
    return z, w


def myPalm_size4(q):
    # check if q satisfies the constraints
    x,y,z = q
    x = x*1e3
    y = y*1e3
    z = z*1e3
    r = 32
    r_test = 33
    xc = 40
    yc = 5 + r
    # print("xc, yc", xc, yc)
    if x < 20: 
        return 1
    if x < 40 and x > 20:
        return 2-y
    else:
        return (x-xc)**2 + (y-yc)**2 - r_test**2

def myPalm_size3(q):
    # check if q satisfies the constraints, if negative, then it is a contact vertice
    x,y,z = q
    x = x*1e3
    y = y*1e3
    z = z*1e3
    r = 27
    r_test = 28
    xc = 40
    yc = 5 + r
    # print("xc, yc", xc, yc)
    if x < 20: 
        return 1
    if x < 40 and x > 20:
        return 2-y
    else:
        return (x-xc)**2 + (y-yc)**2 - r_test**2

def myPalm_size3_feedback(q):
    x,y,z = q
    x = x*1e3
    y = y*1e3
    z = z*1e3
    r = 27
    r_test = 28
    xc = 40
    yc = 5 + r
    # print("xc, yc", xc, yc)
    if x < 20: 
        return 1
    if z < 20 or z > 45:
        return 1
    if y > 30:
        return 1
    if x < 40 and x > 20:
        return y-2
    else:
        return r_test**2 - ((x-xc)**2 + (y-yc)**2) 

def myPalm_size2(q):
    # check if q satisfies the constraints
    x,y,z = q
    x = x*1e3
    y = y*1e3
    z = z*1e3
    r = 22
    r_test = 23
    xc = 40
    yc = 5 + r
    # print("xc, yc", xc, yc)
    if x < 20: 
        return 1
    if x < 40 and x > 20:
        return 2-y
    else:
        return (x-xc)**2 + (y-yc)**2 - r_test**2

def myPalm_size2_feedback(q):
    # check if q satisfies the constraints
    x,y,z = q
    x = x*1e3
    y = y*1e3
    z = z*1e3
    r = 22
    r_test = 23
    xc = 40
    yc = 5 + r
    # print("xc, yc", xc, yc)
    if x < 20: 
        return 1
    if z < 20 or z > 45:
        return 1
    if y > 30:
        return 1
    if x < 40 and x > 20:
        return y-2
    else:
        return r_test**2 - ((x-xc)**2 + (y-yc)**2) 

def lcp_test():
    # Define the matrix M and vector q
    M = np.array([[2, 1], [1, 2]])
    q = np.array([-5, -6])

    z, w = myLCP(M, q)
    print("z:", z)
    print("w:", w)
    print("Complementarity check:", np.dot(z, w))


def run_icp(pointcloud_a, pointcloud_b, max_iterations=100, tolerance=1e-6):
    """
    Perform ICP to align pointcloud_b to pointcloud_a.
    Both point clouds are Nx3 numpy arrays.
    Returns the transformation matrix (4x4) and aligned pointcloud_b.
    """
    # Ensure inputs are numpy arrays
    src = np.copy(pointcloud_b)  # Point cloud B (source)
    dst = np.copy(pointcloud_a)  # Point cloud A (target)
    
    # Initialize transformation matrix
    T = np.eye(4)
    
    # Build KDTree for target points
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(dst)
    
    prev_error = float('inf')
    
    for _ in range(max_iterations):
        # Find nearest neighbors
        distances, indices = nbrs.kneighbors(src)
        matched_points = dst[indices.flatten()]
        
        # Compute centroids
        src_centroid = np.mean(src, axis=0)
        dst_centroid = np.mean(matched_points, axis=0)
        
        # Center the point clouds
        src_centered = src - src_centroid
        dst_centered = matched_points - dst_centroid
        
        # Compute rotation using SVD
        H = np.dot(src_centered.T, dst_centered)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        # Ensure proper rotation (determinant = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Compute translation
        t = dst_centroid - np.dot(R, src_centroid)
        
        # Update transformation matrix
        T_iter = np.eye(4)
        T_iter[:3, :3] = R
        T_iter[:3, 3] = t
        
        # Apply transformation to source points
        src_h = np.hstack((src, np.ones((src.shape[0], 1))))
        src = np.dot(T_iter, src_h.T).T[:, :3]
        
        # Update cumulative transformation
        T = np.dot(T_iter, T)
        
        # Compute error
        error = np.mean(distances)
        
        # Check convergence
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error
    
    # Return transformation matrix and aligned points
    return T, src

def projected_gauss_seidel_lcp(
    M: np.ndarray,
    q: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-8,
    omega: float = 1.3,        # over-relaxation (1.0 = pure GS, 1.3–1.9 common in robotics)
    warm_start: np.ndarray | None = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Solve LCP:  w = M z + q >= 0,  z >= 0,  z^T w = 0
    using Projected Gauss-Seidel (successive over-relaxation).
    
    Returns the least 2-norm solution (minimum impulse) — the physically correct one in robotics.
    
    Parameters
    ----------
    M : (n,n) np.ndarray
        Usually symmetric positive semi-definite in robotics (e.g., Delassus matrix)
    q : (n,) np.ndarray
    max_iter, tol, omega : convergence controls
    warm_start : optional initial guess (greatly speeds up sequential solves)
    
    Returns
    -------
    z : (n,) solution vector
    """
    n = len(q)
    M = np.asarray(M, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Diagonal dominance check (optional warning)
    diag = np.diag(M)
    if np.any(diag <= 0):
        print("Warning: M has non-positive diagonal entries → may diverge!")
    
    z = np.zeros(n, dtype=np.float64) if warm_start is None else warm_start.copy()
    w = M @ z + q
    
    for iteration in range(max_iter):
        z_old = z.copy()
        
        for i in range(n):
            if M[i,i] == 0:
                z[i] = 0  # avoid division by zero
                continue
                
            # Predicted w_i if z_i were free
            w_pred = q[i] + M[i,:i] @ z[:i] + M[i,i+1:] @ z[i+1:]
            
            # Gauss-Seidel update + projection onto z_i >= 0 and w_i >= 0
            z_new = max(0.0, z[i] - omega * (M[i,i] * z[i] + w_pred) / M[i,i])
            
            # Apply relaxation
            z[i] = z_new
            
            # Optional: update w incrementally (saves one full Mv per sweep)
            # w[i] = M[i,i] * z[i] + w_pred
        
        # Full residual (for convergence check)
        w = M @ z + q
        residual = np.maximum(z, w)  # violation = max(z_i, w_i) when one should be zero
        res_norm = np.linalg.norm(residual, np.inf)
        
        if res_norm < tol:
            if verbose:
                print(f"Converged in {iteration+1} iterations (res = {res_norm:.2e})")
            return z
        
        if np.linalg.norm(z - z_old) < tol:
            if verbose:
                print(f"Stalled at iteration {iteration+1}, res = {res_norm:.2e}")
            break
            
    if verbose:
        print(f"Max iterations reached. Final residual: {res_norm:.2e}")
    
    return z

def sample_rows(arr, n):
    """
    Randomly sample n rows from a NumPy array.
    If n exceeds the number of rows, return the original array.
    
    Parameters:
    arr (np.ndarray): Input array
    n (int): Number of rows to sample
    
    Returns:
    np.ndarray: Array with n randomly sampled rows or original array if n > number of rows
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
        
    num_rows = arr.shape[0]
    
    if n >= num_rows:
        return arr
    
    indices = np.random.choice(num_rows, size=n, replace=False)
    return arr[indices]

def apply_transformation(pointcloud, T):
    """
    Apply a 4x4 homogeneous transformation matrix to a point cloud.
    
    Parameters:
    pointcloud (np.ndarray): Input array of shape (N, 3) representing 3D points
    T (np.ndarray): 4x4 homogeneous transformation matrix
    
    Returns:
    np.ndarray: Transformed point cloud of shape (N, 3)
    """
    # Ensure inputs are numpy arrays
    pointcloud = np.asarray(pointcloud)
    T = np.asarray(T)
    
    # Validate input shapes
    if pointcloud.shape[1] != 3:
        raise ValueError("Point cloud must have 3 columns (x, y, z)")
    if T.shape != (4, 4):
        raise ValueError("Transformation matrix must be 4x4")
    
    # Convert points to homogeneous coordinates (N, 4)
    points_h = np.hstack((pointcloud, np.ones((pointcloud.shape[0], 1))))
    
    # Apply transformation
    transformed_h = np.dot(points_h, T.T)
    
    # Return only the 3D coordinates (N, 3)
    return transformed_h[:, :3]


def thin_plate_spline(x, y, alpha=0.0):
    """
    Compute Thin Plate Spline transformation from source (x) to target (y) points.
    x, y: Nx3 numpy arrays of corresponding points
    alpha: regularization parameter
    Returns: function to transform points, affine matrix, and TPS weights
    """
    n, d = x.shape
    # Compute TPS kernel matrix
    r = np.sum((x[:, None, :] - x[None, :, :])**2, axis=2)
    K = r * np.log(np.sqrt(r + 1e-10))
    np.fill_diagonal(K, alpha)  # Regularization on diagonal
    P = np.hstack((np.ones((n, 1)), x))
    # Construct system matrix
    L = np.vstack([
        np.hstack([K, P]),
        np.hstack([P.T, np.zeros((d+1, d+1))])
    ])
    Y = np.vstack([y, np.zeros((d+1, d))])
    # Solve for TPS parameters
    params = solve(L, Y, assume_a='sym')
    w = params[:n]  # TPS weights
    a = params[n:]  # Affine components
    def transform(x_new):
        r_new = np.sum((x_new[:, None, :] - x[None, :, :])**2, axis=2)
        K_new = r_new * np.log(np.sqrt(r_new + 1e-10))
        P_new = np.hstack((np.ones((x_new.shape[0], 1)), x_new))
        return np.dot(K_new, w) + np.dot(P_new, a)
    return transform, a, w

def tps_rpm(pointcloud_a, pointcloud_b, n_iter=10, reg_init=0.1, reg_final=0.001, rad_init=0.1, rad_final=0.001):
    """
    TPS-RPM algorithm to align pointcloud_b to pointcloud_a.
    pointcloud_a, pointcloud_b: Nx3 numpy arrays
    n_iter: number of iterations
    reg_init, reg_final: initial and final regularization
    rad_init, rad_final: initial and final radius for correspondence
    Returns: transformation function, transformed pointcloud_b, correspondence matrix
    """
    src = np.copy(pointcloud_b)  # Source (B)
    dst = np.copy(pointcloud_a)  # Target (A)
    n, m = src.shape[0], dst.shape[0]
    
    # Annealing schedules
    regs = np.logspace(np.log10(reg_init), np.log10(reg_final), n_iter)
    rads = np.logspace(np.log10(rad_init), np.log10(rad_final), n_iter)
    
    # Initialize transformation
    transform = lambda x: x  # Identity transform initially
    corr_nm = np.ones((n, m)) / m  # Initial uniform correspondence
    
    for i in range(n_iter):
        # Compute warped source points
        src_warped = transform(src)
        
        # Compute correspondence matrix using Sinkhorn normalization
        tree = cKDTree(dst)
        distances, indices = tree.query(src_warped, k=1)
        corr_nm = np.zeros((n, m))
        corr_nm[np.arange(n), indices] = np.exp(-distances**2 / (2 * rads[i]**2))
        corr_nm /= (corr_nm.sum(axis=1)[:, None] + 1e-10)  # Normalize rows
        
        # Compute target points as weighted average
        targ = np.dot(corr_nm, dst) / (corr_nm.sum(axis=1)[:, None] + 1e-10)
        
        # Fit TPS transformation
        transform, _, _ = thin_plate_spline(src, targ, alpha=regs[i])
        
        # Update source points
        src_warped = transform(src)
    
    # Final transformation
    transformed_b = transform(src)
    return transform, transformed_b, corr_nm

def checkpoint(info = ""):
    print("checkpoint: ", info)
    key = input("press enter to continue or q to quit")
    if key == 'q':
        exit()


def closest_point_sdf(p, sdf_func, grad_func=None, eps=1e-6, max_iter=4)->tuple:
    """
    p          : np.array([x,y,z]) – query point
    sdf_func   : callable(p) -> float   – signed distance
    grad_func  : callable(p) -> np.array([nx,ny,nz]) – optional analytic gradient
    eps        : tolerance for "on surface"
    max_iter   : safeguard for numerical gradients
    """
    d = sdf_func(p)
    if d < 0:
        is_inside = 1
    else:
        is_inside = 0

    # 2. Compute unit normal
    if grad_func is not None:
        g = grad_func(p)
    else:
        # central finite differences (6 evaluations)
        g = np.zeros(3)
        for i in range(3):
            e = np.zeros(3)
            e[i] = eps
            g[i] = sdf_func(p + e) - sdf_func(p - e)
        g /= 2*eps

    norm_g = np.linalg.norm(g)
    if norm_g < 1e-8:                     # singularity (e.g. inside a sharp corner)
        # fall back to a short Newton-style iteration
        p = _newton_projection(p, sdf_func, eps, max_iter)

    n = g / norm_g

    # 3. Project
    q = p - d * n
    # print("q: ",q, "p: ",p)
    # return whether inside and the unit vector from q to p
    return is_inside, -np.linalg.norm(q-p), (q-p)/np.linalg.norm(q-p)

def _newton_projection(p, sdf, eps, max_iter):
    x = p.copy()
    for _ in range(max_iter):
        d = sdf(x)
        if abs(d) < eps:
            return x
        g = np.zeros(3)
        for i in range(3):
            e = np.zeros(3); e[i] = eps
            g[i] = sdf(x + e) - sdf(x - e)
        g /= 2*eps
        n = g / (np.linalg.norm(g) + 1e-12)
        x = x - d * n
    return x

if __name__ == '__main__':
    lcp_test()