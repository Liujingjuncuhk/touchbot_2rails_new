import numpy as np
import pyvista as pv
from utilities import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class cableDrivenSurface:
    def __init__(self, file: str, pp_location, pulley_location, initial_rotation = 15.0): # rotation is along -z axis
        self.initial_rotation = initial_rotation
        # get a rotation mat for the initial rotation along z axis
        theta = np.radians(initial_rotation)
        self.Rz = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        
        self.N44 = np.eye(4) - 1/4*np.ones((4, 4))
        # construct vertices matrix and tet-vertices matrix
        # read the file, first line has number of vertices and tetrahedra
        with open(file, 'r') as f:
            self.num_vert, self.num_tet = map(int, f.readline().split())
            vertices = np.array([list(map(float, f.readline().split())) for _ in range(self.num_vert)])
            tetrahedra = np.array([list(map(int, f.readline().split())) for _ in range(self.num_tet)])
        vertices = vertices * 1e-3
        
        self.vertices = vertices
        self.fixed_region = [[-0.001, 0.001],[-0.01, 0.20],[-0.01, 0.30]]
        self.set_fixed_region(self.fixed_region)
        self.tetrahedra = tetrahedra
        self.triangle_list = []
        for i in range(self.num_tet):
            tet = self.tetrahedra[i]
            # check if already in the triangle list
            for j in range(4):
                for k in range(j+1, 4):
                    for l in range(k+1, 4):
                        triangle = [tet[j], tet[k], tet[l]]
                        triangle.sort()
                        if triangle not in self.triangle_list:
                            self.triangle_list.append(triangle)
        # print("triangle_list: ", self.triangle_list)
        self.triangle_list = np.array(self.triangle_list)
        
        print("number of triangle: ", self.triangle_list.shape[0])
        self.contact_vertices = []
        for i in range(self.num_vert):
            if myPalm_size2(vertices[i]) < 0 and self.idxAll_2_idxMoving[i] != -1:
                self.contact_vertices.append(i)
        self.contact_triangles = []
        for i in range(self.triangle_list.shape[0]):
            if self.triangle_list[i][0] in self.contact_vertices and self.triangle_list[i][1] in self.contact_vertices and self.triangle_list[i][2] in self.contact_vertices:
                self.contact_triangles.append(i)

        self.pp_location = pp_location
        self.pulley_location = pulley_location
        self.nCable = pp_location.shape[0]
        

        pp_idx = [0 for _ in range(self.nCable)]
        # fine the index of the closest vertex to the pp_location
        for i in range(self.nCable):
            pp_idx[i] = np.argmin(np.linalg.norm(self.vertices - pp_location[i], axis=1))
        self.pp_idx = pp_idx

        for i in range(self.num_vert):
            self.vertices[i] = np.dot(self.Rz, self.vertices[i])
        self.cur_vertices = self.vertices.copy()

        for i in range(self.nCable):
            self.pp_location[i] = np.dot(self.Rz, self.pp_location[i])
        
        self.vol_list = [0 for _ in range(self.num_tet)]
        for i in range(self.num_tet):
            self.vol_list[i] = cal_vol_tet(vertices[tetrahedra[i]])
        self.weight_list = self.vol_list.copy()
        # print("weight_list: ", self.weight_list)
        
        self.neigbour_list = [[] for _ in range(self.num_vert)]
        for i in range(self.num_tet):
            for j in range(4):
                idxj = self.tetrahedra[i][j]
                for k in range(4):
                    if j != k and k not in self.neigbour_list[j]:
                        self.neigbour_list[j].append(k)
        self.vert_2_tet = [[] for _ in range(self.num_vert)]
        for i in range(self.num_tet):
            for j in range(4):
                if i not in self.vert_2_tet[self.tetrahedra[i][j]]:
                    self.vert_2_tet[self.tetrahedra[i][j]].append(i)
        
        

        self.contact_triangle_area = []
        for i in range(len(self.contact_triangles)):
            tri = self.triangle_list[self.contact_triangles[i]]
            v1 = vertices[tri[0]]
            v2 = vertices[tri[1]]
            v3 = vertices[tri[2]]
            self.contact_triangle_area.append(np.linalg.norm(np.cross(v2-v1, v3-v1))/2)


        self.add_material_properties()
        self.mass_mat = np.zeros((3*self.num_vert, 3*self.num_vert))
        self.W_mat = np.zeros((3*self.num_vert, 3*self.num_vert))
        self.gravity_vec = np.zeros((3*self.num_vert, 1))
        for i in range(self.num_tet):
            mass_tet = self.weight_list[i] * self.density 
            for j in range(4):
                idx = self.tetrahedra[i][j]
                self.gravity_vec[3*idx + 1] += mass_tet * 9.81 / 4
                for k in range(3):
                    self.mass_mat[3*idx + k, 3*idx + k] += mass_tet / 4
        
        for i in range(self.num_vert):
            if self.idxAll_2_idxMoving[i] != -1:
                for j in range(3):
                    self.W_mat[3*i + j, 3*i + j] = 1.0/self.mass_mat[3*i + j, 3*i + j]
            else:
                for j in range(3):
                    self.W_mat[3*i + j, 3*i + j] = 0

        self.original_tet_sk = []
        for i in range(self.num_tet):
            tet = self.tetrahedra[i]
            tet_vert = self.vertices[tet]
            tet_sk = self.N44 @ tet_vert
            self.original_tet_sk.append(tet_sk)

        self.initial_cable_length = []
        for i in range(self.nCable):
            self.initial_cable_length.append(np.linalg.norm(vertices[self.pp_idx[i]] - pulley_location[i]))

        # weight cable is the max of weight_list
        self.weight_cable = max(self.weight_list)*2
        # self.add_ws_points([162,55])
        self.add_ee_pos(np.array([[0.15, 0.0, 0.02]]))
        # self.weight_ee_list = [1]
        # self.add_ws_points([162])
        print("Number of tets: ", self.num_tet)
        print("Number of vertices: ", self.num_vert)
        return 

    def set_fixed_region(self, fixed_region):
        self.fixed_region = fixed_region
        fixed_idx = []
        for i in range(self.num_vert):
            if fixed_region[0][0] <= self.vertices[i][0] <= fixed_region[0][1] and \
                fixed_region[1][0] <= self.vertices[i][1] <= fixed_region[1][1] and \
                fixed_region[2][0] <= self.vertices[i][2] <= fixed_region[2][1]:
                fixed_idx.append(i)
        self.fixed_idx = fixed_idx
        self.nMoving = self.num_vert - len(fixed_idx)
        self.idxAll_2_idxMoving = [-1 for _ in range(self.num_vert)]
        self.idxMoving_2_idxAll = []
        idx_moving = 0
        for i in range(self.num_vert):
            if i not in fixed_idx:
                self.idxAll_2_idxMoving[i] = idx_moving
                self.idxMoving_2_idxAll.append(i)
                idx_moving += 1
        return
      
    def add_material_properties(self):
        self.Youngs_modulus = 4.3e6
        self.Poisson_ratio = 0.4
        self.density = 1160
        self.Bmat_list = []
        self.stiffness_matrix_list = []
        E = self.Youngs_modulus
        nu = self.Poisson_ratio
        # Dmat is the material stiffness matrix, 12x12
        l = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        self.Dmat = np.array([[l + 2 * mu, l, l, 0., 0., 0.],[l , l+ 2 * mu, l, 0., 0., 0.],[l , l, l+ 2 * mu, 0., 0., 0.],[0, 0, 0, mu, 0, 0],[0, 0, 0, 0, mu, 0],[0, 0, 0, 0, 0, mu]])
        for i in range(self.num_tet):
            Bmat = np.zeros((6, 12))
            one_over_6_vol = 1.0 / (6.0 * self.weight_list[i])
            tet = self.tetrahedra[i]
            x1 = self.vertices[tet[0], 0]
            x2 = self.vertices[tet[1], 0]
            x3 = self.vertices[tet[2], 0]
            x4 = self.vertices[tet[3], 0]
            y1 = self.vertices[tet[0], 1]
            y2 = self.vertices[tet[1], 1]
            y3 = self.vertices[tet[2], 1]
            y4 = self.vertices[tet[3], 1]
            z1 = self.vertices[tet[0], 2]
            z2 = self.vertices[tet[1], 2]
            z3 = self.vertices[tet[2], 2]
            z4 = self.vertices[tet[3], 2]
            b1 = (y2 * (z3 - z4) + y3 * (z4 - z2) + y4 * (z2 - z3)) * one_over_6_vol
            b2 = (y1 * (z4 - z3) + y3 * (z1 - z4) + y4 * (z3 - z1)) * one_over_6_vol
            b3 = (y1 * (z2 - z4) + y2 * (z4 - z1) + y4 * (z1 - z2)) * one_over_6_vol
            b4 = (y1 * (z3 - z2) + y2 * (z1 - z3) + y3 * (z2 - z1)) * one_over_6_vol
            c1 = (z2 * (x3 - x4) + z3 * (x4 - x2) + z4 * (x2 - x3)) * one_over_6_vol
            c2 = (z1 * (x4 - x3) + z3 * (x1 - x4) + z4 * (x3 - x1)) * one_over_6_vol
            c3 = (z1 * (x2 - x4) + z2 * (x4 - x1) + z4 * (x1 - x2)) * one_over_6_vol
            c4 = (z1 * (x3 - x2) + z2 * (x1 - x3) + z3 * (x2 - x1)) * one_over_6_vol
            d1 = (x2 * (y3 - y4) + x3 * (y4 - y2) + x4 * (y2 - y3)) * one_over_6_vol
            d2 = (x1 * (y4 - y3) + x3 * (y1 - y4) + x4 * (y3 - y1)) * one_over_6_vol
            d3 = (x1 * (y2 - y4) + x2 * (y4 - y1) + x4 * (y1 - y2)) * one_over_6_vol
            d4 = (x1 * (y3 - y2) + x2 * (y1 - y3) + x3 * (y2 - y1)) * one_over_6_vol
            Bmat = np.array([[b1, 0, 0, b2, 0, 0, b3, 0, 0, b4, 0, 0],
                             [0, c1, 0, 0, c2, 0, 0, c3, 0, 0, c4, 0], 
                             [0, 0, d1, 0, 0, d2, 0, 0, d3, 0, 0, d4],
                             [c1, b1, 0, c2, b2, 0, c3, b3, 0, c4, b4, 0],
                             [0, d1, c1, 0, d2, c2, 0, d3, c3, 0, d4, c4],
                             [d1, 0, b1, d2, 0, b2, d3, 0, b3, d4, 0, b4]])
            self.Bmat_list.append(Bmat)
            self.stiffness_matrix_list.append(np.dot(np.dot(Bmat.T, self.Dmat), Bmat) * self.vol_list[i])

    def add_ws_points(self, ws_idx):
        self.ee_idxs = ws_idx
        self.n_ee = len(self.ee_idxs)

    def add_ee_pos(self, ee_position):
        self.n_ee = ee_position.shape[0]
        self.ee_idxs = [0 for _ in range(self.n_ee)]
        for i in range(self.n_ee):
            self.ee_idxs[i] = np.argmin(np.linalg.norm(self.cur_vertices - ee_position[i], axis=1))

    def draw_cur_mesh(self):
        plotter = pv.Plotter()
        mesh = pv.PolyData(self.cur_vertices)
        mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])

        # plot the cable from pp to pulley
        for i in range(self.nCable):
            plotter.add_lines(np.array([self.pulley_location[i], self.cur_vertices[self.pp_idx[i]]]), color='black')

        # add fixed idx
        for i in self.fixed_idx:
            plotter.add_mesh(pv.Sphere(radius=0.001, center=self.cur_vertices[i]), color='blue')

        # plot contact vertices
        for contact_idx in self.contact_vertices:
            plotter.add_mesh(pv.Sphere(radius=0.001, center=self.cur_vertices[contact_idx]), color='green')

        plotter.add_mesh(mesh, show_edges=True)

        # draw ee points
        # for i in range(self.n_ee):
        #     plotter.add_mesh(pv.Sphere(radius=0.002, center=self.cur_vertices[self.ee_idxs[i]]), color='red')

        # draw pp points
        for i in range(self.nCable):
            plotter.add_mesh(pv.Sphere(radius=0.002, center=self.cur_vertices[self.pp_idx[i]]), color='blue')

        # annotate pp points
        # for i in range(self.nCable):
        #     plotter.add_mesh(pv.Sphere(radius=0.002, center=self.pp_location[i]), color='blue')
            
        plotter.show_grid()
        plotter.show_axes()
        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.show()
        return

    def draw_cur_mesh_w_force(self, force_vec):
        plotter = pv.Plotter()
        mesh = pv.PolyData(self.cur_vertices)
        mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])

        # plot the cable from pp to pulley
        for i in range(self.nCable):
            plotter.add_lines(np.array([self.pulley_location[i], self.cur_vertices[self.pp_idx[i]]]), color='red')

        # add fixed idx
        for i in self.fixed_idx:
            plotter.add_mesh(pv.Sphere(radius=0.001, center=self.cur_vertices[i]), color='blue')

        plotter.add_mesh(mesh, show_edges=True)
        plotter.show_grid()
        plotter.show_axes()
        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.add_arrows(self.cur_vertices, self.cur_vertices + 1e-4*force_vec)
        plotter.show()
        return

    def draw_cur_mesh_w_initial_mesh(self):
        plotter = pv.Plotter()
        mesh = pv.PolyData(self.cur_vertices)
        initial_mesh = pv.PolyData(self.vertices)
        mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])
        initial_mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])
        # plot the cable from pp to pulley
        for i in range(self.nCable):
            plotter.add_lines(np.array([self.pulley_location[i], self.cur_vertices[self.pp_idx[i]]]), color='black')

        # add fixed idx
        for i in self.fixed_idx:
            plotter.add_mesh(pv.Sphere(radius=0.001, center=self.cur_vertices[i]), color='blue')

        plotter.add_mesh(mesh, show_edges=True)
        plotter.add_mesh(initial_mesh, show_edges=True, color='black', opacity=0.5)
        plotter.show_grid()
        plotter.show_axes()
        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.add_mesh(mesh, show_edges=True, color='red')
        plotter.show()
        return

    def draw_three_meshes(self, vert_cg, vert_fkLength, vert_force):
        plotter = pv.Plotter()
        mesh_cg = pv.PolyData(vert_cg)
        mesh_cg.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])   
        mesh_fkLength = pv.PolyData(vert_fkLength)
        mesh_fkLength.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])
        mesh_force = pv.PolyData(vert_force)
        mesh_force.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])
        plotter.add_mesh(mesh_cg, show_edges=True, color='red')
        plotter.add_mesh(mesh_fkLength, show_edges=True, color='green')
        plotter.add_mesh(mesh_force, show_edges=True, color='blue')
        for i in range(self.nCable):
            plotter.add_lines(np.array([self.pulley_location[i], self.cur_vertices[self.pp_idx[i]]]), color='red')

        # add fixed idx
        for i in self.fixed_idx:
            plotter.add_mesh(pv.Sphere(radius=0.001, center=self.cur_vertices[i]), color='blue')
        plotter.show_grid()
        plotter.show_axes()
        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.show()

    def draw_mesh_ik(self, vert_cg, ee_target):
        plotter = pv.Plotter()
        mesh = pv.PolyData(vert_cg)
        mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])
        plotter.add_mesh(mesh, show_edges=True,opacity=0.5)
        for i in range(self.nCable):
            plotter.add_lines(np.array([self.pulley_location[i], vert_cg[self.pp_idx[i]]]), color='green')
        for i in range(self.n_ee):
            plotter.add_mesh(pv.Sphere(radius=0.003, center=vert_cg[self.ee_idxs[i]]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.003, center=ee_target[i]), color='blue')
        # plotter.add_mesh(pv.Sphere(radius=0.003, center=vert_cg[self.ws_idx[0]]), color='red')
        # plotter.add_mesh(pv.Sphere(radius=0.003, center=ee_target[0]), color='blue')
        plotter.show_grid()
        plotter.show_axes()
        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.show()
        
    def draw_3d_w_idx(self): # draw initial tet with plt and label each vertex with idx
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(self.num_vert):
            ax.scatter(self.vertices[i][0], self.vertices[i][1], self.vertices[i][2], color='blue')
            # ax.text(self.vertices[i][0], self.vertices[i][1], self.vertices[i][2], str(i), color='black')
        # annotate ws points
        for i in range(self.n_ws):
            ax.scatter(self.vertices[self.ws_idx[i]][0], self.vertices[self.ws_idx[i]][1], self.vertices[self.ws_idx[i]][2], color='red')
            ax.text(self.vertices[self.ws_idx[i]][0], self.vertices[self.ws_idx[i]][1], self.vertices[self.ws_idx[i]][2], str(self.ws_idx[i]), color='black')

        
        for i in range(self.nCable): # draw the line from pp to pulley
            ax.plot([self.pp_location[i][0], self.pulley_location[i][0]], [self.pp_location[i][1], self.pulley_location[i][1]], [self.pp_location[i][2], self.pulley_location[i][2]], color='red')
        plt.show()
        return

    def draw_traj(self, traj_list):
        plotter = pv.Plotter()
        mesh = pv.PolyData(self.cur_vertices)
        mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])

        # plot the cable from pp to pulley
        for i in range(self.nCable):
            plotter.add_lines(np.array([self.pulley_location[i], self.cur_vertices[self.pp_idx[i]]]), color='green')

        # add fixed idx
        for i in self.fixed_idx:
            plotter.add_mesh(pv.Sphere(radius=0.001, center=self.cur_vertices[i]), color='blue')

        # each traj is an ee position
        for i in range(self.n_ws):
            plotter.add_mesh(pv.Sphere(radius=0.005, center=self.cur_vertices[self.ws_idx[i]]), color='red')
            for j in range(len(traj_list)-1):
                plotter.add_lines(np.array([traj_list[j][i], traj_list[j+1][i]]), color='blue')

        plotter.add_mesh(mesh, show_edges=True)
        plotter.show_grid()
        plotter.show_axes()
        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.show()
        return

    def draw_ws(self, ws_file):
        plotter = pv.Plotter()
        mesh = pv.PolyData(self.cur_vertices)
        mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])

        # plot the cable from pp to pulley
        for i in range(self.nCable):
            plotter.add_lines(np.array([self.pulley_location[i], self.cur_vertices[self.pp_idx[i]]]), color='black')

        # add fixed idx
        for i in self.fixed_idx:
            plotter.add_mesh(pv.Sphere(radius=0.001, center=self.cur_vertices[i]), color='blue')


        plotter.add_mesh(pv.Sphere(radius=0.003, center=self.cur_vertices[self.ws_idx[0]]), color='red')
        plotter.add_mesh(pv.Sphere(radius=0.003, center=self.cur_vertices[self.ws_idx[1]]), color='blue')
        # ws_file has three parts separated by ','
        # thrid part is the ws positions in the form of np array
        ws_1 = []
        ws_2 = []
        with open(ws_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    line_list = line.split(',')
                    if len(line_list) == 6:
                        ws_1_single = line_list[-1]
                        ws_1_single = ws_1_single[3:-1]
                        pos_1 = ws_1_single.split()
                        pos_1 = [float(pos) for pos in pos_1]
                        plotter.add_mesh(pv.Sphere(radius=0.001, center=pos_1), color='red')
                        # print(pos_1)
                        ws_1.append(pos_1)
                    else:
                        ws_2_single = line_list[-1]
                        ws_2_single = ws_2_single[2:-2]
                        pos_2 = ws_2_single.split()
                        pos_2 = [float(pos) for pos in pos_2]
                        plotter.add_mesh(pv.Sphere(radius=0.001, center=pos_2), color='blue')
                        # print(pos_2)
                        ws_2.append(pos_2)
        
                

        plotter.add_mesh(mesh, show_edges=True)
        plotter.show_grid()
        plotter.show_axes()
        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.show()
        return

    def draw_traj_ikRef(self, vert_lists, traj_list):
        # use 4 subplot to draw the traj
        nTraj = 4
        plotter_list = [pv.Plotter() for _ in range(nTraj)]
        for i in range(nTraj):
            mesh = pv.PolyData(vert_lists[i])
            mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])
            plotter_list[i].add_mesh(mesh, show_edges=True)
            for j in range(self.nCable):
                plotter_list[i].add_lines(np.array([self.pulley_location[j], vert_lists[i][self.pp_idx[j]]]), color='green')
            for j in range(self.n_ws):
                plotter_list[i].add_mesh(pv.Sphere(radius=0.001, center=vert_lists[i][self.ws_idx[j]]), color='red')
                plotter_list[i].add_mesh(pv.Sphere(radius=0.001, center=traj_list[i][j]), color='blue')
            plotter_list[i].show_grid()
            plotter_list[i].show_axes()
            # make axis equal
            plotter_list[i].set_scale(1, 1, 1)
            plotter_list[i].show()

    def draw_cur_mesh_and_touchRegion(self, point_cloud, fitted_surface, contact_force, grid_size=20):
        plotter = pv.Plotter()
        mesh = pv.PolyData(self.cur_vertices)
        mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])

        # plot the cable from pp to pulley
        for i in range(self.nCable):
            plotter.add_lines(np.array([self.pulley_location[i], self.cur_vertices[self.pp_idx[i]]]), color='black')

        # add fixed idx
        for i in self.fixed_idx:
            plotter.add_mesh(pv.Sphere(radius=0.001, center=self.cur_vertices[i]), color='blue')
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
        # plot the surface in pyvista
        surf = pv.StructuredGrid(X_grid, Y_fitted, Z_grid)
        # draw the point cloud
        # cloud = pv.PolyData(point_cloud)
        # plotter.add_mesh(cloud, color='blue', point_size=5)
        plotter.add_mesh(surf, color='red', opacity=0.5)
        plotter.add_mesh(mesh, show_edges=True)
        # plot contact force as arrows
        for i in range(self.num_vert):
            force_vec = contact_force[3*i:3*i+3].reshape((1,3))
            arrow_start = self.cur_vertices[i].reshape(1,3)
            # if np.linalg.norm(force_vec) > 1e-6:
            #     # print("force is: ", np.linalg.norm(force_vec))
            #     plotter.add_arrows(arrow_start, 5e-2*force_vec, color='yellow')
            plotter.add_arrows(arrow_start, 1e-2*force_vec, color='yellow')
        # plot all point cloud
        # for i in range(point_cloud.shape[0]):
        #     plotter.add_mesh(pv.Sphere(radius=0.001, center=point_cloud[i]), color='blue')

        plotter.show_grid()
        plotter.show_axes()
        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.show()
        return

    def draw_cur_mesh_touchRegion_noContactConfig(self, point_cloud, fitted_surface, vert_nocontact, grid_size=20):
        plotter = pv.Plotter()
        mesh = pv.PolyData(self.cur_vertices)
        mesh_nocontact = pv.PolyData(vert_nocontact)
        mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])
        mesh_nocontact.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])
        # plot the cable from pp to pulley
        for i in range(self.nCable):
            plotter.add_lines(np.array([self.pulley_location[i], self.cur_vertices[self.pp_idx[i]]]), color='black')

        # add fixed idx
        for i in self.fixed_idx:
            plotter.add_mesh(pv.Sphere(radius=0.001, center=self.cur_vertices[i]), color='blue')
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
        X_grid = U * (x.max() - x.min()) + x.min()
        Z_grid = V * (z.max() - z.min()) + z.min()
        X_grid = X_grid.T
        Z_grid = Z_grid.T
        # plot the surface in pyvista
        surf = pv.StructuredGrid(X_grid, Y_fitted, Z_grid)
        plotter.add_mesh(surf, color='red', opacity=0.5)
        plotter.add_mesh(mesh, show_edges=True)
        plotter.add_mesh(mesh_nocontact, show_edges=True, color='green')
        # add legend
        # plotter.add_text("Red: Contact config", position='upper_left', font_size=10, color='red')
        plotter.add_text("Green: No Contact Config", position='upper_left', font_size=10, color='green')

        plotter.show_grid()
        plotter.show_axes()
        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.show()
        return

    def draw_mesh_w_intersection(self, inter_points, mesh2draw):
        plotter = pv.Plotter()
        mesh = pv.PolyData(mesh2draw)
        mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])
        plotter.add_mesh(mesh, show_edges=True, color='blue', opacity=0.5)  
        # plot the cable from pp to pulley
        for i in range(self.nCable):
            plotter.add_lines(np.array([self.pulley_location[i], mesh2draw[self.pp_idx[i]]]), color='black')

        # add fixed idx
        for i in self.fixed_idx:
            plotter.add_mesh(pv.Sphere(radius=0.001, center=self.vertices[i]), color='blue')

        # draw intersection points
        for i in range(inter_points.shape[0]):
            plotter.add_mesh(pv.Sphere(radius=0.01, center=inter_points[i]), color='red')

        # draw cables
        for i in range(self.nCable):
            plotter.add_lines(np.array([self.pulley_location[i], mesh2draw[self.pp_idx[i]]]), color='green')

        plotter.show_grid()
        plotter.show_axes()
        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.show()

if __name__ == "__main__":
    # pp_location = np.array([[0.15, 0.0, 0.0], [0.15, 0.0, 0.1], [0.10, 0.005, 0.05]])
    # pulley_location = np.array([[0., -0.18, 0], [0., -0.15, 0.1], [0., 0.18, 0.0]])
    # cds = cableDrivenSurface("models/flat_surface/flat_surface.tet", pp_location, pulley_location)
    # # cds.draw_cur_mesh()
    # cds.draw_3d_w_idx()

    # this is size2 palm
    pp_location = np.array([[57.76, 9.89, 4.5], [52.66, 47.39, 8], [52.66, 47.39, 57], [57.61,  9.89, 60.5]])*1e-3
    pulley_location = np.array([[60.93, -117.64, 2.5], [100.93, -117.64, 2.5],[100.93, -117.64, 62.5], [60.93, -117.64, 62.5]]) *1e-3

    # this is size3 palm
    # pp_location = np.array([[57.61, 8.96, 6], [65.12, 46.49, 8.00], [65.12, 46.49, 57], [57.61, 8.96, 60]])*1e-3
    # pulley_location = np.array([[60.93, -117.64, 2.5], [100.93, -117.64, 2.5],[100.93, -117.64, 62.5], [60.93, -117.64, 62.5]]) *1e-3

    # pp_location = np.array([[0.05352376, -0.0127263,   0. ], [0.07471047, 0.02281154, 0.0073504], [0.07471047, 0.02281154, 0.0576496], [0.05352376, -0.0127263 ,  0.065 ]])
    # pulley_location = np.array([[0.07, -0.10, 0],[0.13, -0.10, 0],[0.13, -0.10, 0.07], [0.07, -0.10, 0.07]])
    # cds = cableDrivenSurface("models/palm_size2/palm_size2.tet", pp_location, pulley_location)
    cds = cableDrivenSurface("models/palm_size2/palm_size2.tet", pp_location, pulley_location)
    cds.draw_cur_mesh()

    # cds.draw_cur_mesh_and_touchRegion(points, B_spline_surface)
    # traj_list = [np.array([[0.14, -0.010,  0.010],[0.135, -0.010, 0.086]]), 
    #              np.array([[0.12, -0.010,  0.010],[0.125, 0.0, 0.086]]),
    #              np.array([[0.12, 0.0,  0.010],[0.145, 0.000, 0.086]]),
    #              np.array([[0.14, 0.0,  0.010],[0.135, -0.010, 0.086]])]
    # cds.draw_traj(traj_list)
    # cds.draw_ws("flat_surface_4cable_ws.txt")


        
        
        
                
                        
                

        




        
        