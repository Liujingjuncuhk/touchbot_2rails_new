import numpy as np
import pyvista as pv
from utilities import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import numpy as np
import pyvista as pv
from scipy.linalg import ldl, solve_triangular
from utilities import * 
# from lemkelcp import lemketableau
from qpsolvers import solve_qp

from scipy.optimize import minimize
import time
import quantecon as qe
import lemkelcp as lcp
import cv2
import os
from pathlib import Path

class PalmSimulator:
    def __init__(self, pickleFilename: str,initial_rotation = 15.0):
        self.N44 = np.eye(4) - 1/4*np.ones((4, 4))
        self.N1212 = np.zeros((12, 12))
        for i in range(4):
            for j in range(3):
                for k in range(4):
                    if k==i:
                        self.N1212[3*i+j, 3*k+j] = 3.0/4.0
                    else:
                        self.N1212[3*i+j, 3*k+j] = -1.0/4.0
        self.N22 = np.eye(2) - 1/2*np.ones((2, 2))
        self.N66 = np.zeros((6, 6))
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    if k==i:
                        self.N66[3*i+j, 3*k+j] = 1.0/2.0
                    else:
                        self.N66[3*i+j, 3*k+j] = -1.0/2.0
        self.initial_rotation = initial_rotation/ 180 * np.pi  # convert to radians
        self.Rz = np.array([[np.cos(self.initial_rotation), np.sin(self.initial_rotation), 0], [-np.sin(self.initial_rotation), np.cos(self.initial_rotation), 0], [0, 0, 1]])
        with open(pickleFilename, 'rb') as f:
            data = pickle.load(f)
            self.vertices = data['vertices']
            self.tetrahedra = data['tetrahedra']
            self.pp_idx = data['pp_idx']
            self.pulley_location = data['pulley_location']
        self.num_vertices = len(self.vertices)
        self.num_tetrahedra = len(self.tetrahedra)
        self.fixed_region = [[-0.001, 0.001],[-0.01, 0.20],[-0.01, 0.30]]
        self.set_fixed_region(self.fixed_region)
        self.add_contact_vertices()
        self.triangle_list = []
        for i in range(self.num_tetrahedra):
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
        self.nCable = len(self.pp_idx)
        self.original_verts = self.vertices.copy()
        for i in range(self.num_vertices):
            self.vertices[i] = np.dot(self.Rz, self.vertices[i])
        self.cur_vertices = self.vertices.copy()

        self.vol_list = [0 for _ in range(self.num_tetrahedra)]
        for i in range(self.num_tetrahedra):
            self.vol_list[i] = cal_vol_tet(self.vertices[self.tetrahedra[i]])
        self.weight_list = self.vol_list.copy()
        self.weight_cable = max(self.weight_list) * 5
        self.add_material_properties()
        self.mass_mat = np.zeros((3*self.num_vertices, 3*self.num_vertices))
        self.W_mat = np.zeros((3*self.num_vertices, 3*self.num_vertices))
        self.gravity_vec = np.zeros((3*self.num_vertices, 1))
        for i in range(self.num_tetrahedra):
            mass_tet = self.weight_list[i] * self.density 
            for j in range(4):
                idx = self.tetrahedra[i][j]
                self.gravity_vec[3*idx + 1] += mass_tet * 9.81 / 4
                for k in range(3):
                    self.mass_mat[3*idx + k, 3*idx + k] += mass_tet / 4
        
        for i in range(self.num_vertices):
            if self.idxAll_2_idxMoving[i] != -1:
                for j in range(3):
                    self.W_mat[3*i + j, 3*i + j] = 1.0/self.mass_mat[3*i + j, 3*i + j]
            else:
                for j in range(3):
                    self.W_mat[3*i + j, 3*i + j] = 0

        self.original_tet_sk = []
        for i in range(self.num_tetrahedra):
            tet = self.tetrahedra[i]
            tet_vert = self.vertices[tet]
            tet_sk = self.N44 @ tet_vert
            self.original_tet_sk.append(tet_sk)
        self.original_cable_sk = []
        for i in range(self.nCable):
            cur_cable = np.zeros((2, 3))
            idx_c = self.pp_idx[i]
            cur_cable[0, :] = self.vertices[idx_c, :]
            cur_cable[1, :] = self.pulley_location[i, :]
            cur_cable_sk = self.N22 @ cur_cable
            self.original_cable_sk.append(cur_cable_sk)
        self.initial_cable_length = []
        for i in range(self.nCable):
            self.initial_cable_length.append(np.linalg.norm(self.vertices[self.pp_idx[i]] - self.pulley_location[i]))
        self.Q0 = np.zeros((3*self.num_vertices, 1))
        for i in range(self.num_vertices):
            self.Q0[3*i:3*i+3, 0] = self.vertices[i]
        print("initial pp location:")
        for i in range(self.nCable):
            print(f"Pullpoint {i+1}: {self.vertices[self.pp_idx[i]]}")
        print("pulley location:")
        for i in range(self.nCable):
            print(f"Pulley {i+1}: {self.pulley_location[i]}")
        print("Initial cable length: ", self.initial_cable_length)
        
        print("Number of tets: ", self.num_tetrahedra)
        print("Number of vertices: ", self.num_vertices)
        self.assemble_cg_matrices()
        

    def add_contact_vertices(self):
        func = myPalm_size3
        self.contact_vertex_idx = []
        for i in range(self.num_vertices):
            q = self.vertices[i]
            if func(q) < 0:
                self.contact_vertex_idx.append(i)
        self.num_contact_vertices = len(self.contact_vertex_idx)
        self.idxAll2_idxContact = [-1 for _ in range(self.num_vertices)]
        self.idxContact2_idxAll = []
        idx_contact = 0
        for i in range(self.num_vertices):
            if i in self.contact_vertex_idx:
                self.idxAll2_idxContact[i] = idx_contact
                self.idxContact2_idxAll.append(i)
                idx_contact += 1
        
    def set_fixed_region(self, fixed_region):
        self.fixed_region = fixed_region
        fixed_idx = []
        for i in range(self.num_vertices):
            if fixed_region[0][0] <= self.vertices[i][0] <= fixed_region[0][1] and \
                fixed_region[1][0] <= self.vertices[i][1] <= fixed_region[1][1] and \
                fixed_region[2][0] <= self.vertices[i][2] <= fixed_region[2][1]:
                fixed_idx.append(i)
        self.fixed_idx = fixed_idx
        self.nMoving = self.num_vertices - len(fixed_idx)
        self.idxAll_2_idxMoving = [-1 for _ in range(self.num_vertices)]
        self.idxMoving_2_idxAll = []
        idx_moving = 0
        for i in range(self.num_vertices):
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
        for i in range(self.num_tetrahedra):
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

    def get_cable_length(self, vertices):
        """
        Calculate the cable length based on the current vertices.
        """
        cable_length = self.initial_cable_length.copy()
        for i in range(self.nCable):
            idx_pp = self.pp_idx[i]
            cable_length[i] = np.linalg.norm(self.pulley_location[i] - vertices[idx_pp])
        return cable_length

    def construct_arm_sdf(self):
        self.seg_r_list = [[0 for _ in range(self.theta_seg)] for _ in range(self.nSeg)]
        self.seg_r_number = [[0 for _ in range(self.theta_seg)] for _ in range(self.nSeg)]
        for i in range(self.nSeg):
            pointCloud_seg = self.pointCloud_seglist[i]
            normalvec_seg = self.normalvec_seglist[i]
            npoint = pointCloud_seg.shape[0]
            center_pts = self.center_line[i]
            for j in range(npoint):
                point = pointCloud_seg[j]
                dist = np.linalg.norm(point - center_pts)
                dx = point[0] - center_pts[0]
                dy = point[1] - center_pts[1]
                theta = int((np.arctan2(dy, dx)+np.pi)/self.theta_unit+0.5)
                self.seg_r_list[i][theta] += dist
                self.seg_r_number[i][theta] += 1
        for i in range(self.nSeg):
            for j in range(self.theta_seg):
                if self.seg_r_number[i][j] > 0:
                    self.seg_r_list[i][j] /= self.seg_r_number[i][j]
                else:
                    self.seg_r_list[i][j] = 0.0
        

    def sdf_func(self, query_point):
        z = query_point[2]
        if z < self.minz or z > self.minz + self.nSeg * self.seg_length:
            return 1.0  # outside the arm region
        seg_idx = int((z - self.minz) / self.seg_length)
        center_seg = self.center_line[seg_idx]
        
        if seg_idx < 0:
            seg_idx = 0
        if seg_idx >= self.nSeg:
            seg_idx = self.nSeg - 1
        if z > center_seg[2] and seg_idx < self.nSeg - 1:
            center_pts_next = self.center_line[seg_idx + 1]
            ratio = (z - center_seg[2]) / (center_pts_next[2] - center_seg[2])
            center_pts = center_seg * (1 - ratio) + center_pts_next * ratio
            # center_radius =  self.seg_r_list[seg_idx][theta]*(1 - ratio) + self.seg_r_list[seg_idx + 1][theta]* ratio
        elif z < center_seg[2] and seg_idx > 0:
            center_pts_prev = self.center_line[seg_idx - 1]
            ratio = (center_seg[2] - z) / (center_seg[2] - center_pts_prev[2])
            center_pts = center_seg * (1 - ratio) + center_pts_prev * ratio
            # center_radius =  self.seg_r_list[seg_idx][theta]*(1 - ratio) + self.seg_r_list[seg_idx - 1][theta]* ratio
        else:
            center_pts = center_seg
            ratio = 0.0
            center_radius = self.seg_r_list[seg_idx][theta]
        dx = query_point[0] - center_pts[0]
        dy = query_point[1] - center_pts[1]
        theta = int((np.arctan2(dy, dx)+np.pi)/self.theta_unit+0.5)
        # define r_arm
        if z > center_seg[2] and seg_idx < self.nSeg - 1:
            r_arm = self.seg_r_list[seg_idx][theta]*(1 - ratio) + self.seg_r_list[seg_idx + 1][theta]* ratio
        elif z < center_seg[2] and seg_idx > 0:
            r_arm = self.seg_r_list[seg_idx][theta]*(1 - ratio) + self.seg_r_list[seg_idx - 1][theta]* ratio
        else:
            r_arm = self.seg_r_list[seg_idx][theta]
        unit_vec = np.array([dx, dy, query_point[2] - center_pts[2]])
        unit_vec = unit_vec / np.linalg.norm(unit_vec) 
        r_query = np.linalg.norm(np.array([dx, dy]))    
        sdf_value = r_query - r_arm
        is_contact = 0
        if sdf_value < 0:
            is_contact = 1
        return is_contact, sdf_value, unit_vec
                
    def sdf_func_Q(self, Q):
        is_contact_list = []
        sdf_value_list = []
        unit_vec_list = []
        for i in range(self.num_vertices):
            query_point = Q[3*i:3*i+3, 0]
            is_contact, sdf_value, unit_vec = self.sdf_func(query_point)
            is_contact_list.append(is_contact)
            sdf_value_list.append(sdf_value)
            unit_vec_list.append(unit_vec)
        return is_contact_list, sdf_value_list, unit_vec_list
        
        
    
# cg part 
    def assemble_cg_matrices(self):
        self.matAcg_original = np.zeros((12*self.num_tetrahedra + 3*self.nCable, 3*self.num_vertices))
        self.matAcg = np.zeros((12*self.num_tetrahedra + 3*self.nCable, 3*self.nMoving))
        self.vecbcg = np.zeros((12*self.num_tetrahedra + 3*self.nCable, 1))
        self.vecbcg_2add = np.zeros((12*self.num_tetrahedra + 3*self.nCable, 1))
        self.unit_cable_vec = np.zeros((self.nCable, 3))
        for i in range(self.nCable):
            idx_pp = self.pp_idx[i]
            self.unit_cable_vec[i, :] = (self.vertices[idx_pp] - self.pulley_location[i] ) / np.linalg.norm(self.pulley_location[i] - self.vertices[idx_pp])
        for i in range(self.num_tetrahedra):
            tet = self.tetrahedra[i]
            for j in range(4):
                idxj = tet[j]
                for k in range(4):
                    idxk = tet[k]
                    self.matAcg_original[12*i + 3*j:12*i + 3*j + 3, 3*idxk:3*idxk + 3] += self.N1212[3*j:3*j+3, 3*k:3*k+3]*self.weight_list[i]
        for i in range(self.nCable):
            idx_pp = self.pp_idx[i]
            for k in range(3):
                self.matAcg_original[12*self.num_tetrahedra + 3*i + k, 3*idx_pp + k] = 1.0/2.0 * self.weight_cable

        for i in range(self.matAcg_original.shape[0]):
            for j in range(self.num_vertices):
                if self.idxAll_2_idxMoving[j] != -1:
                    idx_moving = self.idxAll_2_idxMoving[j]
                    for k in range(3):
                        self.matAcg[i, 3*idx_moving + k] = self.matAcg_original[i, 3*j + k]
                else:
                    for k in range(3):
                        self.vecbcg_2add[i, 0] -= self.matAcg_original[i, 3*j + k] * self.vertices[j, k]
    
        for i in range(self.nCable):
            idx_pp = self.pp_idx[i]
            self.vecbcg_2add[12*self.num_tetrahedra + 3*i:12*self.num_tetrahedra + 3*i + 3, 0] = 0.5 * self.pulley_location[i] * self.weight_cable
        
        self.matATAcg = self.matAcg.T @ self.matAcg
        self.matATAcg_inv = np.linalg.pinv(self.matATAcg)
        self.matATAcg_inv_matAcgT = self.matATAcg_inv @ self.matAcg.T

    def assemble_bveccg(self, R_list_tet, R_list_cable, target_cable_length):
        """
        Assemble the b vector for the CG method.
        """
        self.vecbcg.fill(0)
        for i in range(self.num_tetrahedra):
            tet = self.tetrahedra[i]
            q0 = np.zeros((12, 1))
            for j in range(4):
                idx = tet[j]
                q0[3*j:3*j+3, 0] = self.vertices[idx, :]
            tar_sk = R_list_tet[i] @self.N1212 @  q0 *self.weight_list[i]
            for j in range(4):
                idx = tet[j]
                self.vecbcg[12*i + 3*j:12*i + 3*j + 3, 0] += tar_sk[3*j:3*j+3, 0]
        for i in range(self.nCable):
            idx_pp = self.pp_idx[i]
            unit_vec = (self.vertices[idx_pp] - self.pulley_location[i]) / np.linalg.norm(self.pulley_location[i] - self.vertices[idx_pp])
            q0 = np.zeros((6, 1))  
            
            q0[0:3, 0] = self.pulley_location[i] + target_cable_length[i] * unit_vec
            q0[3:6, 0] = self.pulley_location[i]
            tar_sk = R_list_cable[i] @ self.N66 @ q0 * self.weight_cable
            for k in range(3):
                self.vecbcg[12*self.num_tetrahedra + 3*i + k, 0] += tar_sk[k, 0]
        self.vecbcg += self.vecbcg_2add

    def deform_cg(self, target_cable_length, starting_Q):
        """
        Deform the robot based on the target cable length.
        This function is a placeholder and should be implemented based on the specific requirements.
        """
        # Implement the deformation logic here
        
        Q = starting_Q.copy()
        Q_last = starting_Q.copy()
        tol = 1e-6
        max_iter = 1000
        for i in range(max_iter):
            R_list_tet = self.cal_rotation_fkLength(Q)
            R_list_cable = self.cal_rotation_cable(Q)  # Assuming similar rotation for cables
            self.assemble_bveccg(R_list_tet, R_list_cable, target_cable_length)
            # Atb = self.matAcg.T @ self.vecbcg
            # Q_moving = self.matATAcg_inv @ Atb
            Q_moving = self.matATAcg_inv_matAcgT @ self.vecbcg
            Q = self.Qmoving_2_Q(Q_moving)
            diff = np.linalg.norm(Q - Q_last)/self.num_vertices/3
            # print(f"Iteration {i}, diff: {diff}")
            if diff < tol:
                # print(f"Converged in {i} iterations.")
                break
            Q_last = Q.copy()
        verts = self.Q_2_vertices(Q)
        print("target cable length: ", target_cable_length)
        print("final cable length: ", self.get_cable_length(Q, self.pulley_location))
        return verts, Q

    def cal_rotation_cable(self, Q):
        R_list = [np.zeros((6, 6)) for _ in range(self.nCable)]
        for i in range(self.nCable):
            cur_cable = np.zeros((2, 3))
            
            idx = self.pp_idx[i]
            cur_cable[0, :] = Q[3*idx:3*idx+3].reshape((3,))
            cur_cable[1, :] = self.pulley_location[i, :]
            cur_cable_sk = self.N22 @ cur_cable
            u, s, vh = np.linalg.svd(cur_cable_sk.T @ self.original_cable_sk[i])
            R_this = u @ vh
            for j in range(2):
                R_list[i][3*j:3*j+3, 3*j:3*j+3] = R_this
        return R_list

    def cal_jacobian_cg(self, Q, pulley_location_a):
        R_list_cable = self.cal_rotation_cable(Q)
        vecbv = np.zeros((12*self.num_tetrahedra + 3*self.nCable, 1))
        H_c = np.zeros((12*self.num_tetrahedra + 3*self.nCable, self.nCable))
        unit_cable_vec = np.zeros((self.nCable, 3)) 
        for i in range(self.nCable):
            idx_pp = self.pp_idx[i]
            unit_cable_vec[i, :] = (self.vertices[idx_pp] - pulley_location_a[i]) / np.linalg.norm(pulley_location_a[i] - self.vertices[idx_pp])
        for i in range(self.nCable):
            R = R_list_cable[i]
            unit_vec = unit_cable_vec[i, :]
            for j in range(3):
                H_c[12*self.num_tetrahedra + 3*i + j, i] = 0.5 * R[0:3, 0:3].dot(unit_vec)[j] * self.weight_cable
        Jac = self.matATAcg_inv_matAcgT @ H_c
        # print("Jac cg shape: ", Jac.shape)
        return Jac.T


# FEM part
    def assemble_K(self, Ke_list):
        K = np.zeros((3*self.num_vertices, 3*self.num_vertices))
        for i in range(self.num_tetrahedra):
            tet = self.tetrahedra[i]
            for j in range(4):
                idxj = tet[j]
                for k in range(4):
                    idxk = tet[k]
                    K[3*idxj:3*idxj+3, 3*idxk:3*idxk+3] += Ke_list[i][3*j:3*j+3, 3*k:3*k+3]
        return K
    
    def assemble_f0(self, Ke0_list):
        f0 = np.zeros((3*self.num_vertices, 1))
        q0e = np.zeros((12, 1))
        for i in range(self.num_tetrahedra):
            q0e = np.zeros((12, 1))
            for j in range(4):
                idx = self.tetrahedra[i][j]
                q0e[3*j:3*j+3, 0] = self.vertices[idx,:]
            f0e = Ke0_list[i] @ q0e
            for j in range(4):
                idx = self.tetrahedra[i][j]
                f0[3*idx:3*idx+3, 0] += f0e[3*j:3*j+3, 0]
        return f0
    
    def cal_rotation_fkLength(self, Q):
        R_list = [np.zeros((12, 12)) for _ in range(self.num_tetrahedra + self.nCable)]
        for i in range(self.num_tetrahedra):
            cur_tet = np.zeros((4, 3))
            for j in range(4):
                idx = self.tetrahedra[i][j]
                cur_tet[j, :] = Q[3*idx:3*idx+3].reshape((3,))
            cur_tet_sk = self.N44 @ cur_tet
            u, s, vh = np.linalg.svd(cur_tet_sk.T @ self.original_tet_sk[i])
            R_this = u @ vh
            for j in range(4):
                R_list[i][3*j:3*j+3, 3*j:3*j+3] = R_this
        return R_list

    def cal_Ke_lists(self, R_list_1212):
        Ke_list = [np.zeros((12, 12)) for _ in range(self.num_tetrahedra)]
        Ke0_list = [np.zeros((12, 12)) for _ in range(self.num_tetrahedra)]
        for i in range(self.num_tetrahedra):
            Ke_list[i] = R_list_1212[i] @ self.stiffness_matrix_list[i] @ R_list_1212[i].T @ self.N1212
            Ke0_list[i] = R_list_1212[i] @ self.stiffness_matrix_list[i] @ self.N1212
        return Ke_list, Ke0_list    

    def get_cable_length(self, starting_Q, pulley_location):
        """
        Calculate the cable length based on the current vertices.
        """
        cable_length = self.initial_cable_length.copy()
        for i in range(self.nCable):
            idx_pp = self.pp_idx[i]
            cable_length[i] = np.linalg.norm(pulley_location[i] - starting_Q[3*idx_pp:3*idx_pp+3].reshape((3,)))
        return cable_length

# FDK part
    def FKD_static(self, target_cable_length, starting_Q, starting_Qd):
        Q_a = starting_Q.copy()
        Q_ad = starting_Qd.copy()
        Q_a_last = Q_a.copy()
        t_a = 0.0
        h = 0.02
        total_time = 1.0
        phi_Qfree = np.zeros((self.nCable, 1))
        H_free = np.zeros((self.nCable, 3*self.num_vertices))
        Q_list = [Q_a.copy()]
        starting_cable_length = self.get_cable_length(Q_a, self.pulley_location)
        while t_a < total_time:
            Q_a_last = Q_a.copy()
            R_list_1212 = self.cal_rotation_fkLength(Q_a)
            Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
            K_mat = self.assemble_K(Ke_list)
            f0 = self.assemble_f0(Ke0_list)
            A_mat = (1.0/h)*np.eye(3*self.num_vertices) +h *  self.W_mat @ K_mat
            A_inv = np.linalg.inv(A_mat)
            b_vec = self.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.gravity_vec)
            dv_free = A_inv @ b_vec
            dv_free.reshape((3*self.num_vertices, 1))
            Q_free = Q_a + h * Q_ad + h * dv_free
            for i in range(self.nCable):
                idx_pp = self.pp_idx[i]
                unit_vec = (self.pulley_location[i,:] - Q_free[3*idx_pp:(3*idx_pp+3)].reshape((3,)))
                unit_vec = unit_vec / np.linalg.norm(unit_vec)
                # print("unit_vec: ", unit_vec)
                phi_Qfree[i] = target_cable_length[i] - np.linalg.norm(self.pulley_location[i]-Q_free[3*idx_pp:3*idx_pp+3].reshape((3,)))
                H_free[i, 3*idx_pp:3*idx_pp+3] = unit_vec
            lcp_Mmat = h * H_free @ self.W_mat @ A_inv @ H_free.T
            M_is_PD = np.all(np.linalg.eigvals(lcp_Mmat) > 0)
            lcp_q = phi_Qfree.reshape((self.nCable,))
            cable_tension = projected_gauss_seidel_lcp(lcp_Mmat, phi_Qfree)
            cable_tension.reshape((self.nCable, 1))
            # print("cur time: ", t_a, "cable_tension: ", cable_tension, "lcp_q: ", lcp_q.flatten(), "M_is_PD: ", M_is_PD)
            cable_tension.reshape((self.nCable, 1))
            dv_cor = A_inv @ (self.W_mat @ H_free.T @ cable_tension).reshape((3*self.num_vertices, 1))
            dv = dv_free + dv_cor
            Q_ad = Q_ad + dv
            Q_a = Q_a + h * Q_ad
            t_a += h
            Q_list.append(Q_a.copy())
            if np.linalg.norm(Q_a - Q_a_last)/self.num_vertices < 1e-6:
                break
        vert_length = self.vertices.copy()
        
        for i in range(self.num_vertices):
            vert_length[i] = Q_a[3*i:3*i+3].reshape((3,))
        return vert_length, cable_tension.flatten()

    def FKD_length_forceinput(self, target_cable_length, starting_Q, starting_Qd):
        Q_a = starting_Q.copy()
        Q_ad = starting_Qd.copy()
        Q_a_last = Q_a.copy()
        t_a = 0.0
        h = 0.02
        total_time = 1.0
        phi_Qfree = np.zeros((self.nCable, 1))
        H_free = np.zeros((self.nCable, 3*self.num_vertices))
        Q_list = [Q_a.copy()]
        e_length = np.zeros((self.nCable, 1))
        e_length_last = e_length.copy()
        e_length_d = np.zeros((self.nCable, 1))
        Kp = 100
        Kd = -10
        cable_tension = np.zeros((self.nCable, 1))
        while t_a < total_time:
            Q_a_last = Q_a.copy()
            R_list_1212 = self.cal_rotation_fkLength(Q_a)
            Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
            K_mat = self.assemble_K(Ke_list)
            f0 = self.assemble_f0(Ke0_list)
            A_mat = (1.0/h)*np.eye(3*self.num_vertices) +h *  self.W_mat @ K_mat
            A_inv = np.linalg.inv(A_mat)
            b_vec = self.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.gravity_vec)
            dv_free = A_inv @ b_vec
            dv_free.reshape((3*self.num_vertices, 1))
            Q_free = Q_a + h * Q_ad + h * dv_free
            cur_length = self.get_cable_length(Q_free, self.pulley_location)
            for i in range(self.nCable):
                e_length[i] = cur_length[i] - target_cable_length[i]
            e_length_d = (e_length - e_length_last) / h
            e_length_last = e_length.copy()
            for i in range(self.nCable):
                idx_pp = self.pp_idx[i]
                unit_vec = (self.pulley_location[i,:] - Q_free[3*idx_pp:(3*idx_pp+3)].reshape((3,)))
                unit_vec = unit_vec / np.linalg.norm(unit_vec)
                # print("unit_vec: ", unit_vec)
                phi_Qfree[i] = target_cable_length[i] - np.linalg.norm(self.pulley_location[i]-Q_free[3*idx_pp:3*idx_pp+3].reshape((3,)))
                H_free[i, 3*idx_pp:3*idx_pp+3] = unit_vec
                cable_tension[i] = max(Kp * e_length[i] + Kd * e_length_d[i], 0)
            print("cur time: ", t_a, "cable_tension: ", cable_tension.flatten(), "e_length: ", e_length.flatten(), "e_length_d: ", e_length_d.flatten())
            cable_tension.reshape((self.nCable, 1))
            dv_cor = A_inv @ (self.W_mat @ H_free.T @ cable_tension).reshape((3*self.num_vertices, 1))
            dv = dv_free + dv_cor
            Q_ad = Q_ad + dv
            Q_a = Q_a + h * Q_ad
            t_a += h
            Q_list.append(Q_a.copy())
            if np.linalg.norm(Q_a - Q_a_last)/self.num_vertices < 1e-6:
                break
        vert_length = self.vertices.copy()
        
        for i in range(self.num_vertices):
            vert_length[i] = Q_a[3*i:3*i+3].reshape((3,))
        return Q_list, vert_length

    def FKD_time(self, target_cable_length, total_time, starting_Q, starting_Qd):
        Q_a = starting_Q.copy()
        Q_ad = starting_Qd.copy()
        t_a = 0.0
        h = 0.5
        phi_Qfree = np.zeros((self.nCable, 1))
        H_free = np.zeros((self.nCable, 3*self.num_vertices))
        Q_list = [Q_a.copy()]
        starting_cable_length = self.get_cable_length(Q_a, self.pulley_location)
        while t_a < total_time:
            tar_cable_length_this = starting_cable_length.copy()
            for i in range(self.nCable):
                tar_cable_length_this[i] = target_cable_length[i] * (t_a / total_time) + starting_cable_length[i] * (1 - t_a / total_time)
            Q_a_last = Q_a.copy()
            calR_start = time.time()
            R_list_1212 = self.cal_rotation_fkLength(Q_a)
            calR_end = time.time()
            calAB_start = time.time()
            Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
            K_mat = self.assemble_K(Ke_list)
            f0 = self.assemble_f0(Ke0_list)
            A_mat = (1.0/h)*np.eye(3*self.num_vertices) +h *  self.W_mat @ K_mat
            A_inv = np.linalg.inv(A_mat)
            b_vec = self.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.gravity_vec)
            calAB_end = time.time()
            dv_free = A_inv @ b_vec
            dv_free.reshape((3*self.num_vertices, 1))
            Q_free = Q_a + h * Q_ad + h * dv_free
            for i in range(self.nCable):
                idx_pp = self.pp_idx[i]
                unit_vec = (self.pulley_location[i,:] - Q_free[3*idx_pp:(3*idx_pp+3)].reshape((3,)))
                unit_vec = unit_vec / np.linalg.norm(unit_vec)
                # print("unit_vec: ", unit_vec)
                phi_Qfree[i] = tar_cable_length_this[i] - np.linalg.norm(self.pulley_location[i]-Q_free[3*idx_pp:3*idx_pp+3].reshape((3,)))
                H_free[i, 3*idx_pp:3*idx_pp+3] = unit_vec
            lcp_Mmat = h * H_free @ self.W_mat @ A_inv @ H_free.T
            lcp_q = phi_Qfree.reshape((self.nCable,))
            lcp_sol = qe.optimize.lcp_lemke(lcp_Mmat, lcp_q)
            if not lcp_sol.success:
                print("lcp failed: ")
                break
            cable_tension = lcp_sol.z
            # print("cur time: ", t_a, "cable_tension: ", cable_tension, "lcp_q: ", lcp_q.flatten(), "M_is_PD: ", M_is_PD)
            # cable_tension = solve_qp(lcp_Mmat, phi_Qfree, -lcp_Mmat, phi_Qfree, lb = np.zeros((self.nCable,)), solver = 'cvxopt')
            cable_tension.reshape((self.nCable, 1))
            # print("cur time: ", t_a, "cable_tension: ", cable_tension, "lcp_q: ", lcp_q.flatten(), "M_is_PD: ", M_is_PD)
            print("cur time: ", t_a, "cable_tension: ", cable_tension, "calR time: ", calR_end - calR_start, "calAB time: ", calAB_end - calAB_start)
            cable_tension.reshape((self.nCable, 1))
            dv_cor = A_inv @ (self.W_mat @ H_free.T @ cable_tension).reshape((3*self.num_vertices, 1))
            dv = dv_free + dv_cor
            Q_ad = Q_ad + dv
            Q_a = Q_a + h * Q_ad
            t_a += h
            Q_list.append(Q_a.copy())
        vert_length = self.vertices.copy()
        for i in range(self.num_vertices):
            vert_length[i] = Q_a[3*i:3*i+3].reshape((3,))
        return Q_list, vert_length

    def FKD_sequence(self, target_cable_length_list, time_list, starting_Q):
        time_diff_list = [time_list[0]]
        Q_list_all = []
        for i in range(1, len(time_list)):
            time_diff_list.append(time_list[i] - time_list[i-1])
        for tar_cable_length, time in zip(target_cable_length_list, time_diff_list):
            Q_list, verts = self.FKD_time(tar_cable_length, time, starting_vertices)
            starting_vertices = verts.copy()
            Q_list_all.extend(Q_list)
        return Q_list_all, verts

    def check_contact(self, Q_a):
        is_contact = [0 for _ in range(self.num_vertices)]
        contact_distance = [0 for _ in range(self.num_vertices)]
        contact_unit_direction = [np.zeros((3,)) for _ in range(self.num_vertices)]
        Q_a_minz = np.min([Q_a[3*i+2] for i in range(self.num_vertices)])
        seg_num = int((Q_a_minz-self.minz)/ self.seg_length)
        if seg_num < 0:
            seg_num = 0
        if seg_num >= self.nSeg:
            seg_num = self.nSeg - 1
        armseg_pts = self.pointCloud_seglist[seg_num]
        armseg_normals = self.normalvec_seglist[seg_num]
        for i in range(self.num_vertices):
            query_point = Q_a[3*i:3*i+3].reshape((3,))
            is_inside, distance, unit_direction, closest_point = min_distance_and_direction(self.arm_kdtree,self.arm_contour, self.arm_normals, query_point)
            # is_inside, distance, unit_direction, closest_point = min_distance_and_direction(armseg_pts, armseg_normals, query_point)
            if is_inside:
                is_contact[i] = 1
                contact_distance[i] = distance
                contact_unit_direction[i] = unit_direction
        return is_contact, contact_distance, contact_unit_direction

    def check_contact_idx(self, Q_a):
        is_contact = [0 for _ in range(self.num_contact_vertices)]
        contact_distance = [0 for _ in range(self.num_contact_vertices)]
        contact_unit_direction = [np.zeros((3,)) for _ in range(self.num_contact_vertices)]
        seg_num = int((Q_a[2]-self.minz)/ self.seg_length)
        if seg_num < 0:
            seg_num = 0
        if seg_num >= self.nSeg:
            seg_num = self.nSeg - 1
        armseg_pts = self.pointCloud_seglist[seg_num]
        armseg_normals = self.normalvec_seglist[seg_num]
        for i in range(self.num_contact_vertices):
            query_point = Q_a[3*self.idxContact2_idxAll[i]:3*self.idxContact2_idxAll[i]+3].reshape((3,))
            is_inside, distance, unit_direction, closest_point = min_distance_and_direction(armseg_pts, armseg_normals, query_point)
            # is_inside, distance, normal, closest_point
            is_contact[i] = 1 if is_inside else 0
            contact_distance[i] = distance
            contact_unit_direction[i] = unit_direction
        return is_contact, contact_distance, contact_unit_direction

    def step(self, cur_Q, pulley_location_a, cmd):
        Q_a = cur_Q.copy()
        # Q_vertical_elevation = np.zeros((3*self.num_vertices, 1))
        t_a = 0.0
        elevation_diff = cmd[-1]
        length_diff = cmd[0:self.nCable]
        total_time = 1.0
        pulley_location_stepped = pulley_location_a.copy()
        pulley_location_stepped[:, 1] -= elevation_diff
        for i in range(self.num_vertices):
            Q_a[3*i+1, 0] -= elevation_diff
        h = 0.01
        tar_cable_length = self.get_cable_length(Q_a, pulley_location_stepped)
        for i in range(self.nCable):
            tar_cable_length[i] += length_diff[i]
        Q_a_last = Q_a.copy()
        phi_Qfree = np.zeros((self.nCable, 1))
        H_free = np.zeros((self.nCable, 3*self.num_vertices))
        Q_ad = np.zeros((3*self.num_vertices, 1))
        while t_a < total_time:
            Q_a_last = Q_a.copy()
            R_list_1212 = self.cal_rotation_fkLength(Q_a)
            Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
            K_mat = self.assemble_K(Ke_list)
            f0 = self.assemble_f0(Ke0_list)
            A_mat = (1.0/h)*np.eye(3*self.num_vertices) +h *  self.W_mat @ K_mat
            A_inv = np.linalg.inv(A_mat)
            b_vec = self.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.gravity_vec)
            dv_free = A_inv @ b_vec
            dv_free.reshape((3*self.num_vertices, 1))
            Q_free = Q_a + h * Q_ad + h * dv_free
            for i in range(self.nCable):
                idx_pp = self.pp_idx[i]
                unit_vec = (self.pulley_location[i,:] - Q_free[3*idx_pp:(3*idx_pp+3)].reshape((3,)))
                unit_vec = unit_vec / np.linalg.norm(unit_vec)
                # print("unit_vec: ", unit_vec)
                cl_free = self.get_cable_length(Q_free, pulley_location_stepped)
                phi_Qfree[i] = tar_cable_length[i] - cl_free[i]
                H_free[i, 3*idx_pp:3*idx_pp+3] = unit_vec
            lcp_Mmat = h * H_free @ self.W_mat @ A_inv @ H_free.T
            M_is_PD = np.all(np.linalg.eigvals(lcp_Mmat) > 0)
            lcp_q = phi_Qfree.reshape((self.nCable,))
            # cable_tension = solve_qp(lcp_Mmat, phi_Qfree, -lcp_Mmat, phi_Qfree, lb = np.zeros((self.nCable,)), solver = 'cvxopt')
            # cable_tension.reshape((self.nCable, 1))
            lcp_sol = qe.optimize.lcp_lemke(lcp_Mmat, lcp_q)
            if not lcp_sol.success:
                print("lcp failed: ")
                break
            cable_tension = lcp_sol.z
            # print("cur time: ", t_a, "cable_tension: ", cable_tension, "lcp_q: ", lcp_q.flatten(), "M_is_PD: ", M_is_PD)
            cable_tension.reshape((self.nCable, 1))
            dv_cor = A_inv @ (self.W_mat @ H_free.T @ cable_tension).reshape((3*self.num_vertices, 1))
            dv = dv_free + dv_cor
            Q_ad = Q_ad + dv
            Q_a = Q_a + h * Q_ad
            t_a += h
            if np.linalg.norm(Q_a - Q_a_last)/self.num_vertices < 1e-9:
                break
        vert_length = self.vertices.copy()
        return Q_a, cable_tension, pulley_location_stepped
        
    def find_optimal_control_input(self, starting_configs):
        Q_a = starting_configs['starting_Q']
        Q_ad = starting_configs['starting_Qd']
        starting_elevation = starting_configs['starting_elevation']
        starting_horizontal_move = starting_configs['starting_horizontal_move']
        cur_pulley_location = self.pulley_location.copy()
        cur_pulley_location[:, 1] -= starting_elevation
        cur_pulley_location[:, 2] += starting_horizontal_move
        Q_vertical_elevation = np.zeros((3*self.num_vertices, 1))
        Q_horizontal_move = np.zeros((3*self.num_vertices, 1))
        for i in range(self.num_vertices):
            Q_vertical_elevation[3*i+1, 0] = -starting_elevation
            Q_horizontal_move[3*i+2, 0] = starting_horizontal_move
        Q_a += Q_vertical_elevation + Q_horizontal_move
        num_iter = 0
        max_iter = 100
        Qa_last = Q_a.copy()
        Q_ad = starting_configs['starting_Qd']
        Q_a, cable_tension, pulley_location_stepped = self.step(Q_a, cur_pulley_location, np.zeros((self.nCable+1,)))
        # self.draw_mesh_w_arm(Q_a, pulley_location_stepped)
        while num_iter < max_iter:
            obj, is_contact, contact_distance, contact_unit_direction = self.contact_objective_func(Q_a)
            print("num contact vertices: ", sum(is_contact))
            obj_gradient = np.array([0.0 for _ in range(self.nCable+1)])
            Jac_moving = self.cal_jacobian_cg(Q_a, cur_pulley_location) # nCable X (3*num_moving_vertices)
            for i in range(self.num_contact_vertices):
                idx_all = self.idxContact2_idxAll[i]
                idx_moving = self.idxAll_2_idxMoving[idx_all]
                if is_contact[i]:
                    for j in range(self.nCable):
                        for k in range(3):
                            obj_gradient[j] += 2 * contact_distance[i] * contact_unit_direction[i][k] * Jac_moving[j, 3*idx_moving + k]
                    obj_gradient[-1] += 20 * contact_distance[i] * contact_unit_direction[i][1]  # vertical elevation
                else:
                    for j in range(self.nCable):
                        for k in range(3):
                            obj_gradient[j] += 2 * contact_distance[i] * contact_unit_direction[i][k] * Jac_moving[j, 3*idx_moving + k]
                    obj_gradient[-1] += 20 * contact_distance[i] * contact_unit_direction[i][1] # vertical elevation
            for i in range(self.nCable):
                if cable_tension[i] <= 1e-6 and obj_gradient[i] < 0:
                    obj_gradient[i] = 0.0
            step_size = 0.001
            cmd = -step_size * obj_gradient
            # normalize cmd to have max change of 0.01 m, and minimum change of 0.001 m
            max_cmd = np.max(np.abs(cmd))
            if max_cmd > 0.01:
                cmd = cmd * (0.01 / max_cmd)
            if max_cmd < 0.001:
                cmd = cmd * (0.001 / max_cmd)
            print("Iteration: ", num_iter, "Contact Objective: ", obj, "Gradient: ", obj_gradient, "Cmd: ", cmd)
            Q_a, cable_tension, cur_pulley_location = self.step(Q_a, cur_pulley_location, cmd)
            if np.linalg.norm(Q_a - Qa_last)/self.num_vertices < 1e-6:
                break
            Qa_last = Q_a.copy()
            num_iter += 1
        return Q_a, cur_pulley_location

    def contact_objective_func(self, Q_a):
        is_contact, contact_distance, contact_unit_direction = self.check_contact_idx(Q_a)
        obj = 0.0
        for i in range(self.num_contact_vertices):
            if is_contact[i]:
                obj += 10*contact_distance[i]**2
            else:
                obj += contact_distance[i]**2
        return obj, is_contact, contact_distance, contact_unit_direction

    def simulate_contact_autoele(self, starting_Q, cur_pulley_location, h = 0.001):
        # find an elevation that make starting Q has no interaction
        ele = 0.001 # elevate for 1mm
        for i in range(1000):
            final_ele = ele+i*0.001
            Q_eled = starting_Q.copy()
            for j in range(self.num_vertices):
                Q_eled[3*j+1] -= final_ele
            is_contact, contact_distance, contact_unit_direction = self.check_contact(Q_eled)
            if sum(is_contact) == 0:
                break
        print("final_ele", final_ele)
        Q_a = Q_eled
        Q_a_last = Q_a.copy()
        Q_ad = np.zeros((3*self.num_vertices, 1))
        start_pulley_location = cur_pulley_location.copy()
        start_pulley_location[:, 1] -= final_ele
        Q_list = []
        # move down 0.001 every time step
        n_down = int(final_ele/0.001)
        target_cable_length = self.get_cable_length(Q_a, start_pulley_location)
        cable_tension_list = []
        is_contact_list = []
        contact_force_list = []
        contact_unit_direction_list = []
        pulley_location_list = []
        for i_down in range(n_down):
            # all y coordinate + 0.001
            for j in range(self.num_vertices):
                Q_a[3*j+1] += 0.001
            cur_pulley_location = start_pulley_location.copy()
            cur_pulley_location[:, 1] += 0.001 * (i_down + 1)
            # stabilize for 50 steps
            t_a = 0.0
            h = 0.0001
            for step in range(50):
                R_list_1212 = self.cal_rotation_fkLength(Q_a)
                Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
                K_mat = self.assemble_K(Ke_list)
                f0 = self.assemble_f0(Ke0_list)
                A_mat = (1.0/h)*np.eye(3*self.num_vertices) + h * self.W_mat @ K_mat
                A_inv = np.linalg.inv(A_mat)
                b_vec = self.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.gravity_vec)
                dv_free = A_inv @ b_vec
                dv_free.reshape((3*self.num_vertices, 1))
                Q_free = Q_a + h * Q_ad + h * dv_free
                is_contact, contact_distance, contact_unit_direction = self.check_contact(Q_free)
                num_contact = sum(is_contact)
                idxAll_2_idxContact = [-1 for _ in range(self.num_vertices)]
                idxContact_2_idxAll = []
                idx_contact = 0
                for i in range(self.num_vertices):
                    if is_contact[i]:
                        idxAll_2_idxContact[i] = idx_contact
                        idxContact_2_idxAll.append(i)
                        idx_contact += 1
                phi_Qfree = np.zeros((self.nCable + num_contact, 1))
                H_free = np.zeros((self.nCable + num_contact, 3*self.num_vertices))
                for i in range(self.nCable):
                    idx_pp = self.pp_idx[i]
                    unit_vec = (cur_pulley_location[i,:] - Q_free[3*idx_pp:(3*idx_pp+3)].reshape((3,)))
                    unit_vec = unit_vec / np.linalg.norm(unit_vec)
                    # print("unit_vec: ", unit_vec)
                    phi_Qfree[i] = target_cable_length[i] - np.linalg.norm(cur_pulley_location[i]-Q_free[3*idx_pp:3*idx_pp+3].reshape((3,)))
                    H_free[i, 3*idx_pp:3*idx_pp+3] = unit_vec
                for i in range(num_contact):
                    idx_contact = idxContact_2_idxAll[i]
                    unit_vec = contact_unit_direction[idx_contact]
                    phi_Qfree[self.nCable + i] = -contact_distance[idx_contact]
                    H_free[self.nCable + i, 3*idx_contact:3*idx_contact+3] = unit_vec
                lcp_Mmat = h * H_free @ self.W_mat @ A_inv @ H_free.T
                lcp_q = phi_Qfree.reshape((self.nCable + num_contact,))
                
                
                # extrernal_forces = solve_qp(lcp_Mmat, phi_Qfree, -lcp_Mmat, phi_Qfree, lb = np.zeros((self.nCable + num_contact,)), solver = 'Clarabel')
                res = qe.optimize.lcp_lemke(lcp_Mmat, lcp_q)
                if not res.success:
                    print("LCP solver failed, using cvxopt solver instead.")
                    extrernal_forces = solve_qp(2 * lcp_Mmat, phi_Qfree, -lcp_Mmat, phi_Qfree, lb = np.zeros((self.nCable + num_contact,)), solver = 'cvxopt')
                else:
                    extrernal_forces = res.z
                    # print("check LCP condition: ", np.all(np.isfinite(extrernal_forces)))
                extrernal_forces.reshape((self.nCable + num_contact, 1))
                cable_tension = extrernal_forces[:self.nCable]
                contact_forces = extrernal_forces[self.nCable:]
                cable_tension.reshape((self.nCable, 1))
                dv_cor = A_inv @ (self.W_mat @ H_free.T @ extrernal_forces).reshape((3*self.num_vertices, 1))
                dv = dv_free + dv_cor
                Q_ad = Q_ad + dv
                Q_a = Q_a + h * Q_ad
                t_a += h
                error = np.linalg.norm(Q_a - Q_a_last)/self.num_vertices
                if error < 1e-6:
                    break
                Q_a_last = Q_a.copy()
                print("Down step: ", i_down, "error: ", error, "cable_tension: ", cable_tension.flatten(), "num_contact: ", num_contact)
            is_contact, contact_distance, contact_unit_direction = self.check_contact(Q_a)
            cable_tension_list.append(cable_tension.copy())
            is_contact_list.append(is_contact)
            contact_force_list.append(contact_forces.copy())
            contact_unit_direction_list.append(contact_unit_direction)
            Q_list.append(Q_a.copy())
            pulley_location_list.append(cur_pulley_location.copy())
            configs = {
                'Q_list': Q_list,
                'pulley_location_list': pulley_location_list,
                'is_contact_list': is_contact_list,
                'contact_unit_direction_list': contact_unit_direction_list,
                'contact_force_list': contact_force_list,
                'cable_tension_list': cable_tension_list
            }
        return configs
        
    def simulate_contact(self, starting_Q, cur_pulley_location, final_ele, h = 0.001):
        # find an elevation that make starting Q has no interaction
        ele = 0.001 # elevate for 1mm
        Q_a = starting_Q
        Q_a_last = Q_a.copy()
        target_cable_length = self.get_cable_length(Q_a, cur_pulley_location)
        print("target_cable_length: ", target_cable_length)
        print("initial cable length: ", self.initial_cable_length)
        input("Press Enter to continue...")
        Q_ad = np.zeros((3*self.num_vertices, 1))
        start_pulley_location = cur_pulley_location.copy()
        Q_list = []
        # move down 0.001 every time step
        n_down = int(final_ele/0.001)
        cable_tension_list = []
        is_contact_list = []
        contact_force_list = []
        contact_unit_direction_list = []
        pulley_location_list = []
        idxContact_2_idxAll_list = []
        for i_down in range(n_down):
            # all y coordinate + 0.001
            for j in range(self.num_vertices):
                Q_a[3*j+1] += 0.001
            cur_pulley_location = start_pulley_location.copy()
            cur_pulley_location[:, 1] += 0.001 * (i_down + 1)
            # stabilize for 50 steps
            t_a = 0.0
            h = 0.001
            for step in range(100):
                R_list_1212 = self.cal_rotation_fkLength(Q_a)
                Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
                K_mat = self.assemble_K(Ke_list)
                f0 = self.assemble_f0(Ke0_list)
                A_mat = (1.0/h)*np.eye(3*self.num_vertices) + h * self.W_mat @ K_mat
                A_inv = np.linalg.inv(A_mat)
                b_vec = self.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.gravity_vec)
                dv_free = A_inv @ b_vec
                dv_free.reshape((3*self.num_vertices, 1))
                Q_free = Q_a + h * Q_ad + h * dv_free
                is_contact, contact_distance, contact_unit_direction = self.sdf_func_Q(Q_free)
                num_contact = sum(is_contact)
                idxAll_2_idxContact = [-1 for _ in range(self.num_vertices)]
                idxContact_2_idxAll = []
                idx_contact = 0
                for i in range(self.num_vertices):
                    if is_contact[i]:
                        idxAll_2_idxContact[i] = idx_contact
                        idxContact_2_idxAll.append(i)
                        idx_contact += 1
                phi_Qfree = np.zeros((self.nCable + num_contact, 1))
                H_free = np.zeros((self.nCable + num_contact, 3*self.num_vertices))
                for i in range(self.nCable):
                    idx_pp = self.pp_idx[i]
                    unit_vec = (cur_pulley_location[i,:] - Q_free[3*idx_pp:(3*idx_pp+3)].reshape((3,)))
                    unit_vec = unit_vec / np.linalg.norm(unit_vec)
                    # print("unit_vec: ", unit_vec)
                    phi_Qfree[i] = target_cable_length[i] - np.linalg.norm(cur_pulley_location[i]-Q_free[3*idx_pp:3*idx_pp+3].reshape((3,)))
                    H_free[i, 3*idx_pp:3*idx_pp+3] = unit_vec
                for i in range(num_contact):
                    idx_contact = idxContact_2_idxAll[i]
                    unit_vec = contact_unit_direction[idx_contact]
                    phi_Qfree[self.nCable + i] = contact_distance[idx_contact]
                    H_free[self.nCable + i, 3*idx_contact:3*idx_contact+3] = unit_vec
                lcp_Mmat = h * H_free @ self.W_mat @ A_inv @ H_free.T
                lcp_q = phi_Qfree.reshape((self.nCable + num_contact,))
                # extrernal_forces = solve_qp(lcp_Mmat, phi_Qfree, -lcp_Mmat, phi_Qfree, lb = np.zeros((self.nCable + num_contact,)), solver = 'Clarabel')
                res = qe.optimize.lcp_lemke(lcp_Mmat, lcp_q)
                if not res.success:
                    print("LCP solver failed, using cvxopt solver instead.")
                    extrernal_forces = solve_qp(2 * lcp_Mmat, phi_Qfree, -lcp_Mmat, phi_Qfree, lb = np.zeros((self.nCable + num_contact,)), solver = 'cvxopt')
                else:
                    extrernal_forces = res.z
                    # print("check LCP condition: ", np.all(np.isfinite(extrernal_forces)))
                extrernal_forces.reshape((self.nCable + num_contact, 1))
                cable_tension = extrernal_forces[:self.nCable]
                contact_forces = extrernal_forces[self.nCable:]
                cable_tension.reshape((self.nCable, 1))
                dv_cor = A_inv @ (self.W_mat @ H_free.T @ extrernal_forces).reshape((3*self.num_vertices, 1))
                dv = dv_free + dv_cor
                Q_ad = Q_ad + dv
                Q_a = Q_a + h * Q_ad
                t_a += h
                error = np.linalg.norm(Q_a - Q_a_last)/self.num_vertices
                if error < 1e-6:
                    break
                Q_a_last = Q_a.copy()
                is_contact_list.append(is_contact)
                contact_force_list.append(contact_forces.copy())
                contact_unit_direction_list.append(contact_unit_direction)
                Q_list.append(Q_a.copy())
                pulley_location_list.append(cur_pulley_location.copy())
                idxContact_2_idxAll_list.append(idxContact_2_idxAll.copy())
                print("Down step: ", i_down, "error: ", error, "cable_tension: ", cable_tension.flatten(), "num_contact: ", num_contact)
            # is_contact, contact_distance, contact_unit_direction = self.sdf_func_Q(Q_a)
                cable_tension_list.append(cable_tension.copy())
            
            # self.draw_contact_inmoving(Q_a, cur_pulley_location, is_contact, contact_unit_direction, contact_forces, idxContact_2_idxAll)
            # testkey = input("Press Enter to continue or q to quit and save")
            # if testkey == 'q': 
            #     break
        configs = {
            'Q_list': Q_list,
            'pulley_location_list': pulley_location_list,
            'is_contact_list': is_contact_list,
            'contact_unit_direction_list': contact_unit_direction_list,
            'contact_force_list': contact_force_list,
            'cable_tension_list': cable_tension_list,
            'idxContact_2_idxAll_list': idxContact_2_idxAll_list
        }
        return configs

    def simulation_onetime(self, target_cable_length, vertical_elevation, horizontal_move, total_time, starting_configs, h = 0.001):
        Q_a = starting_configs['starting_Q']
        Q_ad = starting_configs['starting_Qd']
        starting_elevation = starting_configs['starting_elevation']
        starting_horizontal_move = starting_configs['starting_horizontal_move']
        cur_pulley_location = self.pulley_location.copy()
        cur_pulley_location[:, 1] -= starting_elevation
        cur_pulley_location[:, 2] += starting_horizontal_move
        Q_vertical_elevation = np.zeros((3*self.num_vertices, 1))
        Q_horizontal_move = np.zeros((3*self.num_vertices, 1))
        for i in range(self.num_vertices):
            Q_vertical_elevation[3*i+1, 0] = -starting_elevation
            Q_horizontal_move[3*i+2, 0] = starting_horizontal_move
        Q_a += Q_vertical_elevation + Q_horizontal_move
        t_a = 0.0
        
        Q_list = [Q_a.copy()]
        
        is_contact, contact_distance, contact_unit_direction = self.check_contact(Q_a)
        is_contact_list = [is_contact]
        contact_distance_list = [contact_distance]
        contact_force_list = [np.zeros((self.nCable, 1))]
        contact_idx_list = [[-1 for _ in range(self.num_vertices)]]
        contact_unit_direction_list = [contact_unit_direction]
        pulley_location_list = [cur_pulley_location.copy()]
        starting_cable_length = self.get_cable_length(Q_a, cur_pulley_location)

        while t_a < total_time:
            tar_cable_length_this = starting_cable_length.copy()
            vertical_elevation_this = vertical_elevation/(total_time/h)
            horizontal_move_this = horizontal_move/(total_time/h)
            for i in range(self.nCable):
                tar_cable_length_this[i] = target_cable_length[i] * (t_a / total_time) + starting_cable_length[i] * (1 - t_a / total_time)
            calR_start = time.time()
            R_list_1212 = self.cal_rotation_fkLength(Q_a)
            calR_end = time.time()
            calAB_start = time.time()
            Ke_list, Ke0_list = self.cal_Ke_lists(R_list_1212)
            K_mat = self.assemble_K(Ke_list)
            f0 = self.assemble_f0(Ke0_list)
            A_mat = (1.0/h)*np.eye(3*self.num_vertices) +h *  self.W_mat @ K_mat
            A_inv = np.linalg.inv(A_mat)
            b_vec = self.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.gravity_vec)
            calAB_end = time.time()
            dv_free = A_inv @ b_vec
            dv_free.reshape((3*self.num_vertices, 1))
            Q_free = Q_a + h * Q_ad + h * dv_free
            is_contact, contact_distance, contact_unit_direction = self.check_contact(Q_free)
            num_contact = sum(is_contact)
            idxAll_2_idxContact = [-1 for _ in range(self.num_vertices)]
            idxContact_2_idxAll = []
            idx_contact = 0
            for i in range(self.num_vertices):
                if is_contact[i]:
                    idxAll_2_idxContact[i] = idx_contact
                    idxContact_2_idxAll.append(i)
                    idx_contact += 1
            phi_Qfree = np.zeros((self.nCable + num_contact, 1))
            H_free = np.zeros((self.nCable + num_contact, 3*self.num_vertices))
            for i in range(self.nCable):
                idx_pp = self.pp_idx[i]
                unit_vec = (self.pulley_location[i,:] - Q_free[3*idx_pp:(3*idx_pp+3)].reshape((3,)))
                unit_vec = unit_vec / np.linalg.norm(unit_vec)
                # print("unit_vec: ", unit_vec)
                phi_Qfree[i] = tar_cable_length_this[i] - np.linalg.norm(cur_pulley_location[i]-Q_free[3*idx_pp:3*idx_pp+3].reshape((3,)))
                H_free[i, 3*idx_pp:3*idx_pp+3] = unit_vec
            for i in range(num_contact):
                idx_contact = idxContact_2_idxAll[i]
                unit_vec = contact_unit_direction[idx_contact]
                phi_Qfree[self.nCable + i] = -contact_distance[idx_contact]
                H_free[self.nCable + i, 3*idx_contact:3*idx_contact+3] = unit_vec
            lcp_Mmat = h * H_free @ self.W_mat @ A_inv @ H_free.T
            lcp_q = phi_Qfree.reshape((self.nCable + num_contact,))
            
            # extrernal_forces = solve_qp(lcp_Mmat, phi_Qfree, -lcp_Mmat, phi_Qfree, lb = np.zeros((self.nCable + num_contact,)), solver = 'Clarabel')
            res = qe.optimize.lcp_lemke(lcp_Mmat, lcp_q)
            if not res.success:
                print("LCP solver failed, using cvxopt solver instead.")
                extrernal_forces = solve_qp(2 * lcp_Mmat, phi_Qfree, -lcp_Mmat, phi_Qfree, lb = np.zeros((self.nCable + num_contact,)), solver = 'cvxopt')
            else:
                extrernal_forces = res.z
                # print("check LCP condition: ", np.all(np.isfinite(extrernal_forces)))
            extrernal_forces.reshape((self.nCable + num_contact, 1))
            cable_tension = extrernal_forces[:self.nCable]
            contact_forces = extrernal_forces[self.nCable:]
            # print("cur time: ", t_a, "cable_tension: ", cable_tension, "lcp_q: ", lcp_q.flatten(), "M_is_PD: ", M_is_PD)
            print("cur time: ", t_a, "cable_tension: ", cable_tension)
            cable_tension.reshape((self.nCable, 1))
            dv_cor = A_inv @ (self.W_mat @ H_free.T @ extrernal_forces).reshape((3*self.num_vertices, 1))
            dv = dv_free + dv_cor
            Q_ad = Q_ad + dv
            Q_a = Q_a + h * Q_ad
            t_a += h
            Q_vertical_elevation = np.zeros((3*self.num_vertices, 1))
            Q_horizontal_move = np.zeros((3*self.num_vertices, 1))
            for i in range(self.num_vertices):
                Q_vertical_elevation[3*i+1, 0] = -vertical_elevation_this
                Q_horizontal_move[3*i+2, 0] = horizontal_move_this
            Q_a += Q_vertical_elevation + Q_horizontal_move
            cur_pulley_location[:, 1] -= vertical_elevation_this
            cur_pulley_location[:, 2] += horizontal_move_this
            # is_contact, contact_distance, contact_unit_direction = self.check_contact(Q_a)
            is_contact_list.append(is_contact)
            contact_distance_list.append(contact_distance)
            contact_unit_direction_list.append(contact_unit_direction)
            contact_idx_list.append(idxAll_2_idxContact)
            contact_force_list.append(contact_forces)
            Q_list.append(Q_a.copy())
            pulley_location_list.append(cur_pulley_location.copy())
            configs = {
                'Q_list': Q_list,
                'pulley_location_list': pulley_location_list,
                'is_contact_list': is_contact_list,
                'contact_distance_list': contact_distance_list,
                'contact_unit_direction_list': contact_unit_direction_list,
                'contact_force_list': contact_force_list,
                'contact_idx_list': contact_idx_list
            }
        return configs

    def simulation_sequence(self, cable_length_list, vertical_elevation_list, horizontal_move_list, time_list, starting_configs, h = 0.1):
        for i in range(len(cable_length_list)):
            self.simulation_onetime(cable_length_list[i], vertical_elevation_list[i], horizontal_move_list[i], time_list[i], starting_configs)

# modal analysis part
    def generate_dataset(self, data_file, n_samples=1000):
        from scipy.stats import qmc

        icl = self.initial_cable_length
        # max shortening per cable: cables 0,3 by 0.02m; cables 1,2 by 0.06m
        upper_bounds = np.array([0.02, 0.06, 0.06, 0.02])

        # Latin Hypercube Sampling over shortening amounts
        sampler = qmc.LatinHypercube(d=self.nCable, seed=42)
        samples = sampler.random(n=n_samples)          # (n_samples, nCable) in [0,1]
        delta_cl_samples = samples * upper_bounds       # actual shortening in meters

        # Sort by L2 norm ascending so warm-start stays close to previous equilibrium
        sort_idx = np.argsort(np.linalg.norm(delta_cl_samples, axis=1))
        delta_cl_samples = delta_cl_samples[sort_idx]

        dataset = []
        cur_Q = self.Q0.copy()
        cur_Qd = np.zeros((3 * self.num_vertices, 1))

        for k, delta_cl in enumerate(delta_cl_samples):
            target_cl = [icl[i] - delta_cl[i] for i in range(self.nCable)]
            print(f"Sample {k+1}/{n_samples}  delta_cl={np.round(delta_cl, 4)}  target_cl={np.round(target_cl, 4)}")

            try:
                verts = self.FKD_static(target_cl, cur_Q, cur_Qd)
            except Exception as e:
                print(f"  FKD_static failed: {e}  -- resetting to rest pose and skipping")
                cur_Q = self.Q0.copy()
                cur_Qd = np.zeros((3 * self.num_vertices, 1))
                continue

            # Reconstruct Q from returned verts for next warm-start
            cur_Q = np.zeros((3 * self.num_vertices, 1))
            for i in range(self.num_vertices):
                cur_Q[3*i:3*i+3, 0] = verts[i]
            cur_Qd = np.zeros((3 * self.num_vertices, 1))

            # Actual cable lengths after deformation (may differ from target if cable went slack)
            final_cl = self.get_cable_length(cur_Q, self.pulley_location)

            dataset.append({
                'delta_cl': delta_cl.copy(),
                'target_cable_length': np.array(target_cl),
                'final_cable_length': np.array(final_cl),
                'deformed_vertices': verts.copy(),
            })

            # Save incrementally so partial runs are not lost
            with open(data_file, 'wb') as f:
                pickle.dump(dataset, f)

        print(f"Done. {len(dataset)}/{n_samples} samples saved to {data_file}")
        return dataset


    def animate_simulation_sequence(self, configs, fps=30):
        output_dir = Path("frames")
        output_dir.mkdir(exist_ok=True)
        Q_list = configs['Q_list']
        pulley_location_list = configs['pulley_location_list']
        is_contact_list = configs['is_contact_list']
        contact_distance_list = configs['contact_distance_list']
        contact_unit_direction_list = configs['contact_unit_direction_list']
        contact_idx_list = configs['contact_idx_list']
        contact_force_list = configs['contact_force_list']
        n_frames = len(Q_list)
        plotter = pv.Plotter(off_screen=True)
        frames = []
        xmin = min(self.vertices[:, 0].min()-0.01, self.arm_contour[:, 0].min()-0.01)
        xmax = max(self.vertices[:, 0].max()+0.01, self.arm_contour[:, 0].max()+0.01)
        ymin = min(self.vertices[:, 1].min()-0.01, self.arm_contour[:, 1].min()-0.01)
        ymax = max(self.vertices[:, 1].max()+0.01, self.arm_contour[:, 1].max()+0.01)
        zmin = min(self.vertices[:, 2].min()-0.01, self.arm_contour[:, 2].min()-0.01)
        zmax = max(self.vertices[:, 2].max()+0.01, self.arm_contour[:, 2].max()+0.01)
        for i in range(n_frames):
            plotter.clear()
            vert = self.vertices.copy()
            cur_pulley_location = pulley_location_list[i]
            for j in range(self.num_vertices):
                vert[j] = Q_list[i][3*j:3*j+3].reshape((3,))
            mesh = pv.PolyData(vert)
            mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])
            plotter.add_mesh(mesh, color='lightblue', show_edges=True)
            for j in range(self.nCable):
                plotter.add_lines(np.array([cur_pulley_location[j], vert[self.pp_idx[j]]]), color='green')
            plotter.show_grid()
            plotter.show_axes()
            # add 8 points with xyz min and max
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmin, ymin, zmin]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmin, ymin, zmax]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmin, ymax, zmin]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmin, ymax, zmax]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmax, ymin, zmin]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmax, ymax, zmin]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmax, ymax, zmax]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmax, ymin, zmax]), color='red')
            grid = pv.StructuredGrid(self.arm_contour[:, 0], self.arm_contour[:, 1], self.arm_contour[:, 2])
            # add contact points
            idxAll_2_idxContact = contact_idx_list[i]
            for j in range(self.num_vertices):
                if is_contact_list[i][j]:
                    plotter.add_mesh(pv.Sphere(radius=0.01, center=vert[j]), color='blue')
                    # add arrow in the contact direction
                    contact_force = contact_force_list[i][idxAll_2_idxContact[j]]
                    contact_unit_dir = contact_unit_direction_list[i][j].reshape((3,))

                    contact_force_v = contact_unit_dir * 0.1
                    plotter.add_arrows(vert[j], contact_force_v, color='blue')

            plotter.add_mesh(grid, color='orange', show_edges=True, opacity=0.5)
            # make axis equal
            plotter.set_position([0.15, -0.1, 0.5])
            plotter.set_viewup([0, -1, 0])
            plotter.set_focus([0, 0, 0])
            plotter.set_scale(1, 1, 1)
            frame = plotter.screenshot(return_img=True)
            frame_path = output_dir / f"frame_{i:03d}.png"
            # Convert RGB (PyVista output) to BGR (OpenCV format) and save
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), frame_bgr)
            frames.append(frame_path)
        frame = cv2.imread(str(frames[0]))
        height, width, _ = frame.shape

        # Initialize OpenCV video writer
        output_video = "testsimulation.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # Write frames to video
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            video_writer.write(frame)

        # Release the video writer
        video_writer.release()

        # Clean up temporary frame files
        for frame_path in frames:
            os.remove(frame_path)
        output_dir.rmdir()

        plotter.close()

    def animate_contact_sequence(self, configs, fps=30, filename = "contact_simulation.mp4"):
        output_dir = Path("frames")
        output_dir.mkdir(exist_ok=True)
        Q_list = configs['Q_list']
        pulley_location_list = configs['pulley_location_list']
        is_contact_list = configs['is_contact_list']
        contact_unit_direction_list = configs['contact_unit_direction_list']
        contact_force_list = configs['contact_force_list']
        cable_tension_list = configs['cable_tension_list']
        idxContact_2_idxAll_list = configs['idxContact_2_idxAll_list']
        # print(contact_force_list)
        n_frames = len(Q_list)
        plotter = pv.Plotter(off_screen=True)
        frames = []
        xmin = min(self.vertices[:, 0].min()-0.01, self.arm_contour[:, 0].min()-0.01)
        xmax = max(self.vertices[:, 0].max()+0.01, self.arm_contour[:, 0].max()+0.01)
        ymin = min(self.vertices[:, 1].min()-0.01, self.arm_contour[:, 1].min()-0.01)
        ymax = max(self.vertices[:, 1].max()+0.01, self.arm_contour[:, 1].max()+0.01)
        zmin = min(self.vertices[:, 2].min()-0.01, self.arm_contour[:, 2].min()-0.01)
        zmax = max(self.vertices[:, 2].max()+0.01, self.arm_contour[:, 2].max()+0.01)
        for i in range(n_frames):
            plotter.clear()
            vert = self.Q_2_vertices(Q_list[i])
            cur_pulley_location = pulley_location_list[i]
            contact_unit_direction = contact_unit_direction_list[i]
            contact_forces = contact_force_list[i]
            idxContact_2_idxAll = idxContact_2_idxAll_list[i]
            mesh = pv.PolyData(vert)
            mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])
            plotter.add_mesh(mesh, color='lightblue', show_edges=True)
            # cable_tension = cable_tension_list[i]
            for j in range(self.nCable):
                unit_cable_vec = (cur_pulley_location[j] - vert[self.pp_idx[j]]) / np.linalg.norm(cur_pulley_location[j] - vert[self.pp_idx[j]])
                plotter.add_lines(np.array([cur_pulley_location[j], vert[self.pp_idx[j]]]), color='green')
                # plotter.add_arrows(vert[self.pp_idx[j]], cable_tension[j] * unit_cable_vec*0.1, color='green')
            for j in range(len(contact_forces)):
                idx_all = idxContact_2_idxAll[j]
                plotter.add_mesh(pv.Sphere(radius=0.001, center=vert[idx_all]), color='blue')
                if contact_forces[j] > 1e-6:
                    contact_force_v = contact_unit_direction[idx_all] * 0.01 * contact_forces[j]
                    # make an arrow mesh
                    arrow = pv.Arrow(vert[idx_all], contact_force_v, shaft_radius = 0.002, scale = 0.02)
                    plotter.add_mesh(arrow, color='blue')
            plotter.show_grid()
            plotter.show_axes()
            # add 8 points with xyz min and max
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmin, ymin, zmin]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmin, ymin, zmax]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmin, ymax, zmin]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmin, ymax, zmax]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmax, ymin, zmin]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmax, ymax, zmin]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmax, ymax, zmax]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmax, ymin, zmax]), color='red')
            grid = pv.StructuredGrid(self.arm_contour[:, 0], self.arm_contour[:, 1], self.arm_contour[:, 2])
            # add contact points
            
            # for j in range(self.num_vertices):
            #     idx_contact = 0
            #     if is_contact_list[i][j]:
            #         plotter.add_mesh(pv.Sphere(radius=0.01, center=vert[j]), color='blue')
            #         # add arrow in the contact direction
            #         print("contact force: ", contact_force_list[i])
            #         contact_force = contact_force_list[i][idx_contact]
            #         contact_unit_dir = contact_unit_direction_list[i][j].reshape((3,))

            #         contact_force_v = contact_unit_dir * 0.1 * contact_force
            #         plotter.add_arrows(vert[j], contact_force_v, color='blue')
            #         idx_contact += 1

            plotter.add_mesh(grid, color='orange', show_edges=True, opacity=0.5)
            # make axis equal
            plotter.set_position([0.15, -0.1, 0.5])
            plotter.set_viewup([0, -1, 0])
            plotter.set_focus([0, 0, 0])
            plotter.set_scale(1, 1, 1)
            frame = plotter.screenshot(return_img=True)
            frame_path = output_dir / f"frame_{i:03d}.png"
            # Convert RGB (PyVista output) to BGR (OpenCV format) and save
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), frame_bgr)
            frames.append(frame_path)
        frame = cv2.imread(str(frames[0]))
        height, width, _ = frame.shape

        # Initialize OpenCV video writer
        output_video = filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # Write frames to video
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            video_writer.write(frame)

        # Release the video writer
        video_writer.release()

        # Clean up temporary frame files
        for frame_path in frames:
            os.remove(frame_path)
        output_dir.rmdir()

        plotter.close()


    def animate_sequence(self, Q_list, fps=30):
        output_dir = Path("frames")
        output_dir.mkdir(exist_ok=True)
        n_frames = len(Q_list)
        plotter = pv.Plotter(off_screen=True)
        frames = []
        xmin = np.min(self.vertices[:, 0].min()-0.01, self.arm_contour[:, 0].min()-0.01)
        xmax = np.max(self.vertices[:, 0].max()+0.01, self.arm_contour[:, 0].max()+0.01)
        ymin = np.min(self.vertices[:, 1].min()-0.01, self.arm_contour[:, 1].min()-0.01)
        ymax = np.max(self.vertices[:, 1].max()+0.01, self.arm_contour[:, 1].max()+0.01)
        zmin = np.min(self.vertices[:, 2].min()-0.01, self.arm_contour[:, 2].min()-0.01)
        zmax = np.max(self.vertices[:, 2].max()+0.01, self.arm_contour[:, 2].max()+0.01)
        # plotter.open_movie("animation.mp4")
        for i in range(n_frames):
            plotter.clear()
            plotter.set_position([0.2, -0.5, 1.0])
            # plotter.set_viewup([1, -1, 1])
            # plotter.set_focus([0, 0, 0.2])
            plotter.set_scale(1, 1, 1)
            vert = self.vertices.copy()
            for j in range(self.num_vertices):
                vert[j] = Q_list[i][3*j:3*j+3].reshape((3,))
            mesh = pv.PolyData(vert)
            mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])
            plotter.add_mesh(mesh, color='lightblue', show_edges=True)
            for j in range(self.nCable):
                plotter.add_lines(np.array([self.pulley_location[j], vert[self.pp_idx[j]]]), color='green')
            plotter.show_grid()
            plotter.show_axes()
            
            # add 8 points with xyz min and max
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmin, ymin, zmin]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmin, ymin, zmax]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmin, ymax, zmin]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmin, ymax, zmax]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmax, ymin, zmin]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmax, ymax, zmin]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmax, ymax, zmax]), color='red')
            plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmax, ymin, zmax]), color='red')
            # make axis equal
            
            frame = plotter.screenshot(return_img=True)
            frame_path = output_dir / f"frame_{i:03d}.png"
            # Convert RGB (PyVista output) to BGR (OpenCV format) and save
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), frame_bgr)
            frames.append(frame_path)
        frame = cv2.imread(str(frames[0]))
        height, width, _ = frame.shape

        # Initialize OpenCV video writer
        output_video = "testsimulation.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # Write frames to video
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            video_writer.write(frame)

        # Release the video writer
        video_writer.release()

        # Clean up temporary frame files
        for frame_path in frames:
            os.remove(frame_path)
        output_dir.rmdir()

        plotter.close()

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
        for contact_idx in self.contact_vertex_idx:
            plotter.add_mesh(pv.Sphere(radius=0.001, center=self.cur_vertices[contact_idx]), color='green')

        # for feedback_idx in self.feedback_vertices:
        #     plotter.add_mesh(pv.Sphere(radius=0.001, center=self.cur_vertices[feedback_idx]), color='orange')

        plotter.add_mesh(mesh, show_edges=True)
        # annotate origin
        plotter.add_mesh(pv.Sphere(radius=0.01, center=[0, 0, 0]), color='red')

        # draw pp points
        for i in range(self.nCable):
            plotter.add_mesh(pv.Sphere(radius=0.001, center=self.cur_vertices[self.pp_idx[i]]), color='blue')
            
        plotter.show_grid()
        plotter.show_axes()
        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.show()
        return

    def draw_mesh_with_initial(self, mesh2draw):
        plotter = pv.Plotter()
        mesh = pv.PolyData(mesh2draw)
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
        plotter.show()
        return

    def draw_mesh_w_arm(self, Q, cur_pulley_location):
        xmin = min(self.vertices[:, 0].min()-0.01, self.arm_contour[:, 0].min()-0.01)
        xmax = max(self.vertices[:, 0].max()+0.01, self.arm_contour[:, 0].max()+0.01)
        ymin = min(self.vertices[:, 1].min()-0.01, self.arm_contour[:, 1].min()-0.01)
        ymax = max(self.vertices[:, 1].max()+0.01, self.arm_contour[:, 1].max()+0.01)
        zmin = min(self.vertices[:, 2].min()-0.01, self.arm_contour[:, 2].min()-0.01)
        zmax = max(self.vertices[:, 2].max()+0.01, self.arm_contour[:, 2].max()+0.01)
        plotter = pv.Plotter()
        vert = self.Q_2_vertices(Q)
        mesh = pv.PolyData(vert)
        mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])
        plotter.add_mesh(mesh, color='lightblue', show_edges=True)
        for j in range(self.nCable):
            plotter.add_lines(np.array([cur_pulley_location[j], vert[self.pp_idx[j]]]), color='green')
        plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmin, ymin, zmin]), color='red')
        plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmin, ymin, zmax]), color='red')
        plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmin, ymax, zmin]), color='red')
        plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmin, ymax, zmax]), color='red')
        plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmax, ymin, zmin]), color='red')
        plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmax, ymax, zmin]), color='red')
        plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmax, ymax, zmax]), color='red')
        plotter.add_mesh(pv.Sphere(radius=0.0001, center=[xmax, ymin, zmax]), color='red')
        grid = pv.StructuredGrid(self.arm_contour[:, 0], self.arm_contour[:, 1], self.arm_contour[:, 2])
        plotter.add_mesh(grid, color='orange', show_edges=True, opacity=0.5)
        plotter.show_grid()
        plotter.show_axes()
        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.show()
        return

    def draw_arm_contour(self):
        plotter = pv.Plotter()
        mesh = pv.PolyData(self.cur_vertices)
        mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])

        # plot the cable from pp to pulley
        for i in range(self.nCable):
            plotter.add_lines(np.array([self.pulley_location[i], self.cur_vertices[self.pp_idx[i]]]), color='black')

        # add fixed idx
        for i in self.fixed_idx:
            plotter.add_mesh(pv.Sphere(radius=0.001, center=self.cur_vertices[i]), color='blue')

        plotter.add_mesh(mesh, show_edges=True)
        # annotate origin
        plotter.add_mesh(pv.Sphere(radius=0.01, center=[0, 0, 0]), color='red')
        grid = pv.StructuredGrid(self.arm_contour[:, 0], self.arm_contour[:, 1], self.arm_contour[:, 2])
        plotter.add_mesh(grid, color='orange', show_edges=True, opacity=0.5)
        # make axis equal
        plotter.show_grid()
        plotter.show_axes()

        plotter.set_scale(1, 1, 1)
        plotter.show()

    def draw_config(self, config):
        Q = config['starting_Q']
        starting_elevation = config['starting_elevation']
        starting_horizontal_move = config['starting_horizontal_move']
        cur_pulley_location = self.pulley_location.copy()
        cur_pulley_location[:, 1] -= starting_elevation
        cur_pulley_location[:, 2] += starting_horizontal_move
        for i in range(self.num_vertices):
            Q[3*i+1, 0] -= starting_elevation
            Q[3*i+2, 0] += starting_horizontal_move
        self.draw_mesh_w_arm(Q, cur_pulley_location)

    def draw_contact(self, Q, is_contact, contact_distance, contact_unit_direction):
        plotter = pv.Plotter()
        vert = self.Q_2_vertices(Q)
        mesh = pv.PolyData(vert)
        mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])
        plotter.add_mesh(mesh, color='lightblue', show_edges=True)
        for j in range(self.nCable):
            plotter.add_lines(np.array([self.pulley_location[j], vert[self.pp_idx[j]]]), color='green')
        plotter.show_grid()
        plotter.show_axes()
        # add contact points
        for j in range(self.num_vertices):
            if is_contact[j]:
                plotter.add_mesh(pv.Sphere(radius=0.01, center=vert[j]), color='blue')
                # add arrow in the contact direction
                contact_force_v = contact_unit_direction[j] * 0.1
                plotter.add_arrows(vert[j], contact_force_v, color='blue')
        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.show()

    def draw_contact_inmoving(self, Q, cur_pulley_location, is_contact, contact_unit_direction, contact_forces, idxContact_2_idxAll):
        print("contact forces: ", contact_forces)
        plotter = pv.Plotter()
        vert = self.Q_2_vertices(Q)
        mesh = pv.PolyData(vert)
        mesh.faces = np.hstack([[4, *tet] for tet in self.tetrahedra])
        plotter.add_mesh(mesh, color='lightblue', show_edges=True)
        plotter.show_grid()
        plotter.show_axes()
        # add cables
        for j in range(self.nCable):
            plotter.add_lines(np.array([cur_pulley_location[j], vert[self.pp_idx[j]]]), color='green')
        for j in range(len(contact_forces)):
            idx_all = idxContact_2_idxAll[j]
            plotter.add_mesh(pv.Sphere(radius=0.001, center=vert[idx_all]), color='blue')
            if contact_forces[j] > 1e-6:
                contact_force_v = contact_unit_direction[idx_all] * 0.01 * contact_forces[j]
                # make an arrow mesh
                arrow = pv.Arrow(vert[idx_all], contact_force_v, shaft_radius = 0.002, scale = 0.02)
                plotter.add_mesh(arrow, color='blue')
        # add contact points
        # for j in range(self.num_vertices):
        #     if is_contact[j]:
        #         plotter.add_mesh(pv.Sphere(radius=0.001, center=vert[j]), color='blue')
        #         contact_f = contact_forces[idxContact_2_idxAll.index(j)]
        #         # add arrow in the contact direction
        #         contact_force_v = contact_unit_direction[j] * 0.01
        #         # make an arrow mesh
        #         arrow = pv.Arrow(vert[j], contact_force_v, shaft_radius = 0.02, scale = 0.02)
        #         plotter.add_mesh(arrow, color='blue')
        # add arm contour
        grid = pv.StructuredGrid(self.arm_contour[:, 0], self.arm_contour[:, 1], self.arm_contour[:, 2])
        plotter.add_mesh(grid, color='orange', show_edges=True, opacity=0.5)

        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.show()


    def vertices_2_Q(self, vertices):
        Q = np.zeros((3*self.num_vertices, 1))
        for i in range(self.num_vertices):
            Q[3*i:3*i+3, 0] = vertices[i].reshape((3,))
        return Q
    
    def Q_2_vertices(self, Q):
        vertices = self.vertices.copy()
        for i in range(self.num_vertices):
            vertices[i] = Q[3*i:3*i+3].reshape((3,))
        return vertices

    def Qmoving_2_Q(self, Q_moving):
        Q = self.vertices_2_Q(self.cur_vertices)
        for i in range(self.nMoving):
            idx_all = self.idxMoving_2_idxAll[i]
            Q[3*idx_all:3*idx_all+3, 0] = Q_moving[3*i:3*i+3, 0]
        return Q

if __name__ == "__main__":
    pickleFilename = 'models/palm_size3.pickle'
    palm = PalmSimulator(pickleFilename)
    # palm.generate_dataset("dataset.pkl", n_samples=1000)
    # print("pulley_location: ", palm.pulley_location)
    # palm.draw_mesh_with_initial(palm.original_verts)
    # exit(0)

    icl = palm.initial_cable_length.copy()
    print("initial cable length: ", icl)
    # palm.draw_arm_contour()
    tar_cable_length = [0.09140093898987767, 0.10785759469231798, 0.1122490790431223, 0.10040954588727102]
    verts, cable_tension = palm.FKD_static(tar_cable_length, palm.Q0, np.zeros((3*palm.num_vertices, 1)))
    palm.draw_mesh_with_initial(verts)

    
